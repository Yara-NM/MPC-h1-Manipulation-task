import time, math, csv, numpy as np, os
from datetime import datetime

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize 
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

import unitree_legged_const as h1

H1_NUM_MOTOR = 20

# Joint mapping as a dictionary: joint name -> motor index
joint_mapping = {
    # "RightHipYaw": 8,
    # "RightHipRoll": 0,
    # "RightHipPitch": 1,
    # "RightKnee": 2,
    # "RightAnkle": 11,
    # "LeftHipYaw": 7,
    # "LeftHipRoll": 3,
    # "LeftHipPitch": 4,
    # "LeftKnee": 5,
    # "LeftAnkle": 10,
    "WaistYaw": 6,
    "NotUsedJoint": 9,
    "RightShoulderPitch": 12,
    "RightShoulderRoll": 13,
    "RightShoulderYaw": 14,
    "RightElbow": 15,
    "LeftShoulderPitch": 16,
    "LeftShoulderRoll": 17,
    "LeftShoulderYaw": 18,
    "LeftElbow": 19,
}


# List of indices for weak motors (if needed for different PD gains)
weak_motors_indices = [11, 10, 12, 13, 14, 15, 16, 17, 18, 19]

class UnitreeJointController:
    def __init__(self, control_dt=0.02, results_dir = None , gamma = 0.6):
        self.control_dt_ = control_dt
        # Initialize target positions dictionary (radians); all start at 0.
        self.target_positions = {joint: 0.0 for joint in joint_mapping.keys()}
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None
        self.crc = CRC()
        self.time_ = 0.0
        self.running = True  # flag to control the loop

        # PD gains – you may tune these or expand to add feedforward torque.
        # for week motors // arms 
        self.kp_low_ = 35.0
        self.kd_low_ = 3.3
        # for strong motors
        self.kp_high_ = 200.0
        self.kd_high_ = 5.0

        # Adaptive parameters
        self.gravity = 9.81
        self.M_hat = {joint: 0.0 for joint in joint_mapping.keys()}
        self.gamma = gamma  # Adaptation gain

        # Torque and Speed limit (safety factor applied to max torque for elbow joint = 75 Nm)
        self.max_torque = 0.75 * 75.0  # 75 Nm * 75% safety = 56.25 Nm

        # Logs
        self.log_data = []
        self.logging_enabled = False

        self.results_dir = results_dir or os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = os.path.join(self.results_dir, f"joint_log_{timestamp}.csv")

        # intialize msgs
        self._init_low_cmd()
        self._init_topics()
        self.control_thread = None

    def _init_low_cmd(self):
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(H1_NUM_MOTOR):
            if i in weak_motors_indices:
                self.low_cmd.motor_cmd[i].mode = 0x01
            else:
                self.low_cmd.motor_cmd[i].mode = 0x0A
            self.low_cmd.motor_cmd[i].q = h1.PosStopF
            self.low_cmd.motor_cmd[i].dq = h1.VelStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def _init_topics(self, channel="lo"):
        # Initialize publisher for low-level commands.
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        # Initialize subscriber for low-level state.
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._low_state_handler, 10)

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

    def _compute_adaptive_torque(self, joint_name):
            if self.low_state is None or joint_name not in joint_mapping:
                return 0.0

            idx = joint_mapping[joint_name]
            q = self.low_state.motor_state[idx].q
            q_des = self.target_positions.get(joint_name, q)
            e = q_des - q
            tau_ff = self.M_hat[joint_name] * self.gravity * math.cos(q)
            tau_ff = np.clip(tau_ff, -self.max_torque, self.max_torque)
            self.M_hat[joint_name] += self.gamma * e * self.gravity * math.cos(q) * self.control_dt_
            return tau_ff


    def _low_state_handler(self, msg: LowState_):
        self.low_state = msg

    def write_motor_state(self):
        """
        This function is called repeatedly by the control loop.
        It reads the latest low_state and commands each joint to move
        toward its target position by taking a small interpolation step.
        """
        if self.low_state is None:
            return
        log_entry = {"time": time.time()}
        # For each joint, compute a new command based on the difference between
        # the current position and the target.

        for joint, idx in joint_mapping.items():
            current_q = self.low_state.motor_state[idx].q
            target_q = self.target_positions.get(joint, current_q)
            alpha = 1
            error = (target_q - current_q)
            # if error <= (self.max_joint_speed * self.control_dt_ ):
            #     step = error
            # else: step = self.max_joint_speed * self.control_dt_
            new_q = current_q + alpha * error
            self.low_cmd.motor_cmd[idx].q = new_q
            self.low_cmd.motor_cmd[idx].kp = self.kp_low_ if idx in weak_motors_indices else self.kp_high_
            self.low_cmd.motor_cmd[idx].kd = self.kd_low_ if idx in weak_motors_indices else self.kd_high_
            self.low_cmd.motor_cmd[idx].dq = 0.0
            # tau_ff = self._compute_adaptive_torque(joint)
            # self.low_cmd.motor_cmd[idx].tau = tau_ff
            self.low_cmd.motor_cmd[idx].tau = 0

            if self.logging_enabled:
                log_entry[f"{joint}_target"] = target_q
                log_entry[f"{joint}_pos"] = self.low_state.motor_state[idx].q
                log_entry[f"{joint}_vel"] = self.low_state.motor_state[idx].dq
                log_entry[f"{joint}_tau"] = self.low_state.motor_state[idx].tau_est

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

        # Save state values
        if self.logging_enabled:
            self.log_data.append(log_entry)


    def start_control_loop(self):
        """
        Starts a single control loop that continuously writes motor commands.
        """
        self.running = True
        self.control_thread = RecurrentThread(
            interval=self.control_dt_, target=self.write_motor_state, name="control_loop"
        )
        self.control_thread.Start()

    def stop_control_loop(self):
        """
        Stops the control loop.
        """
        self.running = False
        # Depending on your RecurrentThread implementation, you might also call a stop or join method.
        if self.control_thread is not None:
            self.control_thread.Wait()  # Proper way to request loop exit and join

    def read_raw_motor_state(self):
        """
        Returns a dictionary of the current motor positions (radians).
        """
        if self.low_state is None:
            return {}
        return {joint: self.low_state.motor_state[idx].q_raw for joint, idx in joint_mapping.items()}

    
    def read_motor_state(self):
        """
        Returns a dictionary of the current motor positions (radians).
        """
        if self.low_state is None:
            return {}
        return {joint: self.low_state.motor_state[idx].q for joint, idx in joint_mapping.items()}


    def update_target_positions(self, new_targets: dict):
        """
        Updates the target positions dictionary.
        """
        for joint, target in new_targets.items():
            if joint in self.target_positions:
                self.target_positions[joint] = target
            else:
                print(f"Warning: {joint} not found in target positions.")

    def print_imu_state(self):
        """
        Returns the current IMU sensor values: RPY, gyroscope, and accelerometer.
        return: 
        {
            "rpy": (roll, pitch, yaw),
            "gyroscope": (x, y, z),
            "accelerometer": (x, y, z)
        }
        """
        if self.low_state and hasattr(self.low_state, "imu_state"):
            imu = self.low_state.imu_state
            rpy = imu.rpy
            gyro = imu.gyroscope
            accel = imu.accelerometer
            imu_data = {
                "rpy": tuple(rpy),
                "gyroscope": tuple(gyro),
                "accelerometer": tuple(accel)
            }
            print(f"[IMU RPY]         Roll: {rpy[0]:.4f}, Pitch: {rpy[1]:.4f}, Yaw: {rpy[2]:.4f}")
            print(f"[IMU Gyroscope]   X: {gyro[0]:.4f}, Y: {gyro[1]:.4f}, Z: {gyro[2]:.4f}")
            print(f"[IMU Accelerometer] X: {accel[0]:.4f}, Y: {accel[1]:.4f}, Z: {accel[2]:.4f}")
            return imu_data
        else:
            print("IMU state not available yet.")
            return None


    def read_torque_state(self):
        """
        Returns a dictionary of the current motor positions (radians).
        """
        if self.low_state is None:
            return {}
        return {joint: self.low_state.motor_state[idx].tau_est for joint, idx in joint_mapping.items()}
  
    def read_vel_state(self):
            """
            Returns a dictionary of the current motor positions (radians).
            """
            if self.low_state is None:
                return {}
            return {joint: self.low_state.motor_state[idx].dq for joint, idx in joint_mapping.items()}

    
    def enable_logging(self, filename = None):
       self.logging_enabled = True
       if filename is not None:
            self.log_filename = filename
       self.log_data = []

    
    def save_log_to_csv(self):
        if not self.logging_enabled or not self.log_data:
            return
        keys = self.log_data[0].keys()
        with open(self.log_filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.log_data)
        print(f"[INFO] Log saved to {self.log_filename}")

    def apply_torques(self, torque_dict):
        """
        Apply direct torque commands to motors (torque control mode).
        All position-related terms are disabled.
        """
        if self.low_state is None:
            return

        for joint, idx in joint_mapping.items():
            tau = torque_dict.get(joint, 0.0)
            self.low_cmd.motor_cmd[idx].mode = 0x01  # torque control
            self.low_cmd.motor_cmd[idx].q = 0.0
            self.low_cmd.motor_cmd[idx].dq = 0.0
            self.low_cmd.motor_cmd[idx].kp = 0.0
            self.low_cmd.motor_cmd[idx].kd = 0.0
            self.low_cmd.motor_cmd[idx].tau = tau

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)



if __name__ == "__main__": 

    # ChannelFactoryInitialize(0,"enp2s0")
    ChannelFactoryInitialize(1,"lo")
    time.sleep (0.50)
    # Example: create the arm controller, update target positions, and start control. 
    controller = UnitreeJointController(control_dt=0.01) 
    print("Press ENTER to start arm control...") 
    input() # For instance, command the left shoulder pitch to 0.5 rad and right elbow to -0.3 rad. 

    controller.start_control_loop() 
    time.sleep (1)
    controller.update_target_positions({ "LeftShoulderPitch": 0.5, "RightElbow": -0.3 }) 
    time.sleep(2)
    controller.update_target_positions({ "LeftShoulderPitch": 0.0, "RightElbow": 0 }) 
    time.sleep(2)

    controller.update_target_positions({ "LeftShoulderPitch": 0.5, "RightElbow": -0.3 }) 
    time.sleep(2)
    controller.update_target_positions({ "LeftShoulderPitch": 0.0, "RightElbow": 0 }) 
    time.sleep(2)
    controller.update_target_positions({ "LeftShoulderPitch": 0.5, "RightElbow": -0.3 }) 
    time.sleep(2)
    controller.update_target_positions({ "LeftShoulderPitch": 0.0, "RightElbow": 0 }) 
    time.sleep(2)
    # Stop the control loop
    controller.stop_control_loop()
    print("Control loop stopped. Done.")
