import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
import casadi as ca
import logging
import os
from pinocchio import RobotWrapper

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mpc_control.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class H1TwoLayerMPC:
    def __init__(self, control_dt_traj=0.1, control_dt_kin=0.02, N_traj=20, N_kin=5, urdf_path="/home/isr_lab/h1_arms_controller/Low_Level_Controller/meshes/h1.urdf"):
        """
        Initialize two-layer MPC controller for Unitree H1 arm control.

        Args:
            control_dt_traj (float): Trajectory control timestep (s).
            control_dt_kin (float): Kinematics control timestep (s).
            N_traj (int): Trajectory prediction horizon.
            N_kin (int): Kinematics prediction horizon.
            urdf_path (str): Path to the Unitree H1 URDF file.
        """
        self.control_dt_traj = control_dt_traj
        self.control_dt_kin = control_dt_kin
        self.N_traj = N_traj
        self.N_kin = N_kin
        self.urdf_path = urdf_path

        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",)) # path to root 
        urdf_file_path = os.path.join(base_path, "src/h1_description/urdf/h1.urdf")
        mesh_base_path = os.path.join(base_path, "src/h1_description/")

        if not os.path.exists(urdf_file_path):
                raise FileNotFoundError(f"URDF file not found at: {urdf_file_path}")
        
        # Manually replace package:// URIs
        with open(urdf_file_path, "r") as f:
            urdf_text = f.read().replace("package://h1_description/", mesh_base_path + "/")

        # Write to a temporary file for Pinocchio to parse
        temp_urdf_path = os.path.join(base_path, "temp_h1.urdf")
        with open(temp_urdf_path, "w") as f:
            f.write(urdf_text)

        
        # Load URDF
        self.robot = RobotWrapper.BuildFromURDF(temp_urdf_path, [mesh_base_path])
        self._build_reduced_robot_model()


        # Load URDF and initialize Pinocchio model
        if not os.path.exists(urdf_path):
            logger.error(f"URDF file not found at {urdf_path}")
            raise FileNotFoundError(f"URDF file not found at {urdf_path}")
        
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        logger.info(f"Loading URDF from {urdf_path}")
        logger.info(f"Model joint names: {self.model.names.tolist()}")
        logger.info(f"Model DOF (nq): {self.model.nq}")

        # Initialize frame indices
        self.left_elbow_frame_idx = self.reduced_robot.model.getFrameId("L_ee")
        self.right_elbow_frame_idx = self.reduced_robot.model.getFrameId("R_ee")
        # if self.left_elbow_frame_idx >= self.model.nframes or self.right_elbow_frame_idx >= self.model.nframes:
        #     logger.error("Frame left_elbow_link or right_elbow_link not found in URDF")
        #     raise ValueError("Frame left_elbow_link or right_elbow_link not found in URDF")
        # logger.info(f"Frame indices: left_elbow={self.left_elbow_frame_idx}, right_elbow={self.right_elbow_frame_idx}")

        # Define active joints
        self.joint_names = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint"
        ]
        self.mujoco_joint_names = [
            "LeftShoulderPitch", "LeftShoulderRoll", "LeftShoulderYaw", "LeftElbow",
            "RightShoulderPitch", "RightShoulderRoll", "RightShoulderYaw", "RightElbow"
        ]
        self.active_joint_ids = [self.reduced_robot.model.getJointId(name) for name in self.joint_names]
        self.active_joint_indices = []
        if len(self.active_joint_ids) != 8:
            logger.error(f"Expected 8 arm joints, found {len(self.active_joint_ids)}")
            raise ValueError(f"Expected 8 arm joints, found {len(self.active_joint_ids)}")

        # Build reduced model for MPC
        # self.reduced_model = pin.Model()
        # for jid in self.active_joint_ids:
        #     joint = self.model.joints[jid]
        #     self.reduced_model.addJoint(
        #         self.model.names[jid], joint, pin.SE3.Identity(),
        #         self.model.jointPlacements[jid]
        #     )
        #     self.active_joint_indices.append(joint.idx_q)
        # self.reduced_data = self.reduced_model.createData()
        

        self.reduced_robot_model = self.reduced_robot.model.createData()
        self.nq_reduced = self.reduced_robot.model.nq  # 8
        # CasADi model
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = cpin.Data(self.cmodel)

        # Cost function weights
        self.w_t = 10.0  # Trajectory position error
        self.w_s = 0.1    # Trajectory smoothness
        self.w_f = 10 # Terminal constraint
        self.w_p = 0.10  # Kinematics position error
        self.w_q = 2.0   # Kinematics joint difference
        self.w_r = 0.02   # Kinematics regularization
        self.d_min = 0.05 # Minimum distance to obstacle (m)

        # Workspace approximation
        self.L_max = 0.7  # Max arm length (m)
        self.c_arm_l = np.array([1000, 1000, 1000])  # Left shoulder
        self.c_arm_r = np.array([1000, 1000, 1000]) # Right shoulder

        # Joint limits and velocity bounds
        # self.q_min = np.array([-np.pi/2, -np.pi/4, -np.pi/2, -np.pi, -np.pi/2, -np.pi/4, -np.pi/2, -np.pi])
        # self.q_max = np.array([np.pi/2, np.pi/4, np.pi/2, 0.0, np.pi/2, np.pi/4, np.pi/2, 0.0])
        self.q_min = self.reduced_robot.model.lowerPositionLimit
        self.q_max = self.reduced_robot.model.upperPositionLimit
        self.qdot_max = np.ones(self.nq_reduced) * np.pi  # Velocity limits

        # Define obstacle position and radius numerically
        self.obstacle_position = np.array([0.3, 0.0, 1])  # [x, y, z] in meters
        self.obstacle_radius = 0.1  # radius in meters

        # Define goal end-effector positions numerically
        self.goal_position_left = np.array([0.4, 0.2, 0.4])   # Left arm goal [x, y, z]
        self.goal_position_right = np.array([0.4, -0.2, 0.4]) # Right arm goal [x, y, z]
        # index = self.reduced_robot.model.getJointId("left_shoulder_pitch_joint")
        # logger.info(f"left_shoulder_pitch_joint : {index}")
        # index = self.reduced_robot.model.getJointId("left_shoulder_roll_joint")
        # logger.info(f"left_shoulder_roll_joint : {index}")
        # index = self.reduced_robot.model.getJointId("left_shoulder_yaw_joint")
        # logger.info(f"left_shoulder_yaw_joint : {index}")
        # self.reduced_robot.model.getJointId("left_elbow_joint")
        # logger.info(f"left_elbow_joint : {index}")
        # index = self.reduced_robot.model.getJointId("right_shoulder_pitch_joint")
        # logger.info(f"right_shoulder_pitch_joint : {index}")
        # index= self.reduced_robot.model.getJointId("right_shoulder_roll_joint")
        # logger.info(f"right_shoulder_roll_joint : {index}")
        # index= self.reduced_robot.model.getJointId("right_shoulder_yaw_joint")
        # logger.info(f"right_shoulder_yaw_joint : {index}")
        # index= self.reduced_robot.model.getJointId("right_elbow_joint")
        # logger.info(f"right_elbow_joint : {index}")

        # Initialize MPC problems
        self._setup_mpc_trajectory()
        self._setup_mpc_kinematics()

    def _build_reduced_robot_model (self):
            self.joints_to_lock = [
                    "right_hip_roll_joint", "right_hip_pitch_joint", "right_knee_joint",
                    "left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint",
                    "left_hip_yaw_joint", "right_hip_yaw_joint",
                    "left_ankle_joint", "right_ankle_joint",
                    "torso_joint", 
                ]

            self.reduced_robot = self.robot.buildReducedRobot(
                    list_of_joints_to_lock=self.joints_to_lock,
                    reference_configuration=np.zeros(self.robot.model.nq)
                )

            self.locked_joint_config = {j: 0.0 for j in self.joints_to_lock}

            self.reduced_robot.model.addFrame(
                    pin.Frame('L_ee',
                            self.reduced_robot.model.getJointId('left_elbow_joint'),
                            pin.SE3(np.eye(3), np.array([0.25, 0, 0]).T),
                            pin.FrameType.OP_FRAME)
                )
            self.reduced_robot.model.addFrame(
                    pin.Frame('R_ee',
                            self.reduced_robot.model.getJointId('right_elbow_joint'),
                            pin.SE3(np.eye(3), np.array([0.25, 0, 0]).T),
                            pin.FrameType.OP_FRAME)
                )    

    def _setup_mpc_trajectory(self):
        """
        Set up MPC for trajectory generation.
        """
        P_l = [ca.SX.sym(f"P_l_{k}", 3) for k in range(self.N_traj)]
        P_r = [ca.SX.sym(f"P_r_{k}", 3) for k in range(self.N_traj)]
        P_init_l = ca.SX.sym("P_init_l", 3)
        P_init_r = ca.SX.sym("P_init_r", 3)
        P_g_l = ca.SX.sym("P_g_l", 3)
        P_g_r = ca.SX.sym("P_g_r", 3)
        P_o = ca.SX.sym("P_o", 3)
        r_o = ca.SX.sym("r_o")

        J = 0
        g = []
        lb = []
        ub = []
        for k in range(self.N_traj):
            J += self.w_t * (ca.sumsqr(P_l[k] - P_g_l) + ca.sumsqr(P_r[k] - P_g_r))
            if k > 0:
                J += self.w_s * (ca.sumsqr(P_l[k] - P_l[k-1]) + ca.sumsqr(P_r[k] - P_r[k-1]))
            g.append(ca.sumsqr(P_l[k] - P_o) >= (r_o + self.d_min)**2)
            g.append(ca.sumsqr(P_r[k] - P_o) >= (r_o + self.d_min)**2)
            g.append(ca.sumsqr(P_l[k] - self.c_arm_l) <= self.L_max**2)
            g.append(ca.sumsqr(P_r[k] - self.c_arm_r) <= self.L_max**2)
            lb.extend([1] * 2 + [0] * 2)
            ub.extend([ca.inf] * 4)
        J += self.w_f * (ca.sumsqr(P_l[-1] - P_g_l) + ca.sumsqr(P_r[-1] - P_g_r))

        # Initial condition
        g.append(P_l[0] - P_init_l)
        g.append(P_r[0] - P_init_r)
        lb.extend([0, 0, 0, 0, 0, 0])
        ub.extend([0, 0, 0, 0, 0, 0])

        nlp = {
            'x': ca.vertcat(*[ca.vertcat(P_l[k], P_r[k]) for k in range(self.N_traj)]),
            'f': J,
            'g': ca.vertcat(*g),
            'p': ca.vertcat(P_init_l, P_init_r, P_g_l, P_g_r, P_o, r_o)
        }
        self.traj_solver = ca.nlpsol('solver', 'ipopt', nlp, {
            'ipopt': {'print_level': 0, 'max_iter': 1000, 'tol': 1e-6}
        })
        self.traj_lbg = lb
        self.traj_ubg = ub
        self.traj_lbx = -ca.inf * np.ones(6 * self.N_traj)
        self.traj_ubx = ca.inf * np.ones(6 * self.N_traj)
        self.P_l_solution = [np.zeros(3) for _ in range(self.N_traj)]
        self.P_r_solution = [np.zeros(3) for _ in range(self.N_traj)]

    def _setup_mpc_kinematics(self):
        """
        Set up MPC for joint angle computation.
        """
        q = [ca.SX.sym(f"q_{k}", self.nq_reduced) for k in range(self.N_kin)]
        q_init = ca.SX.sym("q_init", self.nq_reduced)
        P_traj_l = [ca.SX.sym(f"P_traj_l_{k}", 3) for k in range(self.N_kin)]
        P_traj_r = [ca.SX.sym(f"P_traj_r_{k}", 3) for k in range(self.N_kin)]

        J = 0
        g = []
        lb = []
        ub = []
        for k in range(self.N_kin):
            cpin.framesForwardKinematics(self.cmodel, self.cdata, q[k])
            P_l_k = self.cdata.oMf[self.left_elbow_frame_idx].translation
            P_r_k = self.cdata.oMf[self.right_elbow_frame_idx].translation
            J += self.w_p * (ca.sumsqr(P_l_k - P_traj_l[k]) + ca.sumsqr(P_r_k - P_traj_r[k]))
            J += self.w_r * ca.sumsqr(q[k])
            if k > 0:
                delta_q = q[k] - q[k-1]
                J += self.w_q * ca.sumsqr(delta_q)
                g.append(delta_q / self.control_dt_kin <= self.qdot_max)
                g.append(delta_q / self.control_dt_kin >= -self.qdot_max)
                lb.extend([0] * self.nq_reduced * 2)
                ub.extend([ca.inf] * self.nq_reduced * 2)
            g.append(q[k] >= self.q_min)
            g.append(q[k] <= self.q_max)
            lb.extend([0] * self.nq_reduced * 2)
            ub.extend([ca.inf] * self.nq_reduced * 2)
        g.append(q[0] - q_init)
        lb.extend([0] * self.nq_reduced)
        ub.extend([0] * self.nq_reduced)

        nlp = {
            'x': ca.vertcat(*q),
            'f': J,
            'g': ca.vertcat(*g),
            'p': ca.vertcat(q_init, *[ca.vertcat(P_traj_l[k], P_traj_r[k]) for k in range(self.N_kin)])
        }
        self.kin_solver = ca.nlpsol('solver', 'ipopt', nlp, {
            'ipopt': {'print_level': 0, 'max_iter': 100, 'tol': 1e-6}
        })
        self.kin_lbg = lb
        self.kin_ubg = ub
        self.kin_lbx = ca.vertcat(*[self.q_min] * self.N_kin)
        self.kin_ubx = ca.vertcat(*[self.q_max] * self.N_kin)
        self.q_solution = [np.zeros(self.nq_reduced) for _ in range(self.N_kin)]

    def solve_trajectory(self, P_init_l, P_init_r, P_g_l=None, P_g_r=None, P_o=None, r_o=None):
        """
        Solve MPC for end-effector trajectory.

        Args:
            P_init_l, P_init_r (np.ndarray): Initial left/right end-effector positions (3,).
            P_g_l, P_g_r (np.ndarray): Goal left/right end-effector positions (3,).
            P_o (np.ndarray): Obstacle position (3,), optional.
            r_o (float): Obstacle radius (m), optional.

        Returns:
            tuple: (P_l_traj, P_r_traj) - Lists of trajectory points.
        """
        if P_g_l is None:
            P_g_l = self.goal_position_left
        if P_g_r is None:
            P_g_r = self.goal_position_right
        if P_o is None:
            P_o = self.obstacle_position
        if r_o is None:
            r_o = self.obstacle_radius

        logger.info(f"[Traj MPC] P_init_l: {P_init_l}, P_g_l: {P_g_l}")
        logger.info(f"[Traj MPC] P_init_r: {P_init_r}, P_g_r: {P_g_r}")

        # Initialize x0 as linear interpolation
        for k in range(self.N_traj):
            alpha = k / (self.N_traj - 1)
            self.P_l_solution[k] = (1 - alpha) * P_init_l + alpha * P_g_l
            self.P_r_solution[k] = (1 - alpha) * P_init_r + alpha * P_g_r

        p = ca.vertcat(P_init_l, P_init_r, P_g_l, P_g_r, P_o, r_o)
        x0 = ca.vertcat(*[ca.vertcat(self.P_l_solution[k], self.P_r_solution[k]) for k in range(self.N_traj)])

        try:
            sol = self.traj_solver(x0=x0, p=p, lbg=self.traj_lbg, ubg=self.traj_ubg, lbx=self.traj_lbx, ubx=self.traj_ubx)
            # print(f" ---------------{sol}--------------")
            x_sol = sol['x'].full().reshape(self.N_traj, 6)
            # print(x_sol)
            for k in range(self.N_traj):
                print(f"Time Step {k}: P_l{k}={x_sol[k,:3]}, P_r{k}={x_sol[k, 3:]}")
                self.P_l_solution[k] = x_sol[k, :3]
                self.P_r_solution[k] = x_sol[k, 3:]
            return self.P_l_solution, self.P_r_solution
        except Exception as e:
            logger.error(f"[Traj MPC Error] Optimization failed: {e}")
            return self.P_l_solution, self.P_r_solution


    def solve_kinematics(self, q_init, P_traj_l, P_traj_r):
        """
        Solve MPC for joint angles.

        Args:
            q_init (np.ndarray): Initial joint angles (nq=19,).
            P_traj_l, P_traj_r (np.array): Array of trajectory points for left/right end-effectors (N_kin x 3).
        Returns:
            dict: Joint angles for the first step.
        """
        # Map q_init to reduced model
        q_init_reduced = np.zeros(self.nq_reduced)
        for name in self.reduced_robot.model.names:
                if name in q_init:
                        joint = self.reduced_robot.model.joints[self.reduced_robot.model.getJointId(name)]
                        idx = joint.idx_q
                        q_init_reduced[idx] = q_init[name]
       

        # Initialize q_solution to current state for warm start
        for k in range(self.N_kin):
            self.q_solution[k] = q_init_reduced

        p = ca.vertcat(q_init_reduced, *[ca.vertcat(P_traj_l[k], P_traj_r[k]) for k in range(self.N_kin)])
        x0 = ca.vertcat(*self.q_solution)

        logger.info(f"[Kin MPC] q_init_reduced: {q_init_reduced}")
        logger.info(f"[Kin MPC] p norm: {np.linalg.norm(p.full())}")
        logger.info(f"[Kin MPC] x0 norm: {np.linalg.norm(x0.full())}")

        try:
            sol = self.kin_solver(x0=x0, p=p, lbg=self.kin_lbg, ubg=self.kin_ubg, lbx=self.kin_lbx, ubx=self.kin_ubx)
            q_sol = sol['x'].full().reshape(self.N_kin, self.nq_reduced)
            for k in range(self.N_kin):
                self.q_solution[k] = q_sol[k]
            return self._q_to_dict(self.q_solution[0])
        except Exception as e:
            logger.error(f"[Kin MPC Error] Optimization failed: {e}")
            return self._q_to_dict(q_init_reduced)


    def update_state(self, joint_input):
        """
        Update the robot state using joint positions.

        Args:
            joint_input: Dictionary of joint names and angles, or numpy array of size nq.
        
        Returns:
            tuple: (q_init, P_init_l, P_init_r) - Updated configuration and end-effector positions.
        """
        if isinstance(joint_input, dict):
            q_init = np.zeros(self.reduced_robot.model.nq)  # 8
            for name in self.reduced_robot.model.names:
                if name in joint_input:
                    try:
                        joint = self.reduced_robot.model.joints[self.reduced_robot.model.getJointId(name)]
                        idx = joint.idx_q
                        q_init[idx] = joint_input[name]
                    except Exception as e:
                        logger.warning(f"Skipping joint {name}: {e}")
                        continue
        else:
            q_init = np.array(joint_input)
            if q_init.shape[0] != self.reduced_robot.model.nq:
                logger.error(f"Expected {self.reduced_robot.model.nq} DOF, got {q_init.shape[0]}")
                raise ValueError(f"Expected {self.reduced_robot.model.nq} DOF, got {q_init.shape[0]}")
        
        pin.framesForwardKinematics(self.reduced_robot.model, self.reduced_robot_model, q_init)

        P_init_l = self.reduced_robot_model.oMf[self.left_elbow_frame_idx].translation
        P_init_r = self.reduced_robot_model.oMf[self.right_elbow_frame_idx].translation
        return q_init, P_init_l, P_init_r

    def _q_to_dict(self, q):
        """
        Convert reduced joint angle vector to dictionary.

        Args:
            q (np.ndarray): Joint angles (nq_reduced=8).

        Returns:
            dict: {joint_name: angle}
        """
        return {self.mujoco_joint_names[i]: q[i] for i in range(self.nq_reduced)}

    def _joint_dict_to_q(self, joint_dict):
        """
        Convert joint dictionary to joint angle vector.

        Args:
            joint_dict (dict): {joint_name: angle}

        Returns:
            np.ndarray: Joint angles (nq=19)
        """
        q = np.zeros(self.reduced_robot.model.nq)
        for i, name in enumerate(self.mujoco_joint_names):
            if name in joint_dict:
                idx = self.active_joint_indices[i]
                q[idx] = joint_dict[name]
        return q
    

    def solve_kinematics_J(self, q_init, P_l_kin, P_r_kin):
        """
        Solve inverse kinematics for desired end-effector positions.

        Args:
            q_init (np.array): Initial joint configuration (nq,).
            P_l_kin (np.array): Desired left end-effector positions (N_kin x 3).
            P_r_kin (np.array): Desired right end-effector positions (N_kin x 3).

        Returns:
            dict: Joint names and target angles.
        """
        joint_dict = {}
        q = self._joint_dict_to_q(q_init)

        for i in range(self.N_kin):
            # Desired end-effector positions
            p_l_des = P_l_kin[i]
            p_r_des = P_r_kin[i]

            # Numerical IK (simplified)
            for _ in range(10):  # Max iterations
                pin.framesForwardKinematics(self.reduced_robot.model, self.reduced_robot.data, q)
                pin.updateFramePlacements (self.reduced_robot.model , self.reduced_robot.data)
                p_l = self.reduced_robot.data.oMf[self.left_elbow_frame_idx].translation
                p_r = self.reduced_robot.data.oMf[self.right_elbow_frame_idx].translation

                # Error
                e_l = p_l_des - p_l
                e_r = p_r_des - p_r
                if np.linalg.norm(e_l) < 1e-3 and np.linalg.norm(e_r) < 1e-3:
                    break

                # Jacobian
                J_l = pin.computeFrameJacobian(self.reduced_robot.model, self.reduced_robot.data, q, self.left_elbow_frame_idx)[:3, :]
                J_r = pin.computeFrameJacobian(self.reduced_robot.model, self.reduced_robot.data, q, self.right_elbow_frame_idx)[:3, :]
                J = np.vstack([J_l, J_r])
                e = np.hstack([e_l, e_r])

                # Damped least squares
                damping = 0.01
                J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(6))
                dq = J_pinv @ e
                q += dq

        # Extract arm joint angles
        arm_joint_names = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint"
        ]
        for i, name in enumerate(self.reduced_robot.model.names[1:]):  # Skip 'universe'
            if name in arm_joint_names:
                joint_dict[name] = q[i]

        return joint_dict