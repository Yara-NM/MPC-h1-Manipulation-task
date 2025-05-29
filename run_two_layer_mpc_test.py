import numpy as np
import time
import logging
from h1_two_layer_mpc_test import H1TwoLayerMPC
from command_interface import UnitreeJointController
from utils import map_motor_state
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
import matplotlib.pyplot as plt

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

# Initialize DDS communication
try:
    ChannelFactoryInitialize(1, "lo")    # Simulation mode
    # ChannelFactoryInitialize(0,"enp2s0")   # Real Robot mode
    time.sleep(0.5)
except Exception as e:
    logger.error(f"DDS communication failed: {e}")
    raise

# Initialize controllers
try:
    logger.info("Initializing H1TwoLayerMPC...")
    mpc = H1TwoLayerMPC(control_dt_traj=0.1, control_dt_kin=0.02, N_traj=5, N_kin=5)
    logger.info("Initializing UnitreeJointController...")
    joint_controller = UnitreeJointController(control_dt=0.002)
except Exception as e:
    logger.error(f"Initialization failed: {e}")
    raise

# Start control loop
logger.info("Starting control loop...")
joint_controller.start_control_loop()
time.sleep(0.5)

# Get initial joint state
try:
    q_init = map_motor_state(joint_controller.read_motor_state())
    logger.info(f"Initial joint state: {q_init}")
    q_init, P_init_l, P_init_r = mpc.update_state(q_init)
    logger.info(f"Initial state: q={q_init}\nP_init_l={P_init_l}\nP_init_r={P_init_r}")
except Exception as e:
    logger.error(f"Failed to read initial state: {e}")
    joint_controller.stop_control_loop()
    raise

# Define goal and obstacle
P_g_l = np.array([5, 2.4, 10.0])
P_g_r = np.array([5, -2.4, 10.0])
P_o = np.array([0.3, 0.15, 2.0])
r_o = 0.25
logger.info(f"Goal positions: Left={P_g_l}, Right={P_g_r}, Obstacle={P_o}, Radius={r_o}")

# Control loop
try:
    for k in range(100):
        logger.debug(f"Step {k}: Solving trajectory...")
        P_l_traj, P_r_traj = mpc.solve_trajectory(P_init_l, P_init_r, P_g_l, P_g_r, P_o, r_o)
        t_kin = np.arange(0, mpc.N_kin * mpc.control_dt_kin, mpc.control_dt_kin)
        t_traj = np.arange(0, mpc.N_traj * mpc.control_dt_traj, mpc.control_dt_traj)
        P_l_kin = [np.interp(t_kin, t_traj, [P_l_traj[i][j] for i in range(mpc.N_traj)]) for j in range(3)]
        P_r_kin = [np.interp(t_kin, t_traj, [P_r_traj[i][j] for i in range(mpc.N_traj)]) for j in range(3)]
        P_l_kin = np.array(P_l_kin).T[:mpc.N_kin]
        P_r_kin = np.array(P_r_kin).T[:mpc.N_kin]
        
        logger.debug(f"Step {k}: Solving kinematics...")
        joint_dict = mpc.solve_kinematics(q_init, P_l_kin, P_r_kin)
        logger.info(joint_dict)
        
        logger.debug(f"Step {k}: Sending motor commands: {joint_dict}")
        joint_controller.update_target_positions(joint_dict)
        print(f"step {k} - Joint_Dict: {joint_dict}")
        time.sleep(0.5)

        # Read and update joint state
        measured_joint_dict = map_motor_state(joint_controller.read_motor_state())
        q_init = measured_joint_dict
        q_init, P_init_l, P_init_r = mpc.update_state(q_init)
        logger.info(f"Step {k}: Updated EE positions: P_l={P_init_l}, P_r={P_init_r}")
        time.sleep(0.02)

except KeyboardInterrupt:
    logger.info("Control loop interrupted by user.")
except Exception as e:
    logger.error(f"Error in control loop: {e}")
finally:
    logger

#######################################################################
# Lists to store data for plotting
ee_left_positions = []
ee_right_positions = []
joint_trajectories = {name: [] for name in mpc.mujoco_joint_names}

# Rerun the simulation to collect data (no control updates)
q_init = map_motor_state(joint_controller.read_motor_state())
q_init, P_init_l, P_init_r = mpc.update_state(q_init)

for k in range(100):
    P_l_traj, P_r_traj = mpc.solve_trajectory(P_init_l, P_init_r, P_g_l, P_g_r, P_o, r_o)
    t_kin = np.arange(0, mpc.N_kin * mpc.control_dt_kin, mpc.control_dt_kin)
    t_traj = np.arange(0, mpc.N_traj * mpc.control_dt_traj, mpc.control_dt_traj)
    P_l_kin = [np.interp(t_kin, t_traj, [P_l_traj[i][j] for i in range(mpc.N_traj)]) for j in range(3)]
    P_r_kin = [np.interp(t_kin, t_traj, [P_r_traj[i][j] for i in range(mpc.N_traj)]) for j in range(3)]
    P_l_kin = np.array(P_l_kin).T[:mpc.N_kin]
    P_r_kin = np.array(P_r_kin).T[:mpc.N_kin]

    joint_dict = mpc.solve_kinematics(q_init, P_l_kin, P_r_kin)
    #convergence check
    # Log joint values for plotting
    for name in mpc.mujoco_joint_names:
        joint_trajectories[name].append(joint_dict[name])

    # Log EE positions
    q_init = joint_dict
    q_init, P_init_l, P_init_r = mpc.update_state(q_init)
    ee_left_positions.append(P_init_l)
    ee_right_positions.append(P_init_r)

# Convert lists to numpy arrays
ee_left_positions = np.array(ee_left_positions)
ee_right_positions = np.array(ee_right_positions)

# Plot end-effector trajectories
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot(ee_left_positions[:, 0], ee_left_positions[:, 1], ee_left_positions[:, 2], label='Left EE')
ax1.plot(ee_right_positions[:, 0], ee_right_positions[:, 1], ee_right_positions[:, 2], label='Right EE')
ax1.set_title("End-Effector Trajectories")
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Y (m)")
ax1.set_zlabel("Z (m)")
ax1.legend()
ax1.grid()

# Plot joint trajectories
fig2, axs = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
fig2.suptitle("Joint Angle Trajectories")
for i, name in enumerate(mpc.mujoco_joint_names):
    ax = axs[i // 2, i % 2]
    ax.plot(joint_trajectories[name], label=name)
    ax.set_ylabel("Angle (rad)")
    ax.legend()
    ax.grid()

axs[-1, 0].set_xlabel("Time Step")
axs[-1, 1].set_xlabel("Time Step")

plt.tight_layout()
plt.show()
