
# Robot End-Effector Motion Control Using MPC


## Two-Layer Model Predictive Control for Unitree H1 Arm Control
This document describes the two-layer Model Predictive Control (MPC) formulation for controlling the Unitree H1 robot's arms. The architecture consists of a ${Trajectory MPC}$ layer, which generates optimal end-effector trajectories, and a $Kinematics MPC$ layer, which computes joint angles to track these trajectories. The formulation is implemented in \texttt{h1\_two\_layer\_mpc.py}.
![Implementation of MPC](./Implementation%20of%20MPC.svg)


### Description of the two-layer architecture
The two-layer MPC operates as follows:

#### Trajectory MPC:

Optimizes left and right end-effector positions $P_l$, $P_r \in R^3$ over a horizon $N_{traj} = 20$ with timestep $dt_{traj} = 0.1 s$. It ensures goal tracking, obstacle avoidance, workspace constraints, and trajectory smoothness.

#### Kinematics MPC: 

Computes joint angles $q \in R^8$ for the 8 arm joints over a horizon $N_{kin} = 5$ with timestep $dt_{kin} = 0.02 s$. It tracks the trajectory while enforcing joint limits, velocity constraints, and regularization.

The robot model is fixed-base with $n_q = 19$ degrees of freedom (DOF), but a reduced model $n_q^{red} = 8$ is used for MPC. End-effectors are at frames \texttt{left\_elbow\_link} and \texttt{right\_elbow\_link}.

### Trajectory MPC formulation

The Trajectory MPC optimizes end-effector positions to reach goal positions while avoiding obstacles and staying within the workspace.

#### Cost function for Trajectory MPC

The cost function $J_{traj}$ is:

$$J_{traj} = \sum_{k=0}^{N_{traj}-1} ( w_t [||P_l[k] - P_{g}^{l}||^2 + ||P_r[k] - P_g^r||^2] + w_s [||P_l[k] - P_l[k-1]||^2 + ||P_r[k] - P_r[k-1]||^2]) + w_f [||P_l[N_{traj}] - P_g^l||^2 + ||P_r[N_{traj}] - P_g^r||^2]$$

where:
$P_l[k], P_r[k] \in R^3$: Left/right end-effector positions at step $k$

$P_g^l, P_g^r \in R^3$: Left/right goal positions.

$w_t = 100$: Weight for position error.

$w_s = 0.1$: Weight for smoothness (for $k \geq 1$.

$w_f = 1000$: Weight for terminal error.

#### Constraints for Trajectory MPC

$\textbf{Initial Conditions}$

$P_l[0] = P_{init}^l, \quad P_r[0] = P_{init}^r$

where:

$P_{init}^l, P_{init}^r \in R^3$ are initial positions.

$\textbf{Obstacle Avoidance}$:

$$||P_l[k] - P_o||^2 \geq (r_o + d_{min})^2, \quad ||P_r[k] - P_o||^2 \geq (r_o + d_{\text{min}})^2, \quad \forall k$$


where:
$P_o \in R^3$ is the obstacle position, $r_o \in R$ is its radius, and $d_{min} = 0.05, \text{m}$.

$\textbf{Workspace Constraints}$:

$$||P_l[k] - c_{arm}^l||^2 \leq L_{max}^2, \quad ||P_r[k] - c_{arm}^r||^2 \leq L_{max}^2, \quad \forall k$$

where: 
$c_{arm}^l = ([0, 0.2, 0.05]^\top\), \(c_{arm}^r = [0, -0.2, 0.5]^\top\)$ are shoulder positions, and $L_{max} = 0.7 m $

## Kinematics MPC formulation

The Kinematics MPC computes joint angles to track the trajectory from the Trajectory MPC.

### Cost function for Kinematics MPC
The cost function \( J_{\text{kin}} \) is:

$$J_{kin} = \sum_{k=0}^{N_{kin}-1} \left[ w_p \left( ||FK_l(q[k]) - P_{traj}^l[k]||^2 + ||FK_r(q[k]) - P_{traj}^r[k]||^2 \right) + w_q ||q[k] - q[k-1]||^2 + w_r ||q[k]||^2 \right]$$

where:

$q[k] \in R^8$: Joint angles at step $k$.

$FK_{l}, FK_{r}$: Forward kinematics mapping $q[k]$ to left/right end-effector positions.

$P_{traj}^l[k], P_{traj}^r[k] \in R^3$: Desired trajectory positions.

$w_p = 100$: Weight for tracking error.

$w_q = 0$: Weight for joint velocity (disabled).

$w_r = 0.02$: Weight for regularization.


#### Constraints for Kinematics MPC

$\textbf{Initial Condition}$:

$$q[0] = q_{init}$$

where $q_{init} \in R^8$ are initial joint angles (mapped from full model $q \in R^{19}$.

$\textbf{Joint Limits}$:

$$q_{min} \leq q[k] \leq q_{max}, \quad \forall k$$

where: $q_{min}, q_{max} \in R^8$ are joint bounds.

$\textbf{Velocity Limits}$:

$$-\dot{q_{max}} \leq \frac{q[k] - q[k-1]}{dt_{kin}} \leq \dot{q}_{max}, \quad \forall k \geq 1$$

where: $\dot{q_{max}} = \pi . 1_8$, and $dt_{kin} = 0.02 s$

### Implementation notes
The robot model is loaded from a URDF (\texttt{h1.urdf}) with \( n_q = 19 \). A reduced model (\( n_q^{\text{red}} = 8 \)) includes 8 arm joints: 

$\texttt{left/right shoulder pitch/roll/yaw joint}$, and $\texttt{left/right elbow joint}$.

Joint angles are mapped to MuJoCo names $\texttt{Left/RightShoulderPitch}$, etc.) for compatibility with $\texttt{remap ik joints to motor}$.
Both MPC problems are solved using CasADi with the IPOPT solver.
