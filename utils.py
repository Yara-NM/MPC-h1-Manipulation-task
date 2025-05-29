# h1_control/utils.py

import math
import numpy as np
# Map FK/IK joint names to motor controller names
URDF_to_motor_joint_names_map = {
    "left_shoulder_pitch_joint": "LeftShoulderPitch",
    "left_shoulder_roll_joint": "LeftShoulderRoll",
    "left_shoulder_yaw_joint": "LeftShoulderYaw",
    "left_elbow_joint": "LeftElbow",
    "right_shoulder_pitch_joint": "RightShoulderPitch",
    "right_shoulder_roll_joint": "RightShoulderRoll",
    "right_shoulder_yaw_joint": "RightShoulderYaw",
    "right_elbow_joint": "RightElbow",
}


def rotation_matrix_to_quaternion(R):
    trace = np.trace(R)
    if trace > 0:
        S = math.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return np.array([x, y, z, w])

def rotvec_to_quat(rot_vec):
    angle = np.linalg.norm(rot_vec)
    if angle < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = rot_vec / angle
    sin_half_angle = math.sin(angle / 2.0)
    cos_half_angle = math.cos(angle / 2.0)
    return np.array([axis[0] * sin_half_angle, axis[1] * sin_half_angle, axis[2] * sin_half_angle, cos_half_angle])

def remap_ik_joints_to_motor(joint_dict):
    return {
        URDF_to_motor_joint_names_map[k]: v
        for k, v in joint_dict.items()
        if k in URDF_to_motor_joint_names_map
    }


def map_motor_state(motor_state):
    motor_to_fk_mapping = {
        "RightHipYaw": "right_hip_yaw_joint",
        "RightHipRoll": "right_hip_roll_joint",
        "RightHipPitch": "right_hip_pitch_joint",
        "RightKnee": "right_knee_joint",
        "RightAnkle": "right_ankle_joint",
        "LeftHipYaw": "left_hip_yaw_joint",
        "LeftHipRoll": "left_hip_roll_joint",
        "LeftHipPitch": "left_hip_pitch_joint",
        "LeftKnee": "left_knee_joint",
        "LeftAnkle": "left_ankle_joint",
        "WaistYaw": "torso_joint",
        "NotUsedJoint": "not_used_joint",
        "RightShoulderPitch": "right_shoulder_pitch_joint",
        "RightShoulderRoll": "right_shoulder_roll_joint",
        "RightShoulderYaw": "right_shoulder_yaw_joint",
        "RightElbow": "right_elbow_joint",
        "LeftShoulderPitch": "left_shoulder_pitch_joint",
        "LeftShoulderRoll": "left_shoulder_roll_joint",
        "LeftShoulderYaw": "left_shoulder_yaw_joint",
        "LeftElbow": "left_elbow_joint"
    }

    return {
        fk_name: motor_state[motor_name]
        for motor_name, fk_name in motor_to_fk_mapping.items()
        if motor_name in motor_state
    }

