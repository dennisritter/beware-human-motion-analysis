import warnings
import numpy as np

""" This module contains functions to convert different rotation and angle representations from and to each other."""


def medical_from_euler(euler_sequence, euler_angles, joint_name, warnings=False) -> 'np.array':
    """Returns an np.array of medical joint angles in degrees seperated in Flexion/Extension, Abduction/Adduction and Internal/External Rotation from the specified euler sequence.

    Args:
        euler_sequence (str):   The euler sequence to map the medical angles from as a string.
            example: 'xyz'
        euler_angles (list):    The angles of the euler rotations about the corresponding axes of euler_sequence.
        joint_name (str):       The name_id of the joint to retrieve medical angles for.
    """
    def handle_unsupported_joint(joint_name):
        if warnings:
            warnings.warn(f"medical_from_euler: Medical angles for joint_name = '{joint_name}' are not supported.\nReturning [None, None, None].")
        return np.array([None, None, None])

    def handle_unsupported_sequence(euler_sequence, supported_sequences, joint_name):
        if warnings:
            warnings.warn(f"medical_from_euler: The specified Euler sequence '{euler_sequence}' is not supported for joint_name = '{joint_name}'.\nSupported sequences are {supported_euler_sequences}.\nReturning [None, None, None].")
        return np.array([None, None, None])
    # 1. Determine indices of angles for roations about x,y,z axes in euler_angles
    euler_x_idx, euler_y_idx, euler_z_idx = euler_sequence.find('x'), euler_sequence.find('y'), euler_sequence.find('z')

    if joint_name == 'head':
        handle_unsupported_joint(joint_name)
    elif joint_name == 'neck':
        handle_unsupported_joint(joint_name)
    elif joint_name == 'shoulder_l':
        supported_euler_sequences = ['xyz', 'yxz']
        if euler_sequence not in supported_euler_sequences:
            handle_unsupported_sequence(euler_sequence, supported_euler_sequences, joint_name)
        flex = euler_angles[euler_x_idx]
        abd = euler_angles[euler_y_idx]
        # Z-rotation is meaningless at the moment, so ignore it and add no value
        inter = None
        return np.array([flex, abd, inter])
    elif joint_name == 'shoulder_r':
        supported_euler_sequences = ['xyz', 'yxz']
        if euler_sequence not in supported_euler_sequences:
            handle_unsupported_sequence(euler_sequence, supported_euler_sequences, joint_name)
        flex = euler_angles[euler_x_idx]
        abd = euler_angles[euler_y_idx] * -1
        # Z-rotation is meaningless at the moment, so ignore it and add no value
        inter = None
        return np.array([flex, abd, inter])
    elif joint_name == 'elbow_l':
        supported_euler_sequences = ['zxz']
        if euler_sequence not in supported_euler_sequences:
            handle_unsupported_sequence(euler_sequence, supported_euler_sequences, joint_name)
        flex = euler_angles[euler_x_idx]
        # Y-rotation is meaningless at the moment, so ignore it and add no value
        abd = None
        # Z-rotation is meaningless at the moment, so ignore it and add no value
        inter = None
        return np.array([flex, abd, inter])
    elif joint_name == 'elbow_r':
        supported_euler_sequences = ['zxz']
        if euler_sequence not in supported_euler_sequences:
            handle_unsupported_sequence(euler_sequence, supported_euler_sequences, joint_name)
        flex = euler_angles[euler_x_idx]
        # Y-rotation is meaningless at the moment, so ignore it and add no value
        abd = None
        # Z-rotation is meaningless at the moment, so ignore it and add no value
        inter = None
        return np.array([flex, abd, inter])
    elif joint_name == 'wrist_l':
        handle_unsupported_joint(joint_name)
    elif joint_name == 'wrist_r':
        handle_unsupported_joint(joint_name)
    elif joint_name == 'torso':
        handle_unsupported_joint(joint_name)
    elif joint_name == 'pelvis':
        handle_unsupported_joint(joint_name)
    elif joint_name == 'hip_l':
        supported_euler_sequences = ['xyz', 'yxz']
        if euler_sequence not in supported_euler_sequences:
            handle_unsupported_sequence(euler_sequence, supported_euler_sequences, joint_name)
        flex = euler_angles[euler_x_idx]
        abd = euler_angles[euler_y_idx]
        # Z-rotation is meaningless at the moment, so ignore it and add no value
        inter = None
        return np.array([flex, abd, inter])
    elif joint_name == 'hip_r':
        supported_euler_sequences = ['xyz', 'yxz']
        if euler_sequence not in supported_euler_sequences:
            handle_unsupported_sequence(euler_sequence, supported_euler_sequences, joint_name)
        flex = euler_angles[euler_x_idx]
        abd = euler_angles[euler_y_idx] * -1
        # Z-rotation is meaningless at the moment, so ignore it and add no value
        inter = None
        return np.array([flex, abd, inter])
    elif joint_name == 'knee_l':
        supported_euler_sequences = ['zxz']
        if euler_sequence not in supported_euler_sequences:
            handle_unsupported_sequence(euler_sequence, supported_euler_sequences, joint_name)
        flex = euler_angles[euler_x_idx]
        # Y-rotation is meaningless at the moment, so ignore it and add no value
        abd = None
        # Z-rotation is meaningless at the moment, so ignore it and add no value
        inter = None
        return np.array([flex, abd, inter])
    elif joint_name == 'knee_r':
        supported_euler_sequences = ['zxz']
        if euler_sequence not in supported_euler_sequences:
            handle_unsupported_sequence(euler_sequence, supported_euler_sequences, joint_name)
        flex = euler_angles[euler_x_idx]
        # Y-rotation is meaningless at the moment, so ignore it and add no value
        abd = None
        # Z-rotation is meaningless at the moment, so ignore it and add no value
        inter = None
        return np.array([flex, abd, inter])
    elif joint_name == 'ankle_l':
        handle_unsupported_joint(joint_name)
    elif joint_name == 'ankle_r':
        handle_unsupported_joint(joint_name)
    # If nothing has been returned yet, the specified joint_name wasn't supported.
    handle_unsupported_joint(joint_name)
    return np.array([None, None, None])


def medical_from_euler_batch(euler_sequence, euler_angles, joint_name, warnings=False) -> 'np.array':
    """Returns an np.array of medical joint angles in degrees seperated in Flexion/Extension, Abduction/Adduction and Internal/External Rotation from the specified euler sequence.

    Args:
        euler_sequence (str):   The euler sequence to map the medical angles from as a string.
            example: 'xyz'
        euler_angles (list):    The angles of the euler rotations about the corresponding axes of euler_sequence.
        joint_name (str):       The name_id of the joint to retrieve medical angles for.
    """
    def handle_unsupported_joint(joint_name):
        if warnings:
            warnings.warn(f"medical_from_euler: Medical angles for joint_name = '{joint_name}' are not supported.\nReturning [None, None, None].")
        return np.array([None, None, None])

    def handle_unsupported_sequence(euler_sequence, supported_sequences, joint_name):
        if warnings:
            warnings.warn(f"medical_from_euler: The specified Euler sequence '{euler_sequence}' is not supported for joint_name = '{joint_name}'.\nSupported sequences are {supported_euler_sequences}.\nReturning [None, None, None].")
        return np.array([None, None, None])
    # 1. Determine indices of angles for roations about x,y,z axes in euler_angles
    euler_x_idx, euler_y_idx, euler_z_idx = euler_sequence.find('x'), euler_sequence.find('y'), euler_sequence.find('z')

    # Init result array which will be filled and returned afterwards
    n_frames = len(euler_angles)
    n_angle_types = 3
    med_angles = np.full((n_frames, n_angle_types), None)

    if joint_name == 'head':
        handle_unsupported_joint(joint_name)
    elif joint_name == 'neck':
        handle_unsupported_joint(joint_name)
    elif joint_name == 'shoulder_l':
        supported_euler_sequences = ['xyz', 'yxz']
        if euler_sequence not in supported_euler_sequences:
            handle_unsupported_sequence(euler_sequence, supported_euler_sequences, joint_name)
        flex = euler_angles[:, euler_x_idx]
        abd = euler_angles[:, euler_y_idx]
        # Z-rotation is meaningless at the moment, so ignore it and add no value
        inter = None
        med_angles[:, 0] = flex
        med_angles[:, 1] = abd
        return med_angles
    elif joint_name == 'shoulder_r':
        supported_euler_sequences = ['xyz', 'yxz']
        if euler_sequence not in supported_euler_sequences:
            handle_unsupported_sequence(euler_sequence, supported_euler_sequences, joint_name)
        flex = euler_angles[:, euler_x_idx]
        abd = euler_angles[:, euler_y_idx] * -1
        # Z-rotation is meaningless at the moment, so ignore it and add no value
        inter = None
        med_angles[:, 0] = flex
        med_angles[:, 1] = abd
        return med_angles
    elif joint_name == 'elbow_l':
        supported_euler_sequences = ['zxz']
        if euler_sequence not in supported_euler_sequences:
            handle_unsupported_sequence(euler_sequence, supported_euler_sequences, joint_name)
        flex = euler_angles[:, euler_x_idx]
        # Y-rotation is meaningless at the moment, so ignore it and add no value
        abd = None
        # Z-rotation is meaningless at the moment, so ignore it and add no value
        inter = None
        med_angles[:, 0] = flex
        return med_angles
    elif joint_name == 'elbow_r':
        supported_euler_sequences = ['zxz']
        if euler_sequence not in supported_euler_sequences:
            handle_unsupported_sequence(euler_sequence, supported_euler_sequences, joint_name)
        flex = euler_angles[:, euler_x_idx]
        # Y-rotation is meaningless at the moment, so ignore it and add no value
        abd = None
        # Z-rotation is meaningless at the moment, so ignore it and add no value
        inter = None
        med_angles[:, 0] = flex
        return med_angles
    elif joint_name == 'wrist_l':
        handle_unsupported_joint(joint_name)
    elif joint_name == 'wrist_r':
        handle_unsupported_joint(joint_name)
    elif joint_name == 'torso':
        handle_unsupported_joint(joint_name)
    elif joint_name == 'pelvis':
        handle_unsupported_joint(joint_name)
    elif joint_name == 'hip_l':
        supported_euler_sequences = ['xyz', 'yxz']
        if euler_sequence not in supported_euler_sequences:
            handle_unsupported_sequence(euler_sequence, supported_euler_sequences, joint_name)
        flex = euler_angles[:, euler_x_idx]
        abd = euler_angles[:, euler_y_idx]
        # Z-rotation is meaningless at the moment, so ignore it and add no value
        inter = None
        med_angles[:, 0] = flex
        med_angles[:, 1] = abd
        return med_angles
    elif joint_name == 'hip_r':
        supported_euler_sequences = ['xyz', 'yxz']
        if euler_sequence not in supported_euler_sequences:
            handle_unsupported_sequence(euler_sequence, supported_euler_sequences, joint_name)
        flex = euler_angles[:, euler_x_idx]
        abd = euler_angles[:, euler_y_idx] * -1
        # Z-rotation is meaningless at the moment, so ignore it and add no value
        inter = None
        med_angles[:, 0] = flex
        med_angles[:, 1] = abd
        return med_angles
    elif joint_name == 'knee_l':
        supported_euler_sequences = ['zxz']
        if euler_sequence not in supported_euler_sequences:
            handle_unsupported_sequence(euler_sequence, supported_euler_sequences, joint_name)
        flex = euler_angles[:, euler_x_idx]
        # Y-rotation is meaningless at the moment, so ignore it and add no value
        abd = None
        # Z-rotation is meaningless at the moment, so ignore it and add no value
        inter = None
        med_angles[:, 0] = flex
        return med_angles
    elif joint_name == 'knee_r':
        supported_euler_sequences = ['zxz']
        if euler_sequence not in supported_euler_sequences:
            handle_unsupported_sequence(euler_sequence, supported_euler_sequences, joint_name)
        flex = euler_angles[:, euler_x_idx]
        # Y-rotation is meaningless at the moment, so ignore it and add no value
        abd = None
        # Z-rotation is meaningless at the moment, so ignore it and add no value
        inter = None
        med_angles[:, 0] = flex
        return med_angles
    elif joint_name == 'ankle_l':
        handle_unsupported_joint(joint_name)
    elif joint_name == 'ankle_r':
        handle_unsupported_joint(joint_name)
    # If nothing has been returned yet, the specified joint_name wasn't supported.
    handle_unsupported_joint(joint_name)
    return med_angles
