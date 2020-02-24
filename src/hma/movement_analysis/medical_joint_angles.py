import warnings
import numpy as np
from hma.movement_analysis.enums.angle_types import AngleTypes
from hma.movement_analysis.models.sequence import Sequence
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
            warnings.warn(
                f"medical_from_euler: The specified Euler sequence '{euler_sequence}' is not supported for joint_name = '{joint_name}'.\nSupported sequences are {supported_euler_sequences}.\nReturning [None, None, None]."
            )
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
            warnings.warn(
                f"medical_from_euler: The specified Euler sequence '{euler_sequence}' is not supported for joint_name = '{joint_name}'.\nSupported sequences are {supported_euler_sequences}.\nReturning [None, None, None]."
            )
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


def _get_joint_start_dist_x(joint_positions_x):
    """Returns the sum of distances of all frames to the starting x-position.

        Args:
            joint_positions_x (np.ndarray): The 3-D euclidean x-positions of a joint node.
        """
    return np.sum(np.absolute(np.absolute(joint_positions_x) - abs(joint_positions_x[0])))


def _get_joint_start_dist_y(joint_positions_y):
    """Returns the sum of distances of all frames to the starting y-position.

    Args:
        joint_positions_y (np.ndarray): The 3-D euclidean y-positions of a joint node.
    """
    return np.sum(np.absolute(np.absolute(joint_positions_y) - abs(joint_positions_y[0])))


def _fix_range_exceedance(joint_indices: list, joint_angles: 'np.ndarray', thresh: int):
    """Fixes range exceedance of joint angles that exceed the range of [-180, 180] degrees and returns the fixed angles.

    Medical joint angles derived from Euler sequences only support a range form -180 to 180 degrees.
    Some motions, e.g. shoulder flexions or abductions may result in angles that exceed this range. As a result of this exceedance,
    the calculated angle will jump to the other end of the range (e.g. from 180 to -180).
    As this issue affects other analytical parts of exercise evaluations, the range must be exceeded manually.
    Therefore, this function uses a defined threshold (thresh parameter) to detect abrupt changes of joint angles
    and interprets them a range exceedance. Depending on the type of exceedance (from -180 to 180 XOR 180 to -180)
    360° degrees are added to/subtracted from the exceeding joint angles.

    Args:
        joint_indices (list): A list of indices, each representing a specific joint/body_part in a motion sequence.
        joint_angles (np.ndarray): A 3-D list of joint angles of shape (n_frames, n_body_parts, n_angle_types).
        thresh (int): The threshold that determines at which 1-frame angle change a fix is applied. 

    """
    for joint_idx in joint_indices:
        for angle_type in range(len([AngleTypes.FLEX_EX, AngleTypes.AB_AD])):
            # Get Flexion or Abduction angles for current Ball Joint
            angles = joint_angles[:, joint_idx, angle_type]
            # Get distances between angles
            distances = np.diff(angles)
            # Get indices where the absolute distance exceeds defined threshold
            jump = np.argwhere((distances > thresh) | (distances < -thresh))

            # Iterate over all jump indices, with a step size of 2 as we want to change a whole slice
            # E.g.: The angle jumps from 180 to -180 and back to 180 later on. We want to fix all angles in between.
            for i in range(0, len(jump), 2):
                # The index in 'angles' after the jump happened
                # As len(distances) == len(angles) - 1, there is always a angles[jump_idx+1] value.
                jump_post_idx = jump[i][0] + 1
                if distances[jump[i][0]] < -180:
                    # If there is another jump_idx
                    if jump[-1][0] != jump[i][0]:
                        jump_back_idx = jump[i + 1][0] + 1
                        angles[jump_post_idx:jump_back_idx + 1] += 360
                    # Else fix all angles until end
                    else:
                        angles[jump_post_idx:] += 360

                elif distances[jump[i][0]] > 180:
                    # If there is another jump_idx
                    if jump[-1][0] != jump[i][0]:
                        jump_back_idx = jump[i + 1][0]
                        angles[jump_post_idx:jump_back_idx + 1] -= 360
                    # Else fix all angles until end
                    else:
                        angles[jump_post_idx:] -= 360
            joint_angles[:, joint_idx, angle_type] = angles
    return joint_angles


def calc_joint_angles(seq: 'Sequence') -> np.ndarray:
    """Returns a 3-D list of joint angles for all frames, body parts and angle types of a sequence."""
    n_frames = len(seq.positions)
    n_body_parts = len(seq.scene_graph.nodes)
    n_angle_types = 3
    body_parts = seq.body_parts

    joint_angles = np.full((n_frames, n_body_parts, n_angle_types), None)
    ball_joints = ['shoulder_l', 'shoulder_r', 'hip_l', 'hip_r']
    non_ball_joints = ['elbow_l', 'elbow_r', 'knee_l', 'knee_r']

    for node in seq.scene_graph.nodes:
        if 'angles' in seq.scene_graph.nodes[node].keys():
            angles_dict = seq.scene_graph.nodes[node]['angles']
            if node in ball_joints:
                # TODO: Hacky/Naive/Simple solution.. ==> How can we determine order of motions more reliably?
                # * NOTE:   We assume, that the axis, on which the child node moved more, gives information about whether
                # *         the current nodes' performed motion has been a flexion followed by an abduction or vice versa.
                # *         If start_dist_x < start_dist_y, more motion occured 'frontal', which indicates a flexion->abduction order (=> Use XYZ-Euler).
                # *         If start_dist_x > start_dist_y, more motion occured 'sideways', which indicates a abduction->flexion order (=> Use YXZ-Euler).
                child_node = list(seq.scene_graph.successors(node))[0]
                start_dist_x = _get_joint_start_dist_x(seq.positions[:, seq.body_parts[child_node], 0])
                start_dist_y = _get_joint_start_dist_x(seq.positions[:, seq.body_parts[child_node], 1])
                if start_dist_x < start_dist_y:
                    joint_angles[:, body_parts[node]] = medical_from_euler_batch('xyz', angles_dict['euler_xyz'], node)
                else:
                    joint_angles[:, body_parts[node]] = medical_from_euler_batch('yxz', angles_dict['euler_yxz'], node)

            elif node in non_ball_joints:
                joint_angles[:, body_parts[node]] = medical_from_euler_batch('zxz', angles_dict['euler_zxz'], node)
            else:
                joint_angles[:, body_parts[node]] = np.array([None, None, None])
    joint_angles = _fix_range_exceedance([body_parts['shoulder_l'], body_parts['shoulder_r'], body_parts['hip_l'], body_parts['hip_r']], joint_angles, 180)
    return joint_angles
