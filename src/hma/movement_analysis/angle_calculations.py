from hma.movement_analysis import transformations
from hma.movement_analysis.enums.angle_types import AngleTypes
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import matplotlib.pyplot as plt

""" This module contains functions to calculate medical angles for human body joints.
    Functions are Developed for the tracking input of Realsense MOCAP Project (https://gitlab.beuth-hochschule.de/iisy/realsense).
    If other input is used or the joint position output of Realsense MOCAP changes, the functions might need adjustments.
"""


def calc_angle(angle_vertex: np.ndarray, ray_vertex_a: np.ndarray, ray_vertex_b: np.ndarray) -> float:
    """ Calculates the angle between the 'angle_vertex to ray_vertex_a vector' and 'angle_vertex to ray_vertex_b vector' in 2D space.
    Args:
        angle_vertex (np.ndarray): The angular point.
        ray_vertex_a (np.ndarray): The point, which determines the direction of ray_a.
        ray_vertex_b (np.ndarray): The point, which determines the direction of ray_b.
    Returns:
        (float) Angle between angle_vertex_2d-ray_vertex_a and angle_vertex_2d-ray_vertex_b in degrees
    """
    angle_vertex = np.array(angle_vertex)
    ray_vertex_a = np.array(ray_vertex_a)
    ray_vertex_b = np.array(ray_vertex_b)
    ray_a = ray_vertex_a - angle_vertex
    ray_b = ray_vertex_b - angle_vertex
    cos_angle = np.dot(ray_a, ray_b) / (np.linalg.norm(ray_a) * np.linalg.norm(ray_b))
    return np.degrees(np.arccos(cos_angle))


def calc_angles_hip_left(positions: np.ndarray, hip_left_idx: int, hip_right_idx: int, torso_idx: int, knee_left_idx: int) -> np.ndarray:
    """ Calculates Left Hip medical joint angles for Flexion/Extension and Abduction/Adduction motions.

    Creates a Joint Coordinate System in order to determine medical joint angles by utilising the concept of spherical coordinates.
    Further processing ensures that all angles of the anatomical position equal zero.

    Args:
        positions (np.ndarray): The spatial positions of a motion sequence of shape (n_frames, n_body_parts, 3).
        hip_left_idx (int): The  2nd axis index of the left hip x, y, z positions in the 'positions' array.
        hip_right_idx (int): The 2nd axis index of the right hip x, y, z positions in the 'positions' array.
        torso_idx (int): The 2nd axis index of the torso x, y, z positions in the 'positions' array.
        knee_left_idx (int): The 2nd axis index of the left knee x, y, z positions in the 'positions' array.
    Returns:
        (np.ndarray<float>) A Numpy Array containing three floats of Flexion/Extension [0], Abduction/Adduction [1] and Internal/External Rotation [2] angles.
        Note: The Internal/External Rotation value is always 0.0 as it is not calculated at the moment)
    """
    n_frames = len(positions)
    n_angle_types = len(AngleTypes)
    angles = np.zeros((n_frames, n_angle_types))
    for frame in range(0, n_frames):
        # Move coordinate system to left Hip
        left_hip_aligned_positions = transformations.align_coordinates_to(hip_left_idx, hip_right_idx, torso_idx, positions[frame])
        kx = left_hip_aligned_positions[knee_left_idx][0]
        ky = left_hip_aligned_positions[knee_left_idx][1]
        kz = left_hip_aligned_positions[knee_left_idx][2]

        # Convert to spherical coordinates
        kr = math.sqrt(kx**2 + ky**2 + kz**2)

        # Theta is the angle of the Hip-Knee Vector to the YZ-Plane
        theta = math.degrees(math.acos(kx/kr))
        theta = 90 - theta
        abduction_adduction = theta

        # Phi is arbitrary when point is on rotation axis, so we set it to zero
        knee_xaxis_angle = transformations.get_angle(np.array([1, 0, 0]), left_hip_aligned_positions[knee_left_idx])
        if knee_xaxis_angle == 0.0 or knee_xaxis_angle == math.pi:
            phi = 0.0
        else:
            # Phi is the angle of the Knee around the X-Axis (Down = 0) and represents flexion/extension angle
            phi = math.degrees(math.atan2(ky, -kz))
            phi += 90
            # An Extension should be represented in a negative angle
            if phi > 180:
                phi -= 360
        flexion_extension = phi

        angles[frame][AngleTypes.FLEX_EX.value] = flexion_extension
        angles[frame][AngleTypes.AB_AD.value] = abduction_adduction
    return angles


def calc_angles_hip_right(positions: np.ndarray, hip_right_idx: int, hip_left_idx: int, torso_idx: int, knee_right_idx: int) -> np.ndarray:
    """ Calculates Right Hip medical joint angles for Flexion/Extension and Abduction/Adduction motions.

    Creates a Joint Coordinate System in order to determine medical joint angles by utilising the concept of spherical coordinates.
    Further processing ensures that all angles of the anatomical position equal zero.

    Args:
        positions (np.ndarray): The spatial positions of a motion sequence of shape (n_frames, n_body_parts, 3).
        hip_right_idx (int): The 2nd axis index of the right hip x, y, z positions in the 'positions' array.
        hip_left_idx (int): The  2nd axis index of the left hip x, y, z positions in the 'positions' array.
        torso_idx (int): The 2nd axis index of the torso x, y, z positions in the 'positions' array.
        knee_left_idx (int): The 2nd axis index of the left knee x, y, z positions in the 'positions' array.
    Returns:
        (np.ndarray<float>) A Numpy Array containing three floats of Flexion/Extension [0], Abduction/Adduction [1] and Internal/External Rotation [2] angles.
        Note: The Internal/External Rotation value is always 0.0 as it is not calculated at the moment)
    """
    n_frames = len(positions)
    n_angle_types = len(AngleTypes)
    angles = np.zeros((n_frames, n_angle_types))
    for frame in range(0, n_frames):
        # Move coordinate system to right Hip
        right_hip_aligned_positions = transformations.align_coordinates_to(hip_right_idx, hip_left_idx, torso_idx, positions[frame])

        kx = right_hip_aligned_positions[knee_right_idx][0]
        ky = right_hip_aligned_positions[knee_right_idx][1]
        kz = right_hip_aligned_positions[knee_right_idx][2]

        # Convert to spherical coordinates
        kr = math.sqrt(kx**2 + ky**2 + kz**2)

        # Theta is the angle of the Hip-Knee Vector to the YZ-Plane
        theta = math.degrees(math.acos(-kx/kr))
        theta = 90 - theta
        abduction_adduction = theta

        # Phi is arbitrary when point is on rotation axis, so we set it to zero
        knee_xaxis_angle = transformations.get_angle(np.array([1, 0, 0]), right_hip_aligned_positions[knee_right_idx])
        if knee_xaxis_angle == 0.0 or knee_xaxis_angle == math.pi:
            phi = 0.0
        else:
            # Phi is the angle of the Knee around the X-Axis (Down = 0) and represents flexion/extension angle
            phi = math.degrees(math.atan2(ky, -kz))
            phi += 90
            # An Extension should be represented in a negative angle
            if phi > 180:
                phi -= 360
        flexion_extension = phi

        angles[frame][AngleTypes.FLEX_EX.value] = flexion_extension
        angles[frame][AngleTypes.AB_AD.value] = abduction_adduction
    return angles


def calc_angles_knee(positions: list, knee_idx: int, hip_idx: int, ankle_idx: int) -> np.ndarray:
    """ Calculates the Flexion/Extension angles for the left/right knee

    Args:
        positions (np.ndarray): The spatial positions of a motion sequence of shape (n_frames, n_body_parts, 3).
        knee_idx (int): The 2nd axis index of the left/right knee x, y, z positions in the 'positions' array.
        hip_idx (int): The  2nd axis index of the left/right hip x, y, z positions in the 'positions' array.
        ankle_idx (int): The 2nd axis index of the left/right ankle x, y, z positions in the 'positions' array.
    Returns:
        (np.ndarray<float>) A Numpy Array containing three floats of Flexion/Extension [0], Abduction/Adduction [1] and Internal/External Rotation [2] angles.
        Note: The Abduction/Adduction and Internal/External Rotation values are always 0.0 as they are no not calculated for the knees at the moment)
    """
    knee = positions[:, knee_idx, :]
    hip = positions[:, hip_idx, :]
    ankle = positions[:, ankle_idx, :]

    n_frames = len(knee)
    angles = np.zeros((n_frames, len(AngleTypes)))
    for frame in range(n_frames):
        # Substract angle from 180 because 'Normal Standing' is defined as 0°
        angles[frame][AngleTypes.FLEX_EX.value] = 180 - calc_angle(knee[frame], hip[frame], ankle[frame])

    return angles


def calc_angles_shoulder_left(positions: list, shoulder_left_idx: int, shoulder_right_idx: int, torso_idx: int, elbow_left_idx: int) -> np.ndarray:
    """ Calculates Left Shoulder medical joint angles for Flexion/Extension and Abduction/Adduction motions.

    Creates a Joint Coordinate System in order to determine medical joint angles by utilising the concept of spherical coordinates.
    Further processing ensures that all angles of the anatomical position equal zero.

    Args:
        positions (np.ndarray): The spatial positions of a motion sequence of shape (n_frames, n_body_parts, 3).
        shoulder_left_idx (int): The 2nd axis index of the left shoulder x, y, z positions in the 'positions' array.
        shoulder_right_idx (int): The  2nd axis index of the right shoulder x, y, z positions in the 'positions' array.
        torso_idx (int): The 2nd axis index of the torso x, y, z positions in the 'positions' array.
        elbow_left_idx (int): The 2nd axis index of the left elbow x, y, z positions in the 'positions' array.
    Returns:
        (np.ndarray<float>) A Numpy Array containing three floats of Flexion/Extension [0], Abduction/Adduction [1] and Internal/External Rotation [2] angles.
        Note: The Internal/External Rotation value is always 0.0 as it is not calculated at the moment)
    """
    n_frames = len(positions)
    n_angle_types = len(AngleTypes)
    angles = np.zeros((n_frames, n_angle_types))
    for frame in range(0, n_frames):
        # Move coordinate system to left Shoulder
        left_shoulder_aligned_positions = transformations.align_coordinates_to(shoulder_left_idx, shoulder_right_idx, torso_idx, positions[frame])

        ex = left_shoulder_aligned_positions[elbow_left_idx][0]
        ey = left_shoulder_aligned_positions[elbow_left_idx][1]
        ez = left_shoulder_aligned_positions[elbow_left_idx][2]
        # Convert to spherical coordinates
        er = math.sqrt(ex**2 + ey**2 + ez**2)

        # Theta is the angle of the Shoulder-Elbow Vector to the YZ-Plane and represents an abduction/adduction.
        theta = math.degrees(math.acos(ex/er))
        theta = 90.0 - theta
        abduction_adduction = theta

        # Phi is arbitrary when point is on rotation axis, so we set it to zero
        elbow_xaxis_angle = transformations.get_angle(np.array([1, 0, 0]), left_shoulder_aligned_positions[elbow_left_idx])
        if elbow_xaxis_angle == 0.0 or elbow_xaxis_angle == math.pi:
            phi = 0
        else:
            # Phi is the angle of the Elbow around the X-Axis (Down = 0) and represents flexion/extension angle.
            phi = math.degrees(math.atan2(ey, -ez))
            phi += 90.0
            # An Extension should be represented in a negative angle.
            if phi > 180.0:
                phi -= 360.0
        flexion_extension = phi

        angles[frame][AngleTypes.FLEX_EX.value] = flexion_extension
        angles[frame][AngleTypes.AB_AD.value] = abduction_adduction
    return angles


def calc_angles_shoulder_right(positions: list, shoulder_right_idx: int, shoulder_left_idx: int, torso_idx: int, elbow_right_idx: int) -> np.ndarray:
    """ Calculates Right Shoulder medical joint angles for Flexion/Extension and Abduction/Adduction motions.

    Creates a Joint Coordinate System in order to determine medical joint angles by utilising the concept of spherical coordinates.
    Further processing ensures that all angles of the anatomical position equal zero.

    Args:
        positions (np.ndarray): The spatial positions of a motion sequence of shape (n_frames, n_body_parts, 3).
        shoulder_right_idx (int): The  2nd axis index of the right shoulder x, y, z positions in the 'positions' array.
        shoulder_left_idx (int): The 2nd axis index of the left shoulder x, y, z positions in the 'positions' array.
        torso_idx (int): The 2nd axis index of the torso x, y, z positions in the 'positions' array.
        elbow_right_idx (int): The 2nd axis index of the right elbow x, y, z positions in the 'positions' array.
    Returns:
        (np.ndarray<float>) A Numpy Array containing three floats of Flexion/Extension [0], Abduction/Adduction [1] and Internal/External Rotation [2] angles.
        Note: The Internal/External Rotation value is always 0.0 as it is not calculated at the moment)
    """
    n_frames = len(positions)
    n_angle_types = len(AngleTypes)
    angles = np.zeros((n_frames, n_angle_types))
    for frame in range(0, n_frames):
        # Move coordinate system to right shoulder
        right_shoulder_aligned_positions = transformations.align_coordinates_to(shoulder_right_idx, shoulder_left_idx, torso_idx, positions[frame])

        ex = right_shoulder_aligned_positions[elbow_right_idx][0]
        ey = right_shoulder_aligned_positions[elbow_right_idx][1]
        ez = right_shoulder_aligned_positions[elbow_right_idx][2]

        # Convert to spherical coordinates
        er = math.sqrt(ex**2 + ey**2 + ez**2)

        # Theta is the angle of the Shoulder-Elbow Vector to the YZ-Plane
        theta = math.degrees(math.acos(-ex/er))
        theta = 90 - theta
        abduction_adduction = theta

        # Phi is arbitrary when point is on rotation axis, so we set it to zero
        elbow_xaxis_angle = transformations.get_angle(np.array([1, 0, 0]), right_shoulder_aligned_positions[elbow_right_idx])
        if elbow_xaxis_angle == 0.0 or elbow_xaxis_angle == math.pi:
            phi = 0
        else:
            # Phi is the angle of the Elbow around the X-Axis (Down = 0) and represents flexion/extension angle

            phi = math.degrees(math.atan2(ey, -ez))
            phi += 90
            # An Extension should be represented in a negative angle
            if phi > 180:
                phi -= 360
        flexion_extension = phi

        angles[frame][AngleTypes.FLEX_EX.value] = flexion_extension
        angles[frame][AngleTypes.AB_AD.value] = abduction_adduction
    return angles


def calc_angles_elbow(positions: list, elbow_idx: int, shoulder_idx: int, wrist_idx: int) -> np.ndarray:
    """ Calculates the Flexion/Extension angles for the left/right elbow

    Args:
        positions (np.ndarray): The spatial positions of a motion sequence of shape (n_frames, n_body_parts, 3).
        knee_idx (int): The 2nd axis index of the left/right elbow x, y, z positions in the 'positions' array.
        hip_idx (int): The  2nd axis index of the left/right shoulder x, y, z positions in the 'positions' array.
        ankle_idx (int): The 2nd axis index of the left/right wrist x, y, z positions in the 'positions' array.
    Returns:
        (np.ndarray<float>) A Numpy Array containing three floats of Flexion/Extension [0], Abduction/Adduction [1] and Internal/External Rotation [2] angles.
        Note: The Abduction/Adduction and Internal/External Rotation values are always 0.0 as they are no not calculated for the knees at the moment)
    """
    elbow = positions[:, elbow_idx, :]
    wrist = positions[:, wrist_idx, :]
    shoulder = positions[:, shoulder_idx, :]

    n_frames = len(elbow)
    angles = np.zeros((n_frames, len(AngleTypes)))
    for frame in range(n_frames):
        # Substract angle from 180 because 'Normal Standing' (straight arm down) is defined as 0°
        angles[frame][AngleTypes.FLEX_EX.value] = (180 - calc_angle(elbow[frame], wrist[frame], shoulder[frame]))

    return angles
