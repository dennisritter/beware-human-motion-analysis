from .Sequence import Sequence
from . import transformations
from . import plotting
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import matplotlib.pyplot as plt

""" This module contains functions to calculate medical angles for human body joints.
    Functions are Developed for the tracking input of Realsense MOCAP Project (https://gitlab.beuth-hochschule.de/iisy/realsense).
    If other input is used or the joint position output of Realsense MOCAP changes, the functions might need adjustments.
"""


def calc_angle(angle_vertex: list, ray_vertex_a: list, ray_vertex_b: list) -> float:
    """ Calculates the angle between angle_vertex_2d-ray_vertex_a and angle_vertex_2d-ray_vertex_b in 2D space.
    Returns
    ----------
    float
        Angle between angle_vertex_2d-ray_vertex_a and angle_vertex_2d-ray_vertex_b in degrees
    """
    angle_vertex = np.array(angle_vertex)
    ray_vertex_a = np.array(ray_vertex_a)
    ray_vertex_b = np.array(ray_vertex_b)
    ray_a = ray_vertex_a - angle_vertex
    ray_b = ray_vertex_b - angle_vertex
    cos_angle = np.dot(ray_a, ray_b) / (np.linalg.norm(ray_a) * np.linalg.norm(ray_b))
    return np.degrees(np.arccos(cos_angle))


def calc_angles_hip_left(seq: Sequence, hip_left_idx: int, hip_right_idx: int, torso_idx: int, knee_left_idx: int, log: bool = False) -> dict:
    """ Calculates Right Hip angles
    Parameters
    ----------
    seq : Sequence
        A Motion Sequence
    """
    flexion_extension_arr = []
    abduction_adduction_arr = []

    for frame in range(0, len(seq.positions)):
        # Move coordinate system to left Hip
        left_hip_aligned_positions = transformations.align_coordinates_to(hip_left_idx, hip_right_idx, torso_idx, seq, frame=frame)
        kx = left_hip_aligned_positions[knee_left_idx][0]
        ky = left_hip_aligned_positions[knee_left_idx][1]
        kz = left_hip_aligned_positions[knee_left_idx][2]

        # Convert to spherical coordinates
        kr = math.sqrt(kx**2 + ky**2 + kz**2)

        # Theta is the angle of the Hip-Knee Vector to the YZ-Plane
        theta = math.degrees(math.acos(-kx/kr))
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

        flexion_extension_arr.append(flexion_extension)
        abduction_adduction_arr.append(abduction_adduction)

        if log:
            print("\n##### HIP LEFT ANGLES #####")
            print(f"[{frame}] flexion_extension angle: {flexion_extension}")
            print(f"[{frame}] abduction_adduction angle: {abduction_adduction}")
        # plotting.plot_ball_joint_angle(left_hip_aligned_positions, hip_left_idx, knee_left_idx)

    return {
        "flexion_extension": flexion_extension_arr,
        "abduction_adduction": abduction_adduction_arr,
    }


def calc_angles_hip_right(seq: Sequence, hip_right_idx: int, hip_left_idx: int, torso_idx: int, knee_right_idx: int, log: bool = False) -> dict:
    """ Calculates Right Hip angles
    Parameters
    ----------
    seq : Sequence
        A Motion Sequence
    """
    flexion_extension_arr = []
    abduction_adduction_arr = []

    for frame in range(0, len(seq.positions)):
        # Move coordinate system to right Hip
        right_hip_aligned_positions = transformations.align_coordinates_to(hip_right_idx, hip_left_idx, torso_idx, seq, frame=frame)

        kx = right_hip_aligned_positions[knee_right_idx][0]
        ky = right_hip_aligned_positions[knee_right_idx][1]
        kz = right_hip_aligned_positions[knee_right_idx][2]

        # Convert to spherical coordinates
        kr = math.sqrt(kx**2 + ky**2 + kz**2)

        # Theta is the angle of the Hip-Knee Vector to the YZ-Plane
        theta = math.degrees(math.acos(kx/kr))
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

        flexion_extension_arr.append(flexion_extension)
        abduction_adduction_arr.append(abduction_adduction)

        if log:
            print("\n##### HIP RIGHT ANGLES #####")
            print(f"[{frame}] flexion_extension angle: {flexion_extension}")
            print(f"[{frame}] abduction_adduction angle: {abduction_adduction}")
        # plotting.plot_ball_joint_angle(right_hip_aligned_positions, hip_right_idx, knee_right_idx)

    return {
        "flexion_extension": flexion_extension_arr,
        "abduction_adduction": abduction_adduction_arr,
    }


def calc_angles_knee(seq: Sequence, knee_idx: int, hip_idx: int, ankle_idx: int) -> dict:
    """ Calculates the Knees flexion/extension angles for each frame of the Sequence.
    Parameters
    ----------
    seq : Sequence
        A Motion Sequence
    """
    knee = seq.positions[:, knee_idx, :]
    hip = seq.positions[:, hip_idx, :]
    ankle = seq.positions[:, ankle_idx, :]
    angles = []
    for i in range(len(knee)):
        # Substract angle from 180 because 'Normal Standing' is defined as 0°
        angles.append(180 - calc_angle(knee[i], hip[i], ankle[i]))

    return {
        "flexion_extension": angles
    }


def calc_angles_shoulder_left(seq: Sequence, shoulder_left_idx: int, shoulder_right_idx: int, torso_idx: int, elbow_left_idx: int, log: bool = False) -> dict:
    """ Calculates Left Shoulder angles
    Parameters
    ----------
    seq : Sequence
        A Motion Sequence
    """
    flexion_extension_arr = []
    abduction_adduction_arr = []

    for frame in range(0, len(seq.positions)):

        # Move coordinate system to left Shoulder
        left_shoulder_aligned_positions = transformations.align_coordinates_to(shoulder_left_idx, shoulder_right_idx, torso_idx, seq, frame=frame)

        ex = left_shoulder_aligned_positions[elbow_left_idx][0]
        ey = left_shoulder_aligned_positions[elbow_left_idx][1]
        ez = left_shoulder_aligned_positions[elbow_left_idx][2]

        # Convert to spherical coordinates
        er = math.sqrt(ex**2 + ey**2 + ez**2)

        # Theta is the angle of the Shoulder-Elbow Vector to the YZ-Plane and represents an abduction/adduction
        theta = math.degrees(math.acos(-ex/er))
        theta = 90.0 - theta
        abduction_adduction = theta

        # Phi is arbitrary when point is on rotation axis, so we set it to zero
        elbow_xaxis_angle = transformations.get_angle(np.array([1, 0, 0]), left_shoulder_aligned_positions[elbow_left_idx])
        if elbow_xaxis_angle == 0.0 or elbow_xaxis_angle == math.pi:
            phi = 0
        else:
            # Phi is the angle of the Elbow around the X-Axis (Down = 0) and represents flexion/extension angle
            phi = math.degrees(math.atan2(ey, -ez))
            phi += 90.0
            # An Extension should be represented in a negative angle
            if phi > 180.0:
                phi -= 360.0
        flexion_extension = phi

        flexion_extension_arr.append(flexion_extension)
        abduction_adduction_arr.append(abduction_adduction)

        if log:
            print("\n##### SHOULDER LEFT ANGLES #####")
            print(f"[{frame}] flexion_extension angle: {flexion_extension}")
            print(f"[{frame}] abduction_adduction angle: {abduction_adduction}")
        plotting.plot_ball_joint_angle(left_shoulder_aligned_positions, shoulder_left_idx, elbow_left_idx)

    return {
        "flexion_extension": flexion_extension_arr,
        "abduction_adduction": abduction_adduction_arr,
    }


def calc_angles_shoulder_right(seq: Sequence, shoulder_right_idx: int, shoulder_left_idx: int, torso_idx: int, elbow_right_idx: int, log: bool = False) -> dict:
    """ Calculates Right Shoulder angles 
    Parameters
    ----------
    seq : Sequence
        A Motion Sequence
    """
    flexion_extension_arr = []
    abduction_adduction_arr = []

    for frame in range(0, len(seq.positions)):
        # Move coordinate system to right shoulder
        right_shoulder_aligned_positions = transformations.align_coordinates_to(shoulder_right_idx, shoulder_left_idx, torso_idx, seq, frame=frame)

        ex = right_shoulder_aligned_positions[elbow_right_idx][0]
        ey = right_shoulder_aligned_positions[elbow_right_idx][1]
        ez = right_shoulder_aligned_positions[elbow_right_idx][2]

        # Convert to spherical coordinates
        er = math.sqrt(ex**2 + ey**2 + ez**2)

        # Theta is the angle of the Shoulder-Elbow Vector to the YZ-Plane
        theta = math.degrees(math.acos(ex/er))
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

        flexion_extension_arr.append(flexion_extension)
        abduction_adduction_arr.append(abduction_adduction)

        if log:
            print("\n##### SHOULDER RIGHT ANGLES #####")
            print(f"[{frame}] flexion_extension angle: {flexion_extension}")
            print(f"[{frame}] abduction_adduction angle: {abduction_adduction}")
        # plotting.plot_ball_joint_angle(right_shoulder_aligned_positions, shoulder_right_idx, elbow_right_idx)

    return {
        "flexion_extension": flexion_extension_arr,
        "abduction_adduction": abduction_adduction_arr,
    }


def calc_angles_elbow(seq: Sequence, elbow_idx: int, shoulder_idx: int, wrist_idx: int) -> dict:
    """ Calculates the Elbows flexion/extension angles for each frame of the Sequence.
    Parameters
    ----------
    seq : Sequence
        A Motion Sequence
    """
    elbow = seq.positions[:, elbow_idx, :]
    wrist = seq.positions[:, wrist_idx, :]
    shoulder = seq.positions[:, shoulder_idx, :]
    angles = []
    for i in range(len(elbow)):
        # Substract angle from 180 because 'Normal Standing' (straight arm) is defined as 0°
        angles.append(180 - calc_angle(elbow[i], wrist[i], shoulder[i]))

    return {
        "flexion_extension": angles
    }
