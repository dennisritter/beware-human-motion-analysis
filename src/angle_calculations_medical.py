from Sequence import Sequence
import numpy as np
import math
import transformations


def calc_angle(angle_vertex: list, ray_vertex_a: list, ray_vertex_b: list) -> float:
    """ Calculates the angle between angle_vertex_2d-ray_vertex_a and angle_vertex_2d-ray_vertex_b in 2D space.
    Returns
    ----------
    float
        Angle between angle_vertex_2d-ray_vertex_a and angle_vertex_2d-ray_vertex_b in degrees
    """
    ray_a = ray_vertex_a - angle_vertex
    ray_b = ray_vertex_b - angle_vertex
    cos_angle = np.dot(ray_a, ray_b) / (np.linalg.norm(ray_a) * np.linalg.norm(ray_b))
    return np.degrees(np.arccos(cos_angle))


def calc_angle_hip_flexion_extension(seq: Sequence, joints: dict) -> list:
    # NOTE: Observations and Potential Problems:
    #       * Torso position might be incorrect for heavy people with big upper body
    #           => Possible solution: Calibration to calculate a bias? [Tell user to Stand (start) -> Track pose -> calculate bias]
    """ Calculates the hips flexion/extension angles for each frame of the Sequence.
    Parameters
    ----------
    seq : Sequence
        A Motion Sequence
    joints : dict
        The joints to use for angle calculation.
        Attributes:
            angle_vertex : int
            rays : list<int>
        Example: { "angle_vertex": 1, "rays": [0, 2] }
    """
    hip = seq.positions[:, joints["angle_vertex"], 1:]
    knee = seq.positions[:, joints["rays"][0], 1:]
    torso = seq.positions[:, joints["rays"][1], 1:]
    angles = []
    for i in range(len(hip)):
        # Substract angle from 180 because 'Normal Standing' is defined as 0°
        angles.append(180 - calc_angle(hip[i], knee[i], torso[i]))

    return angles


def calc_angle_hip_abduction_adduction(seq: Sequence, joints: dict) -> list:
    # NOTE: Observations and Potential Problems:
    #   * 'Normal Standing' angle will never be 0° because using left/right hip-torso ray for calculation.
    #       Possible solution:  -> Use a bias of 5-10°
    #                           -> Don't use shoulder-hip ray but direct shoulder-floor ray for calculation
    """ Calculates the hips abduction/adduction angles for each frame of the Sequence.
    Parameters
    ----------
    seq : Sequence
        A Motion Sequence
    joints : dict
        The joints to use for angle calculation.
        Attributes:
            angle_vertex : int
            rays : list<int>
        Example: { "angle_vertex": 1, "rays": [0, 2] }
    """
    # Ignore Z Axis for abduction/adduction
    hip = seq.positions[:, joints["angle_vertex"], :2]
    knee = seq.positions[:, joints["rays"][0], :2]
    torso = seq.positions[:, joints["rays"][1], :2]
    angles = []
    for i in range(len(hip)):
        # Substract angle from 180 because 'Normal Standing' is defined as 0°
        angles.append(180 - calc_angle(hip[i], knee[i], torso[i]))

    return angles


def calc_angle_knee_flexion_extension(seq: Sequence, joints: dict) -> list:
    """ Calculates the Knees flexion/extension angles for each frame of the Sequence.
    Parameters
    ----------
    seq : Sequence
        A Motion Sequence
    joints : dict
        The joints to use for angle calculation.
        Attributes:
            angle_vertex : int
            rays : list<int>
        Example: { "angle_vertex": 1, "rays": [0, 2] }
    """
    knee = seq.positions[:, joints["angle_vertex"], :]
    hip = seq.positions[:, joints["rays"][0], :]
    ankle = seq.positions[:, joints["rays"][1], :]
    angles = []
    for i in range(len(knee)):
        # Substract angle from 180 because 'Normal Standing' is defined as 0°
        angles.append(180 - calc_angle(knee[i], hip[i], ankle[i]))

    return angles


def calc_angles_shoulder_left(seq: Sequence, shoulder_left_idx: int, shoulder_right_idx: int, neck_idx: int, elbow_left_idx: int, log: bool = False) -> dict:
    """ Calculates Left Shoulder angles 
    Parameters
    ----------
    seq : Sequence
        A Motion Sequence
    """

    # Move coordinate system to left shoulder for frame 20
    # align_coordinates_to(origin_bp_idx: int, x_direction_bp_idx: int, y_direction_bp_idx: int, seq: Sequence, frame: int)
    left_shoulder_aligned_positions = transformations.align_coordinates_to(shoulder_left_idx, shoulder_right_idx, neck_idx, seq, frame=60)
    # x,y,z coordinates for left elbow
    x = left_shoulder_aligned_positions[elbow_left_idx][0]
    y = left_shoulder_aligned_positions[elbow_left_idx][1]
    z = left_shoulder_aligned_positions[elbow_left_idx][2]

    # Convert to spherical coordinates
    r = math.sqrt(x**2 + y**2 + z**2)
    # Y-Axis points upwards
    # Theta should be the angle between downwards vector and r
    # So we mirror Y-Axis
    theta = math.degrees(math.acos(-y/r))
    # Phi is the anti-clockwise angle between Z and X
    # For Left shoulder, Z-Axis points away from camera and X-Axis is aligned to the right shoulder after transformations.
    # So for Left shoulder, we mirror the Z and X Axes
    phi = math.degrees(math.atan2(-z, -x))

    # The phi_ratio will determine how much of the theta angle is flexion_extension and abduction_adduction
    # phi_ratio == -1 -> 0% Abduction_Adduction / 100% Extension
    # phi_ratio == 0(-2) -> 100% Abduction / 0% Flexion_Extension
    # phi_ratio == 1 -> 0% Abduction_Adduction / 100% Flexion
    # phi_ratio == 2 -> 100% Adduction / 0% Flexion_Extension
    phi_ratio = phi/90

    # Ensure phi_ratio_flex_ex alters between -1 and 1
    # flexion_extension > 0 -> Flexion
    # flexion_extension < 0 -> Extension
    phi_ratio_flex_ex = phi_ratio
    if phi_ratio_flex_ex <= 1 and phi_ratio_flex_ex >= -1:
        flexion_extension = theta*phi_ratio_flex_ex
    elif phi_ratio > 1:
        phi_ratio_flex_ex = 2-phi_ratio_flex_ex
        flexion_extension = theta*phi_ratio_flex_ex
    elif phi_ratio < -1:
        phi_ratio_flex_ex = -2-phi_ratio_flex_ex
        flexion_extension = theta*phi_ratio_flex_ex

    # Ensure phi_ratio_abd_add is between -1 and 1
    phi_ratio_abd_add = 1-abs(phi_ratio)
    # abduction_adduction > 0 -> Abduction
    # abduction_adduction < 0 -> Adduction
    abduction_adduction = theta*phi_ratio_abd_add

    if log:
        print(f"r spherical: {theta}")
        print(f"theta spherical: {theta}")
        print(f"phi spherical: {phi}")
        print(f"flexion_extension angle: {flexion_extension} (phi ratio: {phi_ratio_flex_ex})")
        print(f"abduction_adduction angle: {abduction_adduction} (phi ratio: {phi_ratio_abd_add})")

    return {
        "flexion_extension": flexion_extension,
        "abduction_adduction": abduction_adduction
    }


def calc_angles_shoulder_right(seq: Sequence, shoulder_right_idx: int, shoulder_left_idx: int, neck_idx: int, elbow_right_idx: int) -> dict:
    """ Calculates Right Shoulder angles 
    Parameters
    ----------
    seq : Sequence
        A Motion Sequence
    """

    # Phi is the anti-clockwise angle between Z and X
    # For Right shoulder, Z-Axis points to camera after transformations.
    # So for Right shoulder, we only mirror the X-Axis


def calc_angle_elbow_flexion_extension(seq: Sequence, joints: dict) -> list:
    """ Calculates the Elbows flexion/extension angles for each frame of the Sequence.
    Parameters
    ----------
    seq : Sequence
        A Motion Sequence
    joints : dict
        The joints to use for angle calculation.
        Attributes:
            angle_vertex : int
            rays : list<int>
        Example: { "angle_vertex": 1, "rays": [0, 2] }
    """
    elbow = seq.positions[:, joints["angle_vertex"], :]
    wrist = seq.positions[:, joints["rays"][0], :]
    shoulder = seq.positions[:, joints["rays"][1], :]
    angles = []
    for i in range(len(elbow)):
        # Substract angle from 180 because 'Normal Standing' (straight arm) is defined as 0°
        angles.append(180 - calc_angle(elbow[i], wrist[i], shoulder[i]))

    return angles
