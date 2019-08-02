from Sequence import Sequence
import numpy as np
import math
import transformations
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


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


def calc_angles_shoulder_left(seq: Sequence, shoulder_left_idx: int, shoulder_right_idx: int, neck_idx: int, elbow_left_idx: int, wrist_left_idx: int, log: bool = False) -> dict:
    """ Calculates Left Shoulder angles
    Parameters
    ----------
    seq : Sequence
        A Motion Sequence
    """

    # Move coordinate system to left shoulder for frame 20
    # align_coordinates_to(origin_bp_idx: int, x_direction_bp_idx: int, y_direction_bp_idx: int, seq: Sequence, frame: int)
    left_shoulder_aligned_positions = transformations.align_coordinates_to(shoulder_left_idx, shoulder_right_idx, neck_idx, seq, frame=35)
    # x,y,z coordinates for left elbow
    sx = left_shoulder_aligned_positions[shoulder_left_idx][0]
    sy = left_shoulder_aligned_positions[shoulder_left_idx][1]
    sz = left_shoulder_aligned_positions[shoulder_left_idx][2]
    ex = left_shoulder_aligned_positions[elbow_left_idx][0]
    ey = left_shoulder_aligned_positions[elbow_left_idx][1]
    ez = left_shoulder_aligned_positions[elbow_left_idx][2]
    wx = left_shoulder_aligned_positions[wrist_left_idx][0]
    wy = left_shoulder_aligned_positions[wrist_left_idx][1]
    wz = left_shoulder_aligned_positions[wrist_left_idx][2]

    # Convert to spherical coordinates
    er = math.sqrt(ex**2 + ey**2 + ez**2)
    # Y-Axis points upwards
    # Theta should be the angle between downwards vector and r
    # So we mirror Y-Axis
    theta = math.degrees(math.acos(-ey/er))
    # Phi is the anti-clockwise angle between Z and X
    # For Left shoulder, Z-Axis points away from camera and X-Axis is aligned to the right shoulder after transformations.
    # So for Left shoulder, we mirror the Z and X Axes
    phi = math.degrees(math.atan2(-ez, -ex))

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

    # rotation angle

    # Get normal for down and front vectors
    vx = transformations.norm(np.array([1, 0, 0]))
    vy = transformations.norm(np.array([0, 1, 0]))
    vz = transformations.norm(np.array([0, 0, 1]))

    # Get reversed Normal vector for YZ Plane ()
    zero_rot_plane_normal = transformations.get_perpendicular_vector(vy, vz)
    # Rotate the zero reference planes normal by abduction_adduction angle about wrist-elbow axis
    Re = transformations.rotation_matrix_4x4(left_shoulder_aligned_positions[0] - left_shoulder_aligned_positions[elbow_left_idx], np.radians(abduction_adduction))
    trans_zero_rot_plane_normal = np.matmul(Re, np.append(zero_rot_plane_normal, 1))[:3]

    # Elbow-Shoulder and Elbow-Wrist vectors
    v_es = np.array([sx, sy, sz]) - np.array([ex, ey, ez])
    v_ew = np.array([wx, wy, wz]) - np.array([ex, ey, ez])
    # Compute actual shoulder-elbow-wrist plane normal
    sew_plane_normal = transformations.get_perpendicular_vector(v_ew, v_es)
    # Calc angle between transformed zero reference plane normal and actual plane normal to get the inner/outer rotation
    inner_outer_rotation = np.degrees(transformations.get_angle(trans_zero_rot_plane_normal, sew_plane_normal))

    if log:
        print("\n##### SHOULDER LEFT ANGLES #####")
        print(f"r spherical: {er}")
        print(f"theta spherical: {theta}")
        print(f"phi spherical: {phi}")
        print(f"flexion_extension angle: {flexion_extension} (phi ratio: {phi_ratio_flex_ex})")
        print(f"abduction_adduction angle: {abduction_adduction} (phi ratio: {phi_ratio_abd_add})")
        print(f"inner_outer_rotation angle: {inner_outer_rotation}")

    ### Plotting ###

    fig = plt.figure(figsize=plt.figaspect(1)*2)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d(-0.5, 0.5)
    for i, p in enumerate(left_shoulder_aligned_positions):
        if i == 1:
            ax.scatter(p[0], p[1], p[2], c="blue")
        else:
            ax.scatter(p[0], p[1], p[2], c="blue")
    ax.plot([left_shoulder_aligned_positions[shoulder_left_idx][0], ex],
            [left_shoulder_aligned_positions[shoulder_left_idx][1], ey],
            [left_shoulder_aligned_positions[shoulder_left_idx][2], ez],
            color="pink", linewidth=1)
    zero_position = left_shoulder_aligned_positions[shoulder_left_idx]

    ax.plot([zero_position[0], vx[0]],
            [zero_position[1], vx[1]],
            [zero_position[2], vx[2]],
            color="pink", linewidth=1)
    ax.plot([zero_position[0], vy[0]],
            [zero_position[1], vy[1]],
            [zero_position[2], vy[2]],
            color="maroon", linewidth=1)
    ax.plot([zero_position[0], vz[0]],
            [zero_position[1], vz[1]],
            [zero_position[2], vz[2]],
            color="red", linewidth=1)

    # inner/outer Rotation plane normals
    ax.plot([ex, zero_rot_plane_normal[0] + ex],
            [ey, zero_rot_plane_normal[1] + ey],
            [ez, zero_rot_plane_normal[2] + ez],
            color="green", linewidth=3)
    ax.plot([ex, trans_zero_rot_plane_normal[0] + ex],
            [ey, trans_zero_rot_plane_normal[1] + ey],
            [ez, trans_zero_rot_plane_normal[2] + ez],
            color="red", linewidth=3)
    ax.plot([ex, sew_plane_normal[0] + ex],
            [ey, sew_plane_normal[1] + ey],
            [ez, sew_plane_normal[2] + ez],
            color="blue", linewidth=3)
    plt.show()

    return {
        "flexion_extension": flexion_extension,
        "abduction_adduction": abduction_adduction,
        "inner_outer_rotation": inner_outer_rotation
    }


def calc_angles_shoulder_right(seq: Sequence, shoulder_right_idx: int, shoulder_left_idx: int, neck_idx: int, elbow_right_idx: int, wrist_right_idx: int, log: bool = False) -> dict:
    """ Calculates Right Shoulder angles 
    Parameters
    ----------
    seq : Sequence
        A Motion Sequence
    """

    # Move coordinate system to left shoulder for frame 20
    # align_coordinates_to(origin_bp_idx: int, x_direction_bp_idx: int, y_direction_bp_idx: int, seq: Sequence, frame: int)
    right_shoulder_aligned_positions = transformations.align_coordinates_to(shoulder_right_idx, shoulder_left_idx, neck_idx, seq, frame=60)
    # x,y,z coordinates for left elbow
    sx = right_shoulder_aligned_positions[shoulder_right_idx][0]
    sy = right_shoulder_aligned_positions[shoulder_right_idx][1]
    sz = right_shoulder_aligned_positions[shoulder_right_idx][2]
    ex = right_shoulder_aligned_positions[elbow_right_idx][0]
    ey = right_shoulder_aligned_positions[elbow_right_idx][1]
    ez = right_shoulder_aligned_positions[elbow_right_idx][2]
    wx = right_shoulder_aligned_positions[wrist_right_idx][0]
    wy = right_shoulder_aligned_positions[wrist_right_idx][1]
    wz = right_shoulder_aligned_positions[wrist_right_idx][2]

    # Convert to spherical coordinates
    er = math.sqrt(ex**2 + ey**2 + ez**2)
    # Y-Axis points upwards
    # Theta should be the angle between downwards vector and r
    # So we mirror Y-Axis
    theta = math.degrees(math.acos(-ey/er))
    # Phi is the anti-clockwise angle between Z and X
    # For Right shoulder, Z-Axis points to the camera and X-Axis is aligned to the left shoulder after transformations.
    # So for Right shoulder, we mirror ONLY the X Axes to get a positive angle for flexion and negative angle for extension
    phi = math.degrees(math.atan2(ez, -ex))

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
        print("\n##### SHOULDER RIGHT ANGLES #####")
        print(f"r spherical: {er}")
        print(f"theta spherical: {theta}")
        print(f"phi spherical: {phi}")
        print(f"flexion_extension angle: {flexion_extension} (phi ratio: {phi_ratio_flex_ex})")
        print(f"abduction_adduction angle: {abduction_adduction} (phi ratio: {phi_ratio_abd_add})")
        # print(f"inner_outer_rotation angle: {inner_outer_rotation}")

    return {
        "flexion_extension": flexion_extension,
        "abduction_adduction": abduction_adduction,
        # "inner_outer_rotation": inner_outer_rotation
    }


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
