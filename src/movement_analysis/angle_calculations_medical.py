from .Sequence import Sequence
from . import transformations
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
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

    # for frame in range(37, 40):
    for frame in range(0, len(seq.positions)):
        left_hip_aligned_positions = transformations.align_coordinates_to(hip_left_idx, hip_right_idx, torso_idx, seq, frame=frame)
        print(left_hip_aligned_positions)
        vx = transformations.norm(np.array([1, 0, 0]))
        vy = transformations.norm(np.array([0, 1, 0]))
        vz = transformations.norm(np.array([0, 0, 1]))
        kx = left_hip_aligned_positions[knee_left_idx][0]
        ky = left_hip_aligned_positions[knee_left_idx][1]
        kz = left_hip_aligned_positions[knee_left_idx][2]

        # Convert to spherical coordinates
        kr = math.sqrt(kx**2 + ky**2 + kz**2)
                # NOTE: Hacky: We assume that the neck coords are above shoulders to determine if Y points up or down
        #       and Z points to front or back. Spherical coords must be calculated differently for each case.
        torso_pos = left_hip_aligned_positions[torso_idx]
        # Theta should be the angle between downwards vector and r
        # True: Y-Axis Points up -> Mirror Y-Axis so it points down (to 0° medical definition)
        # False: Y-Axis Points down -> Don't mirror Y-Axis
        theta = math.degrees(math.acos(-ky/kr)) if (torso_pos[1] >= 0) else math.degrees(math.acos(ky/kr))

        # Phi is the anti-clockwise angle between Z and X
        # True: Y-Axis Points up -> Z points away from camera -> mirror Z-Axis
        # False: Y-Axis Points down -> Z points to camera -> mirror Z-Axis
        # NOTE: This is different to right hip because X-Axis is rotated by 180° in respect to the other hips positions
        #       (mirror Z for left hip, but not for right hip if Y Points up)
        phi = math.degrees(math.atan2(-kz, -kx)) if (torso_pos[1] >= 0) else math.degrees(math.atan2(kz, -kx))

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
        
        flexion_extension_arr.append(flexion_extension)
        abduction_adduction_arr.append(abduction_adduction)
        
        if log:
            print("\n##### HIP LEFT ANGLES #####")
            print(f"[{frame}] r spherical: {kr}")
            print(f"[{frame}] theta spherical: {theta}")
            print(f"[{frame}] phi spherical: {phi}")
            print(f"[{frame}] flexion_extension angle: {flexion_extension} (phi ratio: {phi_ratio_flex_ex})")
            print(f"[{frame}] abduction_adduction angle: {abduction_adduction} (phi ratio: {phi_ratio_abd_add})")
    
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

    # for frame in range(37, 40):
    for frame in range(0, len(seq.positions)):
        right_hip_aligned_positions = transformations.align_coordinates_to(hip_right_idx, hip_left_idx, torso_idx, seq, frame=frame)
        
        vx = transformations.norm(np.array([1, 0, 0]))
        vy = transformations.norm(np.array([0, 1, 0]))
        vz = transformations.norm(np.array([0, 0, 1]))
        kx = right_hip_aligned_positions[knee_right_idx][0]
        ky = right_hip_aligned_positions[knee_right_idx][1]
        kz = right_hip_aligned_positions[knee_right_idx][2]

        # Convert to spherical coordinates
        kr = math.sqrt(kx**2 + ky**2 + kz**2)
        # NOTE: Hacky: We assume that the neck coords are above shoulders to determine if Y points up or down
        #       and Z points to front or back. Spherical coords must be calculated differently for each case.
        torso_pos = right_hip_aligned_positions[torso_idx]
        # Theta should be the angle between downwards vector and r
        # True: Y-Axis Points up -> Mirror Y-Axis so it points down (to 0° medical definition)
        # False: Y-Axis Points down -> Don't mirror Y-Axis
        theta = math.degrees(math.acos(-ky/kr)) if (torso_pos[1] >= 0) else math.degrees(math.acos(ky/kr))

        # Phi is the anti-clockwise angle between Z and X
        # True: Y-Axis Points up -> Z points to camera -> don't mirror Z-Axis
        # False: Y-Axis Points down -> Z points away from camera -> mirror Z-Axis
        phi = math.degrees(math.atan2(kz, -kx)) if (torso_pos[1] >= 0) else math.degrees(math.atan2(-kz, -kx))

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
        
        flexion_extension_arr.append(flexion_extension)
        abduction_adduction_arr.append(abduction_adduction)
        
        if log:
            print("\n##### HIP RIGHT ANGLES #####")
            print(f"[{frame}] r spherical: {kr}")
            print(f"[{frame}] theta spherical: {theta}")
            print(f"[{frame}] phi spherical: {phi}")
            print(f"[{frame}] flexion_extension angle: {flexion_extension} (phi ratio: {phi_ratio_flex_ex})")
            print(f"[{frame}] abduction_adduction angle: {abduction_adduction} (phi ratio: {phi_ratio_abd_add})")

    ### Plotting ###
    """
    fig = plt.figure(figsize=plt.figaspect(1)*2)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d(-0.5, 0.5)
    for i, p in enumerate(right_hip_aligned_positions):
        if i == 1:
            ax.scatter(p[0], p[1], p[2], c="blue")
        else:
            ax.scatter(p[0], p[1], p[2], c="blue")
    ax.plot([right_hip_aligned_positions[hip_right_idx][0], kx],
            [right_hip_aligned_positions[hip_right_idx][1], ky],
            [right_hip_aligned_positions[hip_right_idx][2], kz],
            color="pink", linewidth=1)
    zero_position = right_hip_aligned_positions[hip_right_idx]

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
    plt.show()
    """
    
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

def calc_angles_shoulder_left(seq: Sequence, shoulder_left_idx: int, shoulder_right_idx: int, neck_idx: int, elbow_left_idx: int, wrist_left_idx: int, log: bool = False) -> dict:
    """ Calculates Left Shoulder angles
    Parameters
    ----------
    seq : Sequence
        A Motion Sequence
    """
    flexion_extension_arr = []
    abduction_adduction_arr = []
    # inner_outer_rotation_arr = []
    # for frame in range(44, 45):
    for frame in range(0, len(seq.positions)):

        # Move coordinate system to left shoulder for frame 20
        # align_coordinates_to(origin_bp_idx: int, x_direction_bp_idx: int, y_direction_bp_idx: int, seq: Sequence, frame: int)
        left_shoulder_aligned_positions = transformations.align_coordinates_to(shoulder_left_idx, shoulder_right_idx, neck_idx, seq, frame=frame)

        vx = transformations.norm(np.array([1, 0, 0]))
        vy = transformations.norm(np.array([0, 1, 0]))
        vz = transformations.norm(np.array([0, 0, 1]))
        ex = left_shoulder_aligned_positions[elbow_left_idx][0]
        ey = left_shoulder_aligned_positions[elbow_left_idx][1]
        ez = left_shoulder_aligned_positions[elbow_left_idx][2]
        """ Positions only for inner/outer rotation angle
        sx = left_shoulder_aligned_positions[shoulder_left_idx][0]
        sy = left_shoulder_aligned_positions[shoulder_left_idx][1]
        sz = left_shoulder_aligned_positions[shoulder_left_idx][2]
        wx = left_shoulder_aligned_positions[wrist_left_idx][0]
        wy = left_shoulder_aligned_positions[wrist_left_idx][1]
        wz = left_shoulder_aligned_positions[wrist_left_idx][2]
        """
        # Convert to spherical coordinates
        er = math.sqrt(ex**2 + ey**2 + ez**2)
        # # Y-Axis points upwards
        # # Theta should be the angle between downwards vector and r
        # # So we mirror Y-Axis
        # theta = math.degrees(math.acos(-ey/er))
        # # Phi is the anti-clockwise angle between Z and X
        # # For Left shoulder, Z-Axis points away from camera and X-Axis is aligned to the right shoulder after transformations.
        # # So for Left shoulder, we mirror the Z and X Axes
        # phi = math.degrees(math.atan2(-ez, -ex))

        # NOTE: Hacky: We assume that the neck coords are above shoulders to determine if Y points up or down
        #       and Z points to front or back. Spherical coords must be calculated differently for each case.
        neck_pos = left_shoulder_aligned_positions[neck_idx]
        # Theta should be the angle between downwards vector and r
        # True: Y-Axis Points up -> Mirror Y-Axis so it points down (to 0° medical definition)
        # False: Y-Axis Points down -> Don't mirror Y-Axis
        theta = math.degrees(math.acos(-ey/er)) if (neck_pos[1] >= 0) else math.degrees(math.acos(ey/er))

        # Phi is the anti-clockwise angle between Z and X
        # True: Y-Axis Points up -> Z points away from camera -> mirror Z-Axis
        # False: Y-Axis Points down -> Z points to camera -> mirror Z-Axis
        # NOTE: This is different to right shoulder because X-Axis is rotated by 180° in respect to the other shoulders positions
        #       (mirror Z for left shoulder, but not for right shoulder if Y Points up)
        phi = math.degrees(math.atan2(-ez, -ex)) if (neck_pos[1] >= 0) else math.degrees(math.atan2(ez, -ex))

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

        """ INNER OUTER ROTATION
        # rotation angle
        # TODO: Confirm rotation angle correctness by calculating it for more obvious poses

        # Get Normal vector for YZ Plane ()
        zero_rot_plane_normal = transformations.get_perpendicular_vector(vy, vz)
        # Rotate the zero reference planes normal by abduction_adduction angle about wrist-elbow axis
        Re = transformations.rotation_matrix_4x4(left_shoulder_aligned_positions[0] - left_shoulder_aligned_positions[elbow_left_idx], np.radians(abduction_adduction))
        trans_zero_rot_plane_normal = np.matmul(Re, np.append(zero_rot_plane_normal, 1))[:3]

        # Elbow-Shoulder and Elbow-Wrist vectors
        v_es = np.array([sx, sy, sz]) - np.array([ex, ey, ez])
        v_ew = np.array([wx, wy, wz]) - np.array([ex, ey, ez])
        # Compute actual shoulder-elbow-wrist plane normal
        # TODO: In some cases, the sew_normal points to the opposite direction it actually should.
        #       That case leads to very high rotation angle, when they should be 180° - angle.
        #       How can we ensure the correct normal direction / When to substract {angle} from 180° ?
        sew_plane_normal = transformations.get_perpendicular_vector(v_ew, v_es)
        # Calc angle between transformed zero reference plane normal and actual plane normal to get the inner/outer rotation
        inner_outer_rotation = np.degrees(transformations.get_angle(trans_zero_rot_plane_normal, sew_plane_normal))
        """

        flexion_extension_arr.append(flexion_extension)
        abduction_adduction_arr.append(abduction_adduction)
        # inner_outer_rotation_arr.append(inner_outer_rotation)
        
        if log:
            print("\n##### SHOULDER LEFT ANGLES #####")
            print(f"[{frame}] r spherical: {er}")
            print(f"[{frame}] theta spherical: {theta}")
            print(f"[{frame}] phi spherical: {phi}")
            print(f"[{frame}] flexion_extension angle: {flexion_extension} (phi ratio: {phi_ratio_flex_ex})")
            print(f"[{frame}] abduction_adduction angle: {abduction_adduction} (phi ratio: {phi_ratio_abd_add})")
            # print(f"[{frame}] inner_outer_rotation angle: {inner_outer_rotation}")

    ### Plotting ###
    """
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
    """
    return {
        "flexion_extension": flexion_extension_arr,
        "abduction_adduction": abduction_adduction_arr,
        # "inner_outer_rotation": inner_outer_rotation_arr
    }

def calc_angles_shoulder_right(seq: Sequence, shoulder_right_idx: int, shoulder_left_idx: int, neck_idx: int, elbow_right_idx: int, log: bool = False) -> dict:
    """ Calculates Right Shoulder angles 
    Parameters
    ----------
    seq : Sequence
        A Motion Sequence
    """
    flexion_extension_arr = []
    abduction_adduction_arr = []

    # for frame in range(37, 40):
    for frame in range(0, len(seq.positions)):
        # Move coordinate system to left shoulder for frame 20
        # align_coordinates_to(origin_bp_idx: int, x_direction_bp_idx: int, y_direction_bp_idx: int, seq: Sequence, frame: int)
        right_shoulder_aligned_positions = transformations.align_coordinates_to(shoulder_right_idx, shoulder_left_idx, neck_idx, seq, frame=frame)

        vx = transformations.norm(np.array([1, 0, 0]))
        vy = transformations.norm(np.array([0, 1, 0]))
        vz = transformations.norm(np.array([0, 0, 1]))
        ex = right_shoulder_aligned_positions[elbow_right_idx][0]
        ey = right_shoulder_aligned_positions[elbow_right_idx][1]
        ez = right_shoulder_aligned_positions[elbow_right_idx][2]

        # Convert to spherical coordinates
        er = math.sqrt(ex**2 + ey**2 + ez**2)
        # NOTE: Hacky: We assume that the neck coords are above shoulders to determine if Y points up or down
        #       and Z points to front or back. Spherical coords must be calculated differently for each case.
        neck_pos = right_shoulder_aligned_positions[neck_idx]
        # Theta should be the angle between downwards vector and r
        # True: Y-Axis Points up -> Mirror Y-Axis so it points down (to 0° medical definition)
        # False: Y-Axis Points down -> Don't mirror Y-Axis
        theta = math.degrees(math.acos(-ey/er)) if (neck_pos[1] >= 0) else math.degrees(math.acos(ey/er))

        # Phi is the anti-clockwise angle between Z and X
        # True: Y-Axis Points up -> Z points to camera -> don't mirror Z-Axis
        # False: Y-Axis Points down -> Z points away from camera -> mirror Z-Axis
        phi = math.degrees(math.atan2(ez, -ex)) if (neck_pos[1] >= 0) else math.degrees(math.atan2(-ez, -ex))

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
        
        flexion_extension_arr.append(flexion_extension)
        abduction_adduction_arr.append(abduction_adduction)
        
        if log:
            print("\n##### SHOULDER RIGHT ANGLES #####")
            print(f"[{frame}] r spherical: {er}")
            print(f"[{frame}] theta spherical: {theta}")
            print(f"[{frame}] phi spherical: {phi}")
            print(f"[{frame}] flexion_extension angle: {flexion_extension} (phi ratio: {phi_ratio_flex_ex})")
            print(f"[{frame}] abduction_adduction angle: {abduction_adduction} (phi ratio: {phi_ratio_abd_add})")

    ### Plotting ###
    """
    fig = plt.figure(figsize=plt.figaspect(1)*2)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d(-0.5, 0.5)
    for i, p in enumerate(right_shoulder_aligned_positions):
        if i == 1:
            ax.scatter(p[0], p[1], p[2], c="blue")
        else:
            ax.scatter(p[0], p[1], p[2], c="blue")
    ax.plot([right_shoulder_aligned_positions[shoulder_right_idx][0], ex],
            [right_shoulder_aligned_positions[shoulder_right_idx][1], ey],
            [right_shoulder_aligned_positions[shoulder_right_idx][2], ez],
            color="pink", linewidth=1)
    zero_position = right_shoulder_aligned_positions[shoulder_right_idx]

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
    plt.show()
    """
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
