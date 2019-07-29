import tslearn.metrics as ts
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from Exercise import Exercise
from JointsAngleMapper import JointsAngleMapper
from PoseFormatEnum import PoseFormatEnum
import exercise_loader
from Sequence import Sequence
from PoseMapper import PoseMapper
import visualize
import angle_calculations_medical as acm
import numpy as np
import math


# TODO:
# 1. Calc perpendicular vector for start_dir_x and target_dir_x
# 2. Calc Angle between start_dir_x and target_dir_x
# 3. Build R0 Matrix for that rotation
# 4. Multiply R0 with T
# 5. R0 dot T = M -> MatMul with all Start vectors
# 6. Calc Angle between start_dir_y and target_dir_y
# 7. Build R1 Matrix (Rotation around X-Axis)
# 8. M' = M MatMul R1 = R1*R0*T
# 9. Transform all points with M'


# Beispiel Schulter Winkelberechnung Schritte:
# -> Koordinatensystem in Punkt verschieben (keypoint Schulter)
# -> X-Achse bestimmen (Schulter-Schulter)
# -> Z-Achse bestimmen (Schulter-Nacken-Schulter Dreieck Normale)
# -> Y-Achse orthogonal zu XZ Ebene
##
# -> Transformationsmatrix bauen
# -> Translation zu Schulter
# -> Rotationen der Achsen
##
# -> Alle für den Winkel benötigten Punkte transformieren
# -> Winkel in Kugelkoordinaten ausrechnen
# -> Medizinische Winkel ableiten

def get_angle(v1, v2):
    return np.arccos(np.dot(norm(v1), norm(v2)))


def get_perpendicular_vector(v1, v2):
    # If theta 180° (dot product = -1 just switch directions
    v1 = norm(v1)
    v2 = norm(v2)

    if (np.dot(v1, v2) == -1):
        # TODO: Arbitrary Vector.. Find method to ensure its not parallel to vsx
        return np.cross(np.array([3, 2, 1]), v2)
    else:
        return np.cross(v1, v2)


def norm(v):
    return v / math.sqrt(np.dot(v, v))


def rotation_matrix4x4(axis, theta):
    # Source: https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians as 4x4 Transformation Matrix
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0],
                     [0, 0, 0, 1]])


def tranlation_matrix4x4(v):
    T = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    T[:3, 3] = v
    return T


# Get Exercise Object from json file
ex = exercise_loader.load('data/exercises/squat.json')
# Get PoseMapper instance for MOCAP sequences
mocap_posemapper = PoseMapper(PoseFormatEnum.MOCAP)
# Convert mocap json string Positions to Sequence Object
seq = mocap_posemapper.load('data/sequences/squat_3/complete-session.json', 'Squat')

start_cs_origin = np.array([0, 0, 0])
start_dir_x = np.array([1, 0, 0])
start_dir_y = np.array([0, 1, 0])
start_dir_z = np.array([0, 0, 1])
target_cs_origin = np.array([1, 1, 1])
target_dir_x = np.array([1, 0, 1])
target_dir_y = np.array([0, 1, 1])
# find vector perpendicular to xy-plane
target_dir_z = np.cross(target_dir_y, target_dir_x)
print(target_dir_z)

vsx = norm(start_dir_x - start_cs_origin)
vsy = norm(start_dir_y - start_cs_origin)
vsz = norm(start_dir_z - start_cs_origin)
vtx = norm(target_dir_x - target_cs_origin)
vty = norm(target_dir_y - target_cs_origin)
vtz = norm(target_dir_z - target_cs_origin)

start_dir_x_transformed = start_dir_x
start_dir_y_transformed = start_dir_y
start_dir_z_transformed = start_dir_z

axis = get_perpendicular_vector(vsx, vtx)
theta = get_angle(vsx, vtx)
print(f"Theta: {theta} ({np.degrees(theta)}°)")
R = rotation_matrix4x4(axis, theta)
start_dir_x_transformed = np.matmul(R, np.append(start_dir_x_transformed, 1))[:3]
start_dir_y_transformed = np.matmul(R, np.append(start_dir_y_transformed, 1))[:3]
start_dir_z_transformed = np.matmul(R, np.append(start_dir_z_transformed, 1))[:3]
###########################################
axis2 = get_perpendicular_vector(start_dir_y_transformed, vty)
theta2 = get_angle(start_dir_y_transformed, vty)
print(f"Theta2: {theta2} ({np.degrees(theta2)}°)")
R2 = rotation_matrix4x4(norm(start_dir_x_transformed), theta2)
start_dir_x_transformed = np.matmul(R2, np.append(start_dir_x_transformed, 1))[:3]
start_dir_y_transformed = np.matmul(R2, np.append(start_dir_y_transformed, 1))[:3]
start_dir_z_transformed = np.matmul(R2, np.append(start_dir_z_transformed, 1))[:3]
###########################################
# axis3 = get_perpendicular_vector(start_dir_z_transformed, vtz)
# theta3 = get_angle(start_dir_z_transformed, vtz)
# print(f"Theta3: {theta3} ({np.degrees(theta3)}°)")
# R3 = rotation_matrix4x4(start_dir_x_transformed, theta3)
# start_dir_x_transformed = np.matmul(R3, np.append(start_dir_x_transformed, 1))[:3]
# start_dir_y_transformed = np.matmul(R3, np.append(start_dir_y_transformed, 1))[:3]
# start_dir_z_transformed = np.matmul(R3, np.append(start_dir_z_transformed, 1))[:3]
###########################################
T = tranlation_matrix4x4(target_cs_origin)
start_dir_x_transformed = np.matmul(T, np.append(start_dir_x_transformed, 1))[:3]
start_dir_y_transformed = np.matmul(T, np.append(start_dir_y_transformed, 1))[:3]
start_dir_z_transformed = np.matmul(T, np.append(start_dir_z_transformed, 1))[:3]

vtx += target_cs_origin
vty += target_cs_origin
vtz += target_cs_origin
axis += target_cs_origin

# Check angle from transformed start to target
print(f"{get_angle(start_dir_x_transformed, vtx)} ({np.degrees(get_angle(start_dir_x_transformed, vtx))}°)")


fig = plt.figure(figsize=plt.figaspect(1)*2)
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlim3d(-2, 2)
ax.set_ylim3d(-2, 2)
ax.set_zlim3d(-2, 2)
ax.scatter(start_cs_origin[0], start_cs_origin[1], start_cs_origin[2], c='blue')
# ax.scatter(target_cs_origin[0], target_cs_origin[1], target_cs_origin[2], c='red')
ax.plot([start_cs_origin[0], start_dir_x[0]], [start_cs_origin[1], start_dir_x[1]], [start_cs_origin[2], start_dir_x[2]], color="black")
ax.plot([start_cs_origin[0], start_dir_y[0]], [start_cs_origin[1], start_dir_y[1]], [start_cs_origin[2], start_dir_y[2]], color="gray")
ax.plot([start_cs_origin[0], start_dir_z[0]], [start_cs_origin[1], start_dir_z[1]], [start_cs_origin[2], start_dir_z[2]], color="silver")
# ax.plot([target_cs_origin[0], vtx[0]], [target_cs_origin[1], vtx[1]], [target_cs_origin[2], vtx[2]], color="red", linewidth=3)
# ax.plot([target_cs_origin[0], vty[0]], [target_cs_origin[1], vty[1]], [target_cs_origin[2], vty[2]], color="red", linewidth=3)
# ax.plot([target_cs_origin[0], vtz[0]], [target_cs_origin[1], vtz[1]], [target_cs_origin[2], vtz[2]], color="red", linewidth=3)
ax.plot([target_cs_origin[0], target_dir_x[0]], [target_cs_origin[1], target_dir_x[1]], [target_cs_origin[2], target_dir_x[2]], color="pink", linewidth=3)
ax.plot([target_cs_origin[0], target_dir_y[0]], [target_cs_origin[1], target_dir_y[1]], [target_cs_origin[2], target_dir_y[2]], color="maroon", linewidth=3)
ax.plot([target_cs_origin[0], target_dir_z[0]], [target_cs_origin[1], target_dir_z[1]], [target_cs_origin[2], target_dir_z[2]], color="red", linewidth=3)
# ax.plot([trans_cs_origin[0], trans_dir_x[0]], [trans_cs_origin[1], trans_dir_x[1]], [trans_cs_origin[2], trans_dir_x[2]], color="black")
# ax.plot([trans_cs_origin[0], trans_dir_y[0]], [trans_cs_origin[1], trans_dir_y[1]], [trans_cs_origin[2], trans_dir_y[2]], color="green")
# ax.plot([trans_cs_origin[0], trans_dir_z[0]], [trans_cs_origin[1], trans_dir_z[1]], [trans_cs_origin[2], trans_dir_z[2]], color="green")
# ax.plot([trans_cs_origin[0], k[0]], [trans_cs_origin[1], k[1]], [trans_cs_origin[2], k[2]], color="red", linestyle="dotted")

# ax.plot([target_cs_origin[0], vsx[0]], [target_cs_origin[1], vsx[1]], [target_cs_origin[2], vsx[2]], color="blue")
ax.plot([target_cs_origin[0], start_dir_x_transformed[0]], [target_cs_origin[1], start_dir_x_transformed[1]], [target_cs_origin[2], start_dir_x_transformed[2]], color="olive")
ax.plot([target_cs_origin[0], start_dir_y_transformed[0]], [target_cs_origin[1], start_dir_y_transformed[1]], [target_cs_origin[2], start_dir_y_transformed[2]], color="springgreen")
ax.plot([target_cs_origin[0], start_dir_z_transformed[0]], [target_cs_origin[1], start_dir_z_transformed[1]], [target_cs_origin[2], start_dir_z_transformed[2]], color="green")
# ax.plot([target_cs_origin[0], vtx[0]], [target_cs_origin[1], vtx[1]], [target_cs_origin[2], vtx[2]], color="red", alpha=.5)
ax.plot([target_cs_origin[0], axis[0]], [target_cs_origin[1], axis[1]], [target_cs_origin[2], axis[2]], color="black", linestyle="dotted")
plt.show()


""" LEGACY CODE

# Add joints to angles property of exercise
# jam = JointsAngleMapper(PoseFormatEnum.MOCAP)
# jam.addJointsToAngles(ex)

# joints = jam.jointsMap
# # Calculate angles for Sequence
# # Left Hip Flexion/Extension
# hip_left_flexion_extension_angles = acm.calc_angle_hip_flexion_extension(seq, joints["hip_left"]["flexion_extension"])
# # Right Hip Flexion/Extension
# hip_right_flexion_extension_angles = acm.calc_angle_hip_flexion_extension(seq, joints["hip_right"]["flexion_extension"])
# # Left Hip Abduction/Adduction
# hip_left_abduction_adduction_angles = acm.calc_angle_hip_abduction_adduction(seq, joints["hip_left"]["abduction_adduction"])
# # Right Hip Abduction/Adduction
# hip_right_abduction_adduction_angles = acm.calc_angle_hip_abduction_adduction(seq, joints["hip_right"]["abduction_adduction"])

# # Left Knee Flexion/Extension
# knee_left_flexion_extension_angles = acm.calc_angle_knee_flexion_extension(seq, joints["knee_left"]["flexion_extension"])
# # Right Knee Flexion/Extension
# knee_right_flexion_extension_angles = acm.calc_angle_knee_flexion_extension(seq, joints["knee_right"]["flexion_extension"])

# # Left Elbow Flexion/Extension
# shoulder_left_flexion_extension_angles = acm.calc_angle_shoulder_flexion_extension(seq, joints["shoulder_left"]["flexion_extension"])
# # Right Elbow Flexion/Extension
# shoulder_right_flexion_extension_angles = acm.calc_angle_shoulder_flexion_extension(seq, joints["shoulder_right"]["flexion_extension"])
# # Left Elbow Abduction/Adduction
# shoulder_left_abduction_adduction_angles = acm.calc_angle_shoulder_abduction_adduction(seq, joints["shoulder_left"]["abduction_adduction"])
# # Right Elbow Abduction/Adduction
# shoulder_right_abduction_adduction_angles = acm.calc_angle_shoulder_abduction_adduction(seq, joints["shoulder_right"]["abduction_adduction"])

# # Left Elbow Flexion/Extension
# elbow_left_flexion_extension_angles = acm.calc_angle_elbow_flexion_extension(seq, joints["elbow_left"]["flexion_extension"])
# # Right Elbow Flexion/Extension
# elbow_right_flexion_extension_angles = acm.calc_angle_elbow_flexion_extension(seq, joints["elbow_right"]["flexion_extension"])

FRAME = 0
print(f"Hip Left Flexion/Extension angle [{FRAME}]: {hip_left_flexion_extension_angles[FRAME]}")
print(f"Hip Right Flexion/Extension angle [{FRAME}]: {hip_right_flexion_extension_angles[FRAME]}")
print(f"Hip Left Abduction/Adduction angle [{FRAME}]: {hip_left_abduction_adduction_angles[FRAME]}")
print(f"Hip Right Abduction/Adduction angle [{FRAME}]: {hip_right_abduction_adduction_angles[FRAME]}")
print(f"Knee Left Flexion/Extension angle [{FRAME}]: {knee_left_flexion_extension_angles[FRAME]}")
print(f"Knee Right Flexion/Extension angle [{FRAME}]: {knee_right_flexion_extension_angles[FRAME]}")
print(f"Shoulder Left Flexion/Extension angle [{FRAME}]: {shoulder_left_flexion_extension_angles[FRAME]}")
print(f"Shoulder Right Flexion/Extension angle [{FRAME}]: {shoulder_right_flexion_extension_angles[FRAME]}")
print(f"Shoulder Left Abduction/Adduction angle [{FRAME}]: {shoulder_left_abduction_adduction_angles[FRAME]}")
print(f"Shoulder Right Abduction/Adduction angle [{FRAME}]: {shoulder_right_abduction_adduction_angles[FRAME]}")
print(f"Elbow Left Flexion/Extension angle [{FRAME}]: {elbow_left_flexion_extension_angles[FRAME]}")
print(f"Elbow Right Flexion/Extension angle [{FRAME}]: {elbow_right_flexion_extension_angles[FRAME]}")

# Visualize angle
visualize.vis_angle(seq, joints["shoulder_right"]["abduction_adduction"], FRAME)
"""
