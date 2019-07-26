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

# Source: https://stackoverflow.com/questions/6802577/rotation-of-3d-vector


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


# Get Exercise Object from json file
ex = exercise_loader.load('data/exercises/squat.json')
# Get PoseMapper instance for MOCAP sequences
mocap_posemapper = PoseMapper(PoseFormatEnum.MOCAP)
# Convert mocap json string Positions to Sequence Object
seq = mocap_posemapper.load('data/sequences/squat_3/complete-session.json', 'Squat')

start_cs_origin = np.array([0, 0, 0])
start_dir_x = np.array([.5, 1, 0])
start_dir_y = np.array([0, 1, 0])
start_dir_z = np.array([0, 0, 1])
target_cs_origin = np.array([1, 1, 1])
target_dir_x = np.array([-1, 0, 0])
target_dir_y = np.array([0, -1, 0])
# find vector perpendicular to xy-plane
target_dir_z = np.cross(target_dir_x, target_dir_y)

I = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
# TRANSLATION
T = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
# Translation to target origin
# Add translation values to M
T[:3, 3] = target_cs_origin
# Multiply M with start origin to translate it to target origin
trans_cs_origin = np.matmul(T, np.append(start_cs_origin, 1))[:3]
trans_dir_x = np.matmul(T, np.append(start_dir_x, 1))[:3]
trans_dir_y = np.matmul(T, np.append(start_dir_y, 1))[:3]
trans_dir_z = np.matmul(T, np.append(start_dir_z, 1))[:3]
# Find X-rotation angle
# Source http: // www.euclideanspace.com/maths/algebra/vectors/angleBetween/index.htm
v1 = (trans_dir_x - trans_cs_origin) / math.sqrt(np.dot(trans_dir_x - trans_cs_origin, trans_dir_x - trans_cs_origin))
v2 = (target_dir_x - target_cs_origin) / math.sqrt(np.dot(target_dir_x - target_cs_origin, target_dir_x - target_cs_origin))
v1_dot_v2 = np.dot(v1, v2)
v1_cross_v2 = np.cross(v1, v2)
# print(f"theta_cross v1: {np.degrees(np.arccos(np.dot(v1, v1_cross_v2)))}")
# print(f"theta_cross v2: {np.degrees(np.arccos(np.dot(v2, v1_cross_v2)))}")
# If theta 180° (dot product = -1 just switch directions
if (v1_dot_v2 == -1):
    print("v1_dot_v2 = 0")
    v1 = v1 * -1 + target_cs_origin
else:
    theta = np.arccos(v1_dot_v2)
    print(f"Theta: {theta}")
    # Build Rotation Matrix for rotation about arbitrary axis (Rodrigues formula)
    # Source https: // en.wikipedia.org/wiki/Rodrigues % 27_rotation_formula
    axis = v1_cross_v2
    R0 = rotation_matrix(axis, theta)
    v1_new = np.dot(R0, v1)

print(v1)
print(v1_new)
print(np.dot(v1, v1_new))
print(np.degrees(np.arccos(np.dot(v1, v1_new))))

v1 += target_cs_origin
v1_new += target_cs_origin
v2 += target_cs_origin
v1_cross_v2 += target_cs_origin

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


fig = plt.figure(figsize=plt.figaspect(1)*2)
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlim3d(-2, 2)
ax.set_ylim3d(-2, 2)
ax.set_zlim3d(-2, 2)
# ax.scatter(start_cs_origin[0], start_cs_origin[1], start_cs_origin[2], c='blue')
# ax.scatter(target_cs_origin[0], target_cs_origin[1], target_cs_origin[2], c='red')
# ax.plot([start_cs_origin[0], start_dir_x[0]], [start_cs_origin[1], start_dir_x[1]], [start_cs_origin[2], start_dir_x[2]], color="black")
# ax.plot([start_cs_origin[0], start_dir_y[0]], [start_cs_origin[1], start_dir_y[1]], [start_cs_origin[2], start_dir_y[2]], color="blue")
# ax.plot([start_cs_origin[0], start_dir_z[0]], [start_cs_origin[1], start_dir_z[1]], [start_cs_origin[2], start_dir_z[2]], color="blue")
# ax.plot([target_cs_origin[0], target_dir_x[0]], [target_cs_origin[1], target_dir_x[1]], [target_cs_origin[2], target_dir_x[2]], color="pink")
# ax.plot([target_cs_origin[0], target_dir_y[0]], [target_cs_origin[1], target_dir_y[1]], [target_cs_origin[2], target_dir_y[2]], color="red")
# ax.plot([target_cs_origin[0], target_dir_z[0]], [target_cs_origin[1], target_dir_z[1]], [target_cs_origin[2], target_dir_z[2]], color="red")
# ax.plot([trans_cs_origin[0], trans_dir_x[0]], [trans_cs_origin[1], trans_dir_x[1]], [trans_cs_origin[2], trans_dir_x[2]], color="black")
# ax.plot([trans_cs_origin[0], trans_dir_y[0]], [trans_cs_origin[1], trans_dir_y[1]], [trans_cs_origin[2], trans_dir_y[2]], color="green")
# ax.plot([trans_cs_origin[0], trans_dir_z[0]], [trans_cs_origin[1], trans_dir_z[1]], [trans_cs_origin[2], trans_dir_z[2]], color="green")
# ax.plot([trans_cs_origin[0], k[0]], [trans_cs_origin[1], k[1]], [trans_cs_origin[2], k[2]], color="red", linestyle="dotted")

ax.plot([target_cs_origin[0], v1[0]], [target_cs_origin[1], v1[1]], [target_cs_origin[2], v1[2]], color="blue")
ax.plot([target_cs_origin[0], v1_new[0]], [target_cs_origin[1], v1_new[1]], [target_cs_origin[2], v1_new[2]], color="green")
ax.plot([target_cs_origin[0], v2[0]], [target_cs_origin[1], v2[1]], [target_cs_origin[2], v2[2]], color="red", alpha=.5)
ax.plot([target_cs_origin[0], v1_cross_v2[0]], [target_cs_origin[1], v1_cross_v2[1]], [target_cs_origin[2], v1_cross_v2[2]], color="black", linestyle="dotted")
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
