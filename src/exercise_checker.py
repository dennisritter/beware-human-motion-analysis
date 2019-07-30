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
    # If theta 180° (dot product = -1)
    v1 = norm(v1)
    v2 = norm(v2)

    if (np.dot(v1, v2) == -1):
        # TODO: Arbitrary Vector.. Find method to ensure its not parallel to vsx
        return np.cross(np.array([3, 2, 1]), v2)
    else:
        return norm(np.cross(v1, v2))


def norm(v):
    return v / math.sqrt(np.dot(v, v))


def rotation_matrix_4x4(axis, theta):
    # Source: https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta in radians as 4x4 Transformation Matrix
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


def translation_matrix_4x4(v):
    T = np.array([
        [1.0, 0, 0, 0],
        [0, 1.0, 0, 0],
        [0, 0, 1.0, 0],
        [0, 0, 0, 1.0]
    ])
    T[:3, 3] = v
    return T


FRAME = 20

# Get Exercise Object from json file
ex = exercise_loader.load('data/exercises/squat.json')
# Get PoseMapper instance for MOCAP sequences
mocap_posemapper = PoseMapper(PoseFormatEnum.MOCAP)
# Convert mocap json string Positions to Sequence Object
seq = mocap_posemapper.load('data/sequences/squat_3/complete-session.json', 'Squat')


# Target origin (0,0,0) so shoulder is at (0,0,0) after transforming
target_cs_origin = np.array([0, 0, 0])
# Axesdirection as tracked
vtx = norm(np.array([1, 0, 0]))
vty = norm(np.array([0, 1, 0]))
vtz = norm(np.array([0, 0, 1]))

# Start origin at shoulder keypoint
start_cs_origin = seq.positions[FRAME][2]
# x is vector direction to other shoulder
vsx = seq.positions[FRAME][14] - start_cs_origin
vneck = seq.positions[FRAME][3] - start_cs_origin
vsz = get_perpendicular_vector(vneck, vsx)
# find vector perpendicular to xy-plane
vsy = get_perpendicular_vector(vsx, vsz)

print(vsx, vsy, vsz)
print(np.dot(vsx, vsz), np.dot(vsx, vsy), np.dot(vsy, vsz))
axis = get_perpendicular_vector(vsx, vtx)
theta = get_angle(vsx, vtx)
print(f"Theta: {theta} ({np.degrees(theta)}°)")
R = rotation_matrix_4x4(axis, theta)

# Rotate X to use it as axis for y rotation
temp_x_rot = np.matmul(R, np.append(vsx, 1))[:3]
# Rotate y direction vector to check for Y-Angle
temp_y_rot = np.matmul(R, np.append(vsy, 1))[:3]

# start_dir_x_transformed = np.matmul(R, np.append(start_dir_x_transformed, 1))[:3]
# start_dir_y_transformed = np.matmul(R, np.append(start_dir_y_transformed, 1))[:3]
# start_dir_z_transformed = np.matmul(R, np.append(start_dir_z_transformed, 1))[:3]
###########################################
# axis2 = get_perpendicular_vector(temp_y_rot, vty)
theta2 = get_angle(temp_y_rot, vty)
print(f"Theta2: {theta2} ({np.degrees(theta2)}°)")
R2 = rotation_matrix_4x4(norm(temp_x_rot), theta2)
# start_dir_x_transformed = np.matmul(R2, np.append(start_dir_x_transformed, 1))[:3]
# start_dir_y_transformed = np.matmul(R2, np.append(start_dir_y_transformed, 1))[:3]
# start_dir_z_transformed = np.matmul(R2, np.append(start_dir_z_transformed, 1))[:3]
###########################################
T = translation_matrix_4x4(target_cs_origin-start_cs_origin)
# print(target_cs_origin - start_cs_origin)
# print(f"T: \n {T}")
# start_dir_x_transformed = np.matmul(T, np.append(start_dir_x_transformed, 1))[:3]
# start_dir_y_transformed = np.matmul(T, np.append(start_dir_y_transformed, 1))[:3]
# start_dir_z_transformed = np.matmul(T, np.append(start_dir_z_transformed, 1))[:3]
transformed_positions = []
for bp_pos in seq.positions[FRAME]:
    pos = bp_pos
    pos = np.matmul(T, np.append(pos, 1))[:3]
    pos = np.matmul(R2, np.append(pos, 1))[:3]
    pos = np.matmul(R, np.append(pos, 1))[:3]
    transformed_positions.append(pos)


fig = plt.figure(figsize=plt.figaspect(1)*2)
ax = fig.add_subplot(1, 1, 1, projection='3d')

for i, p in enumerate(transformed_positions):
    ax.scatter(p[0], p[1], p[2], c="blue")
    print(f"{i, p}")
for j in range(len(seq.positions[FRAME])):
    ax.scatter(seq.positions[FRAME][j][0], seq.positions[FRAME][j][1], seq.positions[FRAME][j][2], c="red", alpha=0.5)
    ax.text(seq.positions[FRAME][j][0], seq.positions[FRAME][j][1], seq.positions[FRAME][j][2], j)
    ax.annotate(f"{j}", (seq.positions[FRAME][j][0], seq.positions[FRAME][j][1]))
ax.plot([target_cs_origin[0], vtx[0]], [target_cs_origin[1], vtx[1]], [target_cs_origin[2], vtx[2]], color="pink", linewidth=1)
ax.plot([target_cs_origin[0], vty[0]], [target_cs_origin[1], vty[1]], [target_cs_origin[2], vty[2]], color="maroon", linewidth=1)
ax.plot([target_cs_origin[0], vtz[0]], [target_cs_origin[1], vtz[1]], [target_cs_origin[2], vtz[2]], color="red", linewidth=1)
# ax.plot([target_cs_origin[0], 1], [target_cs_origin[1], 0], [target_cs_origin[2], 0], color="pink", linewidth=1)
# ax.plot([target_cs_origin[0], 0], [target_cs_origin[1], 1], [target_cs_origin[2], 0], color="maroon", linewidth=1)
# ax.plot([target_cs_origin[0], 0], [target_cs_origin[1], 0], [target_cs_origin[2], 1], color="red", linewidth=1)
# ax.plot([start_cs_origin[0], 1+start_cs_origin[0]], [start_cs_origin[1], start_cs_origin[1]], [start_cs_origin[2], start_cs_origin[2]], color="pink", linewidth=1)
# ax.plot([start_cs_origin[0], start_cs_origin[0]], [start_cs_origin[1], 1+start_cs_origin[1]], [start_cs_origin[2], start_cs_origin[2]], color="maroon", linewidth=1)
# ax.plot([start_cs_origin[0], start_cs_origin[0]], [start_cs_origin[1], start_cs_origin[1]], [start_cs_origin[2], 1+start_cs_origin[2]], color="red", linewidth=1)
# ax.plot([start_cs_origin[0], vsx[0]+start_cs_origin[0]], [start_cs_origin[1], vsx[1]+start_cs_origin[1]], [start_cs_origin[2], vsx[2]+start_cs_origin[2]], color="green", linewidth=1)
# ax.plot([start_cs_origin[0], vsy[0]+start_cs_origin[0]], [start_cs_origin[1], vsy[1]+start_cs_origin[1]], [start_cs_origin[2], vsy[2]+start_cs_origin[2]], color="springgreen", linewidth=1)
# ax.plot([start_cs_origin[0], vsz[0]+start_cs_origin[0]], [start_cs_origin[1], vsz[1]+start_cs_origin[1]], [start_cs_origin[2], vsz[2]+start_cs_origin[2]], color="springgreen", linewidth=1)

# ax.plot([start_cs_origin[0], vneck[0]+start_cs_origin[0]], [start_cs_origin[1], vneck[1]+start_cs_origin[1]], [start_cs_origin[2], vneck[2]+start_cs_origin[2]], color="green", linewidth=1)
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
# Visualize angle
# visualize.vis_angle(seq, joints["shoulder_right"]["abduction_adduction"], FRAME)
