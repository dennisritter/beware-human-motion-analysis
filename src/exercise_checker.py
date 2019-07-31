from Exercise import Exercise
from PoseFormatEnum import PoseFormatEnum
import exercise_loader
from Sequence import Sequence
from PoseMapper import PoseMapper
import angle_calculations_medical as acm
import numpy as np
import transformations
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


# Get Exercise Object from json file
ex = exercise_loader.load('data/exercises/squat.json')
# Get PoseMapper instance for MOCAP sequences
mocap_posemapper = PoseMapper(PoseFormatEnum.MOCAP)
# Convert mocap json string Positions to Sequence Object
seq = mocap_posemapper.load('data/sequences/squat_3/complete-session.json', 'Squat')

# Move coordinate system to left shoulder for frame 20
# align_coordinates_to(origin_bp_idx: int, x_direction_bp_idx: int, y_direction_bp_idx: int, seq: Sequence, frame: int)
left_shoulder_aligned_positions = transformations.align_coordinates_to(2, 14, 3, seq, frame=50)

# Transform postitions from Cartesian coordinates to Spherical coordinates
# Example for left shoulder using left elbow to check angles
# x = -0.1
# y = 0.05
# z = -0.1
x = left_shoulder_aligned_positions[1][0]
y = left_shoulder_aligned_positions[1][1]
z = left_shoulder_aligned_positions[1][2]
# print(f"Joint Pos for Angle Calculation: \n{x,y,z}")

# r = math.sqrt(x**2 + y**2 + z**2)
# theta_flex = math.acos(z/r)
# theta_abd = math.acos(x/r)
# # 90° rotation to get angle to downward axis (-Y) -> medical 0°
# theta_flex_med = theta_flex - np.radians(90)
# theta_abd_med = theta_abd - np.radians(90)
# if y > 0:
#     theta_flex_med = np.radians(180) - theta_flex_med
# if x > 0:
#     theta_abd_med = np.radians(180) - theta_abd_med

# print(f"theta_flex: {theta_flex} ({np.degrees(theta_flex)})")
# print(f"theta_abd: {theta_abd} ({np.degrees(theta_abd)})")
# print(f"theta_flex_med: {theta_flex_med} ({np.degrees(theta_flex_med)})")
# print(f"theta_abd_med: {theta_abd_med} ({np.degrees(theta_abd_med)})")

# No Transformations needed because no spherical coords needed??
# FLEXION
# Theta < 0 : Flexion
# Theta > 0 : Extension
# Y > 0 : Theta' = 180 - Theta
# Y < 0 : Theta' = Theta
a = np.array([x, y, z])
b = np.array([0, 0, 1])
# a_dot_b = np.dot(a, b) / (transformations.norm(a) * transformatiorns.norm(b))
a_dot_b = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
theta = np.degrees(np.arccos(a_dot_b) - np.radians(90))
theta = abs(theta)
# if y > 0:
#     theta = 180 - theta
if z < 0:
    print(f"Flexion: {theta}°")
elif z > 0:
    print(f"Extension: {theta}°")
elif z == 0:
    print(f"Flexion/Extension: {0}°")

# ABDUCTION
# Theta < 0 : Abduction
# Theta > 0 : Adduction
# Y > 0 : Theta' = 180 - Theta
# Y < 0 : Theta' = Theta
a = np.array([x, y, z])
b = np.array([1, 0, 0])
# a_dot_b = np.dot(a, b) / (transformations.norm(a) * transformatiorns.norm(b))
a_dot_b = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
theta = np.degrees(np.arccos(a_dot_b) - np.radians(90))
theta = abs(theta)
# if y > 0:
#     theta = 180 - theta
if x < 0:
    print(f"Abduction: {theta}°")
elif x > 0:
    print(f"Adduction: {theta}°")
elif x == 0:
    print(f"Abduction/Adduction: {0}°")
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

frame = 0
print(f"Hip Left Flexion/Extension angle [{frame}]: {hip_left_flexion_extension_angles[frame]}")
print(f"Hip Right Flexion/Extension angle [{frame}]: {hip_right_flexion_extension_angles[frame]}")
print(f"Hip Left Abduction/Adduction angle [{frame}]: {hip_left_abduction_adduction_angles[frame]}")
print(f"Hip Right Abduction/Adduction angle [{frame}]: {hip_right_abduction_adduction_angles[frame]}")
print(f"Knee Left Flexion/Extension angle [{frame}]: {knee_left_flexion_extension_angles[frame]}")
print(f"Knee Right Flexion/Extension angle [{frame}]: {knee_right_flexion_extension_angles[frame]}")
print(f"Shoulder Left Flexion/Extension angle [{frame}]: {shoulder_left_flexion_extension_angles[frame]}")
print(f"Shoulder Right Flexion/Extension angle [{frame}]: {shoulder_right_flexion_extension_angles[frame]}")
print(f"Shoulder Left Abduction/Adduction angle [{frame}]: {shoulder_left_abduction_adduction_angles[frame]}")
print(f"Shoulder Right Abduction/Adduction angle [{frame}]: {shoulder_right_abduction_adduction_angles[frame]}")
print(f"Elbow Left Flexion/Extension angle [{frame}]: {elbow_left_flexion_extension_angles[frame]}")
print(f"Elbow Right Flexion/Extension angle [{frame}]: {elbow_right_flexion_extension_angles[frame]}")

# Visualize angle
visualize.vis_angle(seq, joints["shoulder_right"]["abduction_adduction"], frame)
"""
# Visualize angle
# visualize.vis_angle(seq, joints["shoulder_right"]["abduction_adduction"], frame)
