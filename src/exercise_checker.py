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
left_shoulder_aligned_positions = transformations.align_coordinates_to(2, 14, 3, seq, frame=20)

# Transform postitions from Cartesian coordinates to Spherical coordinates
# Example for left shoulder using left elbow to check angles
x = left_shoulder_aligned_positions[1][0]
y = left_shoulder_aligned_positions[1][1]
z = left_shoulder_aligned_positions[1][2]
print(f"Joint Pos for Angle Calculation: \n{x,y,z}")
# x = 1
# y = 1
# z = 1
# r = 0.3
# theta = np.radians(0)
# phi = np.radians(10)
r = math.sqrt(x**2 + y**2 + z**2)
theta = math.acos(z/r)
phi = math.atan2(y, x)
theta_med = -(math.acos(z/r) - np.radians(270))
phi_med = math.atan2(y, x) - np.radians(90)
print(f"r: {r}")
print(f"theta: {theta} ({np.degrees(theta)})")
print(f"phi: {phi} ({np.degrees(phi)})")
# Flexion(+)/Extension(-) --> y < Flexion / y > 0 Extension
print(f"theta_med: {theta_med} ({np.degrees(theta_med)})")
# x < 0 = Abduction / x > 0 Adduction
print(f"phi_med: {phi_med} ({np.degrees(phi_med)})")
print(f"x: {r*math.sin(theta)*math.cos(phi)}")
print(f"y: {r*math.sin(theta)*math.sin(phi)}")
print(f"z: {r*math.cos(theta)}")

# x = r*cos(phi)*cos(theta)
# y = r*sin(phi)*cos(theta)
# z = r*sin(theta)


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
