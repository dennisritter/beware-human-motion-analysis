from Exercise import Exercise
from JointsAngleMapper import JointsAngleMapper
from PoseFormatEnum import PoseFormatEnum
import exercise_loader
from Sequence import Sequence
from PoseMapper import PoseMapper
import visualize
import angle_calculations_medical as acm
import numpy as np

# Get Exercise Object from json file
ex = exercise_loader.load('data/exercises/squat.json')
# Get PoseMapper instance for MOCAP sequences
mocap_posemapper = PoseMapper(PoseFormatEnum.MOCAP)
# Convert mocap json string Positions to Sequence Object
seq = mocap_posemapper.load('data/sequences/squat_3/complete-session.json', 'Squat')

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


##### Transformation Matrix M #####
M = [
    [1,0,0,1],
    [0,1,0,1],
    [0,0,1,1],
    [0,0,0,1]
    ]
zero = np.zeros(4)
zero = np.matmul(np.linalg.inv(M), zero)

##### OLD (WRONG) ANGLE CALCULATIONS #####

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

# FRAME = 50
# print(f"Hip Left Flexion/Extension angle [{FRAME}]: {hip_left_flexion_extension_angles[FRAME]}")
# print(f"Hip Right Flexion/Extension angle [{FRAME}]: {hip_right_flexion_extension_angles[FRAME]}")
# print(f"Hip Left Abduction/Adduction angle [{FRAME}]: {hip_left_abduction_adduction_angles[FRAME]}")
# print(f"Hip Right Abduction/Adduction angle [{FRAME}]: {hip_right_abduction_adduction_angles[FRAME]}")
# print(f"Knee Left Flexion/Extension angle [{FRAME}]: {knee_left_flexion_extension_angles[FRAME]}")
# print(f"Knee Right Flexion/Extension angle [{FRAME}]: {knee_right_flexion_extension_angles[FRAME]}")
# print(f"Shoulder Left Flexion/Extension angle [{FRAME}]: {shoulder_left_flexion_extension_angles[FRAME]}")
# print(f"Shoulder Right Flexion/Extension angle [{FRAME}]: {shoulder_right_flexion_extension_angles[FRAME]}")
# print(f"Shoulder Left Abduction/Adduction angle [{FRAME}]: {shoulder_left_abduction_adduction_angles[FRAME]}")
# print(f"Shoulder Right Abduction/Adduction angle [{FRAME}]: {shoulder_right_abduction_adduction_angles[FRAME]}")
# print(f"Elbow Left Flexion/Extension angle [{FRAME}]: {elbow_left_flexion_extension_angles[FRAME]}")
# print(f"Elbow Right Flexion/Extension angle [{FRAME}]: {elbow_right_flexion_extension_angles[FRAME]}")

# Visualize angle
# visualize.vis_angle(seq, joints["shoulder_right"]["abduction_adduction"], FRAME)
