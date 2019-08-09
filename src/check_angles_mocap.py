from movement_analysis.Exercise import Exercise
from movement_analysis.PoseFormatEnum import PoseFormatEnum
from movement_analysis.AngleTargetStates import AngleTargetStates
from movement_analysis.Sequence import Sequence
from movement_analysis.PoseMapper import PoseMapper
from movement_analysis import exercise_loader
from movement_analysis import angle_calculations_medical as acm
from movement_analysis import transformations
from movement_analysis import logging
import math
import matplotlib.pyplot as plt
import numpy as np

FRAME = 50

# Get Exercise Object from json file
ex = exercise_loader.load('data/exercises/kniebeuge.json')
# Get PoseMapper instance for MOCAP sequences
mocap_posemapper = PoseMapper(PoseFormatEnum.MOCAP)
# Convert mocap json string Positions to Sequence Object
seq = mocap_posemapper.load('data/sequences/squat_3/complete-session.json', 'Squat')
bp = seq.body_parts

left_shoulder_angles = acm.calc_angles_shoulder_left(seq, seq.body_parts["LeftShoulder"], seq.body_parts["RightShoulder"], seq.body_parts["Neck"], seq.body_parts["LeftElbow"])
right_shoulder_angles = acm.calc_angles_shoulder_right(seq, seq.body_parts["RightShoulder"], seq.body_parts["LeftShoulder"], seq.body_parts["Neck"], seq.body_parts["RightElbow"])
left_hip_angles = acm.calc_angles_hip_left(seq, seq.body_parts["LeftHip"], seq.body_parts["RightHip"], seq.body_parts["Torso"], seq.body_parts["LeftKnee"])
right_hip_angles = acm.calc_angles_hip_right(seq, seq.body_parts["RightHip"], seq.body_parts["LeftHip"], seq.body_parts["Torso"], seq.body_parts["RightKnee"])
left_elbow_angles = acm.calc_angles_elbow(seq, seq.body_parts["LeftElbow"], seq.body_parts["LeftShoulder"], seq.body_parts["LeftWrist"])
right_elbow_angles = acm.calc_angles_elbow(seq, seq.body_parts["RightElbow"], seq.body_parts["RightShoulder"], seq.body_parts["RightWrist"])
left_knee_angles = acm.calc_angles_knee(seq, seq.body_parts["LeftKnee"], seq.body_parts["LeftHip"], seq.body_parts["LeftAnkle"])
right_knee_angles = acm.calc_angles_knee(seq, seq.body_parts["RightKnee"], seq.body_parts["RightHip"], seq.body_parts["RightAnkle"])

# Check left shoulder FlexEx angles
results= []
for angle in left_shoulder_angles["flexion_extension"]:
    results.append(ex._check_angle_shoulder_left_flexion_extension(angle, AngleTargetStates.END, 10))

print(left_shoulder_angles["flexion_extension"])
print(right_shoulder_angles["flexion_extension"])
# print(results[FRAME - 1])


# logging.log_angles(left_shoulder_angles,
#                    right_shoulder_angles,
#                    left_hip_angles,
#                    right_hip_angles,
#                    left_elbow_angles,
#                    right_elbow_angles,
#                    left_knee_angles,
#                    right_knee_angles,
#                    frame=FRAME)
