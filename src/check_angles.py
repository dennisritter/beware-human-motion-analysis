from Exercise import Exercise
from PoseFormatEnum import PoseFormatEnum
import exercise_loader
from Sequence import Sequence
from PoseMapper import PoseMapper
import angle_calculations_medical as acm
import numpy as np
import transformations
import math
import matplotlib.pyplot as plt

MOCAP_BODY_PARTS = {
    "LeftWrist": 0,
    "LeftElbow": 1,
    "LeftShoulder": 2,
    "Neck": 3,
    "Torso": 4,
    "Waist": 5,
    "LeftAnkle": 6,
    "LeftKnee": 7,
    "LeftHip": 8,
    "RightAnkle": 9,
    "RightKnee": 10,
    "RightHip": 11,
    "RightWrist": 12,
    "RightElbow": 13,
    "RightShoulder": 14,
    "Head": 15
}
FRAME = 0
# Get Exercise Object from json file
ex = exercise_loader.load('data/exercises/squat.json')
# Get PoseMapper instance for MOCAP sequences
mocap_posemapper = PoseMapper(PoseFormatEnum.MOCAP)
# Convert mocap json string Positions to Sequence Object
seq = mocap_posemapper.load('data/sequences/squat_3/complete-session.json', 'Squat')

# NOTE:
# flexion_extension and abduction_adduction seem correct
# inner_outer_Rotation getting very high sometimes they shouldn't -> SEW Normal flips 180° sometimes
# [54-64] data/sequences/squat_3/complete-session.json -> false rotation angles
#
# left_shoulder_angles = acm.calc_angles_shoulder_left(seq, 2, 14, 3, 1, 0, log=True)
# NOTE:
# flexion_extension false angles:
# [38, 48] data/sequences/squat_3/complete-session.json
# abduction_adduction false angles:
# [38, 48, 74] data/sequences/squat_3/complete-session.json
# inner_outer_Rotation getting very high sometimes they shouldn't -> SEW Normal flips 180° sometimes
# [54, 55, 58, 59, 106, 107] data/sequences/squat_3/complete-session.json
#
right_shoulder_angles = acm.calc_angles_shoulder_right(seq, 14, 2, 3, 13, 12, log=True)
# left_hip_angles = acm.calc_angles_shoulder_left(seq, 2, 14, 3, 1, 0)
# right_hip_angles = acm.calc_angles_shoulder_right(seq, 14, 2, 3, 13)

# left_elbow_angles = acm.calc_angle_elbow_flexion_extension(seq, 1, 2, 0)
# right_elbow_angles = acm.calc_angle_elbow_flexion_extension(seq, 13, 14, 12)
# print(f"Left Elbow angles: {left_elbow_angles['flexion_extension'][FRAME]}")
# print(f"Right Elbow angles: {right_elbow_angles['flexion_extension'][FRAME]}")
# left_knee_angles = acm.calc_angle_knee_flexion_extension(seq, 7, 8, 6)
# right_knee_angles = acm.calc_angle_knee_flexion_extension(seq, 10, 11, 9)
# print(f"Left Knee angles: {left_knee_angles['flexion_extension'][FRAME]}")
# print(f"Right Knee angles: {right_knee_angles['flexion_extension'][FRAME]}")

### Plotting ###
"""
fig = plt.figure(figsize=plt.figaspect(1)*2)
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(-1, 1)
for i, p in enumerate(seq.positions[FRAME]):
    if i == 1:
        ax.scatter(p[0], p[1], p[2], c="blue")
    else:
        ax.scatter(p[0], p[1], p[2], c="blue")
    ax.plot([seq.positions[FRAME][7][0], seq.positions[FRAME][8][0]],
            [seq.positions[FRAME][7][1], seq.positions[FRAME][8][1]],
            [seq.positions[FRAME][7][2], seq.positions[FRAME][8][2]],
            color="red", linewidth=1)
    ax.plot([seq.positions[FRAME][7][0], seq.positions[FRAME][6][0]],
            [seq.positions[FRAME][7][1], seq.positions[FRAME][6][1]],
            [seq.positions[FRAME][7][2], seq.positions[FRAME][6][2]],
            color="red", linewidth=1)
plt.show()
"""
