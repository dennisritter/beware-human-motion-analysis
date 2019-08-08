from movement_analysis.Exercise import Exercise
from movement_analysis.PoseFormatEnum import PoseFormatEnum
from movement_analysis.Sequence import Sequence
from movement_analysis.PoseMapper import PoseMapper
from movement_analysis import exercise_loader
from movement_analysis import angle_calculations_medical as acm
from movement_analysis import transformations
from movement_analysis import logging
import math
import matplotlib.pyplot as plt
import numpy as np

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

left_shoulder_angles = acm.calc_angles_shoulder_left(seq, 2, 14, 3, 1, log=False)
right_shoulder_angles = acm.calc_angles_shoulder_right(seq, 14, 2, 3, 13, log=False)
left_hip_angles = acm.calc_angles_hip_left(seq, 8, 11, 4, 7, log=False)
right_hip_angles = acm.calc_angles_hip_right(seq, 11, 8, 4, 10, log=False)
left_elbow_angles = acm.calc_angles_elbow(seq, 1, 2, 0)
right_elbow_angles = acm.calc_angles_elbow(seq, 13, 14, 12)
left_knee_angles = acm.calc_angles_knee(seq, 7, 8, 6)
right_knee_angles = acm.calc_angles_knee(seq, 10, 11, 9)

logging.log_angles(left_shoulder_angles,
                   right_shoulder_angles,
                   left_hip_angles,
                   right_hip_angles,
                   left_elbow_angles,
                   right_elbow_angles,
                   left_knee_angles,
                   right_knee_angles,
                   frame=FRAME)
