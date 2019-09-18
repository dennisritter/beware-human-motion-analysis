from hma.movement_analysis.Exercise import Exercise
from hma.movement_analysis.PoseFormatEnum import PoseFormatEnum
from hma.movement_analysis.AngleTargetStates import AngleTargetStates
from hma.movement_analysis.Sequence import Sequence
from hma.movement_analysis.PoseProcessor import PoseProcessor
from hma.movement_analysis import exercise_loader
from hma.movement_analysis import angle_calculations_medical as acm
from hma.movement_analysis import transformations
from hma.movement_analysis import logging
import math
import matplotlib.pyplot as plt
import numpy as np


def process_ball_joint_angles(
        angle_flex_ex: float,
        angle_abd_add: float,
        target_angle_range_flex_ex: list,
        target_angle_range_abd_add: list,
        ignore_flex_abd90_delta: int = 20,
        abd_add_motion_thresh: int = 45) -> tuple:

    # Check if angle-vector.y is higher than origin and adjust abduction/adduction angles if conditions are met
    # If flexion angle is >90.0, angle-vector.y is higher than origin because flexion angle represents a rotation about the X-Axis
    if angle_flex_ex > 90.0:
        # If motion is considered as Abduction, add 90 degrees to the current angle to meet the expected abduction range [0,180] and not only [0,90]
        if angle_abd_add > abd_add_motion_thresh:
            angle_abd_add += 90
        # If motion is considered as Adduction, sub 90 degrees to the current angle to meet the expected abduction range [0,-180] and not only [0,-90]
        if angle_abd_add < -abd_add_motion_thresh:
            angle_abd_add -= 90

    # Set Flexion/Extension to 0.0° when angle-vector is close to X-Axis.
    # -> Flexion/Extension angles get very sensitive and error prone when close to X-Axis because it represents a rotation around it.
    full_absolute_abd_add = 90.0
    if abs(full_absolute_abd_add - abs(angle_abd_add)) < ignore_flex_abd90_delta:
        angle_flex_ex = 0.0

    return angle_flex_ex, angle_abd_add


# Get Exercise Object from json file
ex = exercise_loader.load('data/exercises/kniebeuge.json')
# Get PoseProcessor instance for MOCAP sequences
mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)
# Convert mocap json string Positions to Sequence Object
seq = mocap_poseprocessor.load('data/sequences/squat_3/complete-session.json', 'Squat')
bp = seq.body_parts

left_shoulder_angles = acm.calc_angles_shoulder_left(seq, bp["LeftShoulder"], bp["RightShoulder"], bp["Torso"], bp["LeftElbow"])
right_shoulder_angles = acm.calc_angles_shoulder_right(seq, bp["RightShoulder"], bp["LeftShoulder"], bp["Torso"], bp["RightElbow"])
left_hip_angles = acm.calc_angles_hip_left(seq, bp["LeftHip"], bp["RightHip"], bp["Torso"], bp["LeftKnee"])
right_hip_angles = acm.calc_angles_hip_right(seq, bp["RightHip"], bp["LeftHip"], bp["Torso"], bp["RightKnee"])
left_elbow_angles = acm.calc_angles_elbow(seq, bp["LeftElbow"], bp["LeftShoulder"], bp["LeftWrist"])
right_elbow_angles = acm.calc_angles_elbow(seq, bp["RightElbow"], bp["RightShoulder"], bp["RightWrist"])
left_knee_angles = acm.calc_angles_knee(seq, bp["LeftKnee"], bp["LeftHip"], bp["LeftAnkle"])
right_knee_angles = acm.calc_angles_knee(seq, bp["RightKnee"], bp["RightHip"], bp["RightAnkle"])
# print(left_shoulder_angles)

# ANGLE ANALYSIS
shoulder_left_results = []
shoulder_right_results = []
hip_left_results = []
hip_right_results = []
elbow_left_results = []
elbow_right_results = []
knee_left_results = []
knee_right_results = []

current_target_state = AngleTargetStates.END
for frame in range(0, len(seq.positions)):
    shoulder_left_angle_flex_ex, shoulder_left_angle_abd_add = process_ball_joint_angles(
        left_shoulder_angles["flexion_extension"][frame],
        left_shoulder_angles["abduction_adduction"][frame],
        ex.angles[current_target_state.value]["shoulder_left"]["flexion_extension"]["angle"],
        ex.angles[current_target_state.value]["shoulder_left"]["abduction_adduction"]["angle"])
    shoulder_left_results.append(ex.check_angles_shoulder_left(shoulder_left_angle_flex_ex, shoulder_left_angle_abd_add, current_target_state, 10))
    # print(f"Shoulder Left Flexion: {ex.check_angles_shoulder_left(shoulder_left_angle_flex_ex, shoulder_left_angle_abd_add, current_target_state, 10)['flexion_extension']}")
    # print(f"Shoulder Left Abduction: {ex.check_angles_shoulder_left(shoulder_left_angle_flex_ex, shoulder_left_angle_abd_add, current_target_state, 10)['abduction_adduction']}")
    shoulder_right_angle_flex_ex, shoulder_right_angle_abd_add = process_ball_joint_angles(
        right_shoulder_angles["flexion_extension"][frame],
        right_shoulder_angles["abduction_adduction"][frame],
        ex.angles[current_target_state.value]["shoulder_right"]["flexion_extension"]["angle"],
        ex.angles[current_target_state.value]["shoulder_right"]["abduction_adduction"]["angle"])
    shoulder_right_results.append(ex.check_angles_shoulder_right(shoulder_right_angle_flex_ex, shoulder_right_angle_abd_add, current_target_state, 10))
    # print(f"Shoulder Right Flexion: {ex.check_angles_shoulder_right(shoulder_right_angle_flex_ex, shoulder_right_angle_abd_add, current_target_state, 10)['flexion_extension']}")
    # print(f"Shoulder Right Abduction: {ex.check_angles_shoulder_right(shoulder_right_angle_flex_ex, shoulder_right_angle_abd_add, current_target_state, 10)['abduction_adduction']}")
    hip_left_angle_flex_ex, hip_left_angle_abd_add = process_ball_joint_angles(
        left_hip_angles["flexion_extension"][frame],
        left_hip_angles["abduction_adduction"][frame],
        ex.angles[current_target_state.value]["hip_left"]["flexion_extension"]["angle"],
        ex.angles[current_target_state.value]["hip_left"]["abduction_adduction"]["angle"])
    hip_left_results.append(ex.check_angles_shoulder_left(hip_left_angle_flex_ex, hip_left_angle_abd_add, current_target_state, 10))
    # print(f"Hip Left Flexion: {ex.check_angles_hip_left(hip_left_angle_flex_ex, hip_left_angle_abd_add, current_target_state, 10)['flexion_extension']}")
    # print(f"Hip Left Abduction: {ex.check_angles_hip_left(hip_left_angle_flex_ex, hip_left_angle_abd_add, current_target_state, 10)['abduction_adduction']}")
    hip_right_angle_flex_ex, hip_right_angle_abd_add = process_ball_joint_angles(
        right_hip_angles["flexion_extension"][frame],
        right_hip_angles["abduction_adduction"][frame],
        ex.angles[current_target_state.value]["hip_right"]["flexion_extension"]["angle"],
        ex.angles[current_target_state.value]["hip_right"]["abduction_adduction"]["angle"])
    hip_right_results.append(ex.check_angles_shoulder_right(hip_right_angle_flex_ex, hip_right_angle_abd_add, current_target_state, 10))
    # print(f"Hip Right Flexion: {ex.check_angles_hip_right(hip_right_angle_flex_ex, hip_right_angle_abd_add, current_target_state, 10)['flexion_extension']}")
    # print(f"Hip Right Abduction: {ex.check_angles_hip_right(hip_right_angle_flex_ex, hip_right_angle_abd_add, current_target_state, 10)['abduction_adduction']}")

    elbow_left_angle_flex_ex = left_elbow_angles["flexion_extension"][frame]
    elbow_left_results.append(ex.check_angles_elbow_left(elbow_left_angle_flex_ex, current_target_state, 10))
    # print(f"Elbow Left Flexion: {ex.check_angles_elbow_left(elbow_left_angle_flex_ex, current_target_state, 10)['flexion_extension']}")
    elbow_right_angle_flex_ex = right_elbow_angles["flexion_extension"][frame]
    elbow_right_results.append(ex.check_angles_elbow_right(elbow_right_angle_flex_ex, current_target_state, 10))
    # print(f"Elbow Right Flexion: {ex.check_angles_elbow_right(elbow_right_angle_flex_ex, current_target_state, 10)['flexion_extension']}")
    knee_left_angle_flex_ex = left_knee_angles["flexion_extension"][frame]
    knee_left_results.append(ex.check_angles_knee_left(knee_left_angle_flex_ex, current_target_state, 10))
    # print(f"Knee Left Flexion: {ex.check_angles_knee_left(knee_left_angle_flex_ex, current_target_state, 10)['flexion_extension']}")
    knee_right_angle_flex_ex = right_knee_angles["flexion_extension"][frame]
    knee_right_results.append(ex.check_angles_knee_right(knee_right_angle_flex_ex, current_target_state, 10))
    # print(f"Knee Right Flexion: {ex.check_angles_knee_right(knee_right_angle_flex_ex, current_target_state, 10)['flexion_extension']}")

# logging.log_angles(left_shoulder_angles,
#                    right_shoulder_angles,
#                    left_hip_angles,
#                    right_hip_angles,
#                    left_elbow_angles,
#                    right_elbow_angles,
#                    left_knee_angles,
#                    right_knee_angles,
#                    frame=FRAME)
