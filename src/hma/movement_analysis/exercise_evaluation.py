from .Exercise import Exercise
from .AngleTargetStates import AngleTargetStates
from .Sequence import Sequence
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, savgol_filter


def evaluate(exercise: Exercise, sequence: Sequence):
    ex = exercise
    seq = sequence
    bp = sequence.body_parts

    prio_angles = get_prio_angles(exercise, sequence)
    for prio_joint in prio_angles:
        joint_angles = prio_joint[0]
        ex_target_start = prio_joint[1]
        ex_target_end = prio_joint[2]

        # TODO: Find best value for window size
        savgol_window_max = 51
        savgol_window_generic = int(math.floor(len(joint_angles)/1.5)+1 if math.floor(len(joint_angles)/1.5) % 2 == 0 else math.floor(len(joint_angles)/1.5))
        savgol_window = min(savgol_window_max, savgol_window_generic)
        joint_angles_smooth = savgol_filter(joint_angles, savgol_window, 3, mode="nearest")

        # TODO: Find best value for order parameter
        maxima = argrelextrema(joint_angles_smooth, np.greater, order=10)[0]
        minima = argrelextrema(joint_angles_smooth, np.less, order=10)[0]

        plt.plot(range(0, len(joint_angles)), joint_angles, zorder=1)
        plt.plot(range(0, len(joint_angles)), joint_angles_smooth, color='red', zorder=1)
        plt.scatter(maxima, joint_angles_smooth[maxima], color='green', marker="^", zorder=2)
        plt.scatter(minima, joint_angles_smooth[minima], color='green', marker="v", zorder=2)
        plt.show()

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
                seq.joint_angles[bp["LeftShoulder"]]["flexion_extension"][frame],
                seq.joint_angles[bp["LeftShoulder"]]["abduction_adduction"][frame],
                ex.angles[current_target_state.value]["shoulder_left"]["flexion_extension"]["angle"],
                ex.angles[current_target_state.value]["shoulder_left"]["abduction_adduction"]["angle"])
            shoulder_left_results.append(ex.check_angles_shoulder_left(shoulder_left_angle_flex_ex, shoulder_left_angle_abd_add, current_target_state, 10))
            # print(f"Shoulder Left Flexion: {ex.check_angles_shoulder_left(shoulder_left_angle_flex_ex, shoulder_left_angle_abd_add, current_target_state, 10)['flexion_extension']}")
            # print(f"Shoulder Left Abduction: {ex.check_angles_shoulder_left(shoulder_left_angle_flex_ex, shoulder_left_angle_abd_add, current_target_state, 10)['abduction_adduction']}")
            shoulder_right_angle_flex_ex, shoulder_right_angle_abd_add = process_ball_joint_angles(
                seq.joint_angles[bp["RightShoulder"]]["flexion_extension"][frame],
                seq.joint_angles[bp["RightShoulder"]]["abduction_adduction"][frame],
                ex.angles[current_target_state.value]["shoulder_right"]["flexion_extension"]["angle"],
                ex.angles[current_target_state.value]["shoulder_right"]["abduction_adduction"]["angle"])
            shoulder_right_results.append(ex.check_angles_shoulder_right(shoulder_right_angle_flex_ex, shoulder_right_angle_abd_add, current_target_state, 10))
            # print(f"Shoulder Right Flexion: {ex.check_angles_shoulder_right(shoulder_right_angle_flex_ex, shoulder_right_angle_abd_add, current_target_state, 10)['flexion_extension']}")
            # print(f"Shoulder Right Abduction: {ex.check_angles_shoulder_right(shoulder_right_angle_flex_ex, shoulder_right_angle_abd_add, current_target_state, 10)['abduction_adduction']}")
            hip_left_angle_flex_ex, hip_left_angle_abd_add = process_ball_joint_angles(
                seq.joint_angles[bp["LeftHip"]]["flexion_extension"][frame],
                seq.joint_angles[bp["LeftHip"]]["abduction_adduction"][frame],
                ex.angles[current_target_state.value]["hip_left"]["flexion_extension"]["angle"],
                ex.angles[current_target_state.value]["hip_left"]["abduction_adduction"]["angle"])
            hip_left_results.append(ex.check_angles_shoulder_left(hip_left_angle_flex_ex, hip_left_angle_abd_add, current_target_state, 10))
            # print(f"Hip Left Flexion: {ex.check_angles_hip_left(hip_left_angle_flex_ex, hip_left_angle_abd_add, current_target_state, 10)['flexion_extension']}")
            # print(f"Hip Left Abduction: {ex.check_angles_hip_left(hip_left_angle_flex_ex, hip_left_angle_abd_add, current_target_state, 10)['abduction_adduction']}")
            hip_right_angle_flex_ex, hip_right_angle_abd_add = process_ball_joint_angles(
                seq.joint_angles[bp["RightHip"]]["flexion_extension"][frame],
                seq.joint_angles[bp["RightHip"]]["abduction_adduction"][frame],
                ex.angles[current_target_state.value]["hip_right"]["flexion_extension"]["angle"],
                ex.angles[current_target_state.value]["hip_right"]["abduction_adduction"]["angle"])
            hip_right_results.append(ex.check_angles_shoulder_right(hip_right_angle_flex_ex, hip_right_angle_abd_add, current_target_state, 10))
            # print(f"Hip Right Flexion: {ex.check_angles_hip_right(hip_right_angle_flex_ex, hip_right_angle_abd_add, current_target_state, 10)['flexion_extension']}")
            # print(f"Hip Right Abduction: {ex.check_angles_hip_right(hip_right_angle_flex_ex, hip_right_angle_abd_add, current_target_state, 10)['abduction_adduction']}")

            elbow_left_angle_flex_ex = seq.joint_angles[bp["LeftElbow"]]["flexion_extension"][frame]
            elbow_left_results.append(ex.check_angles_elbow_left(elbow_left_angle_flex_ex, current_target_state, 10))
            # print(f"Elbow Left Flexion: {ex.check_angles_elbow_left(elbow_left_angle_flex_ex, current_target_state, 10)['flexion_extension']}")
            elbow_right_angle_flex_ex = seq.joint_angles[bp["RightElbow"]]["flexion_extension"][frame]
            elbow_right_results.append(ex.check_angles_elbow_right(elbow_right_angle_flex_ex, current_target_state, 10))
            # print(f"Elbow Right Flexion: {ex.check_angles_elbow_right(elbow_right_angle_flex_ex, current_target_state, 10)['flexion_extension']}")
            knee_left_angle_flex_ex = seq.joint_angles[bp["LeftKnee"]]["flexion_extension"][frame]
            knee_left_results.append(ex.check_angles_knee_left(knee_left_angle_flex_ex, current_target_state, 10))
            # print(f"Knee Left Flexion: {ex.check_angles_knee_left(knee_left_angle_flex_ex, current_target_state, 10)['flexion_extension']}")
            knee_right_angle_flex_ex = seq.joint_angles[bp["RightKnee"]]["flexion_extension"][frame]
            knee_right_results.append(ex.check_angles_knee_right(knee_right_angle_flex_ex, current_target_state, 10))
            # print(f"Knee Right Flexion: {ex.check_angles_knee_right(knee_right_angle_flex_ex, current_target_state, 10)['flexion_extension']}")


def get_prio_angles(exercise: Exercise, sequence: Sequence):
    seq = sequence
    ex = exercise
    bp = seq.body_parts
    HIGH_PRIO = 1.0
    prio_angles = []
    # prio_angles -> [([START.min, START.max], [END.min, END.max], [frame0, frame1, frame2, ...]), (...), ...]
    if ex.angles[AngleTargetStates.START.value]["shoulder_left"]["flexion_extension"]["priority"] == HIGH_PRIO:
        prio_angles.append((
            seq.joint_angles[bp["LeftShoulder"]]["flexion_extension"],
            ex.angles[AngleTargetStates.START.value]["shoulder_left"]["flexion_extension"]["angle"],
            ex.angles[AngleTargetStates.END.value]["shoulder_left"]["flexion_extension"]["angle"],
        ))
    if ex.angles[AngleTargetStates.START.value]["shoulder_left"]["abduction_adduction"]["priority"] == HIGH_PRIO:
        prio_angles.append((
            seq.joint_angles[bp["LeftShoulder"]]["abduction_adduction"],
            ex.angles[AngleTargetStates.START.value]["shoulder_left"]["abduction_adduction"]["angle"],
            ex.angles[AngleTargetStates.END.value]["shoulder_left"]["abduction_adduction"]["angle"],
        ))
    if ex.angles[AngleTargetStates.START.value]["shoulder_right"]["flexion_extension"]["priority"] == HIGH_PRIO:
        prio_angles.append((
            seq.joint_angles[bp["RightShoulder"]]["flexion_extension"],
            ex.angles[AngleTargetStates.START.value]["shoulder_right"]["flexion_extension"]["angle"],
            ex.angles[AngleTargetStates.END.value]["shoulder_right"]["flexion_extension"]["angle"],
        ))
    if ex.angles[AngleTargetStates.START.value]["shoulder_right"]["abduction_adduction"]["priority"] == HIGH_PRIO:
        prio_angles.append((
            seq.joint_angles[bp["RightShoulder"]]["abduction_adduction"],
            ex.angles[AngleTargetStates.START.value]["shoulder_right"]["abduction_adduction"]["angle"],
            ex.angles[AngleTargetStates.END.value]["shoulder_right"]["abduction_adduction"]["angle"],
        ))

    if ex.angles[AngleTargetStates.START.value]["hip_left"]["flexion_extension"]["priority"] == HIGH_PRIO:
        prio_angles.append((
            seq.joint_angles[bp["LeftHip"]]["flexion_extension"],
            ex.angles[AngleTargetStates.START.value]["hip_left"]["flexion_extension"]["angle"],
            ex.angles[AngleTargetStates.END.value]["hip_left"]["flexion_extension"]["angle"],
        ))
    if ex.angles[AngleTargetStates.START.value]["hip_left"]["abduction_adduction"]["priority"] == HIGH_PRIO:
        prio_angles.append((
            seq.joint_angles[bp["LeftHip"]]["abduction_adduction"],
            ex.angles[AngleTargetStates.START.value]["hip_left"]["abduction_adduction"]["angle"],
            ex.angles[AngleTargetStates.END.value]["hip_left"]["abduction_adduction"]["angle"],
        ))
    if ex.angles[AngleTargetStates.START.value]["hip_right"]["flexion_extension"]["priority"] == HIGH_PRIO:
        prio_angles.append((
            seq.joint_angles[bp["RightHip"]]["flexion_extension"],
            ex.angles[AngleTargetStates.START.value]["hip_right"]["flexion_extension"]["angle"],
            ex.angles[AngleTargetStates.END.value]["hip_right"]["flexion_extension"]["angle"],
        ))
    if ex.angles[AngleTargetStates.START.value]["hip_right"]["abduction_adduction"]["priority"] == HIGH_PRIO:
        prio_angles.append((
            seq.joint_angles[bp["RightHip"]]["abduction_adduction"],
            ex.angles[AngleTargetStates.START.value]["hip_right"]["fleabduction_adductionxion_extension"]["angle"],
            ex.angles[AngleTargetStates.END.value]["hip_right"]["abduction_adduction"]["angle"],
        ))

    if ex.angles[AngleTargetStates.START.value]["elbow_left"]["flexion_extension"]["priority"] == HIGH_PRIO:
        prio_angles.append((
            seq.joint_angles[bp["LeftElbow"]]["flexion_extension"],
            ex.angles[AngleTargetStates.START.value]["elbow_left"]["flexion_extension"]["angle"],
            ex.angles[AngleTargetStates.END.value]["elbow_left"]["flexion_extension"]["angle"],
        ))
    if ex.angles[AngleTargetStates.START.value]["elbow_right"]["flexion_extension"]["priority"] == HIGH_PRIO:
        prio_angles.append((
            seq.joint_angles[bp["RightElbow"]]["flexion_extension"],
            ex.angles[AngleTargetStates.START.value]["elbow_right"]["flexion_extension"]["angle"],
            ex.angles[AngleTargetStates.END.value]["elbow_right"]["flexion_extension"]["angle"],
        ))
    if ex.angles[AngleTargetStates.START.value]["knee_left"]["flexion_extension"]["priority"] == HIGH_PRIO:
        prio_angles.append((
            seq.joint_angles[bp["LeftKnee"]]["flexion_extension"],
            ex.angles[AngleTargetStates.START.value]["knee_left"]["flexion_extension"]["angle"],
            ex.angles[AngleTargetStates.END.value]["knee_left"]["flexion_extension"]["angle"],
        ))
    if ex.angles[AngleTargetStates.START.value]["knee_right"]["flexion_extension"]["priority"] == HIGH_PRIO:
        prio_angles.append((
            seq.joint_angles[bp["RightKnee"]]["flexion_extension"],
            ex.angles[AngleTargetStates.START.value]["knee_right"]["flexion_extension"]["angle"],
            ex.angles[AngleTargetStates.END.value]["knee_right"]["flexion_extension"]["angle"],
        ))
    return prio_angles


# TODO: This function needs a review, whether the processing is correct.
# => Always altering abduction/adduction angles might be incorrect because in case of a flexion->abduction rotation order,
#    the abduction represents the horizontal abduction, which is actually limited to [-90, 90°]. In this case, adding 90° would be wrong.
#    Possible solution: Add 90° only if the prioritised angle is an abduction/adduction.
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
        # If motion is considered an Abduction, add 90 degrees to the current angle to meet the expected abduction range [0,180] and not only [0,90]
        if angle_abd_add > abd_add_motion_thresh:
            angle_abd_add += 90
        # If motion is considered an Adduction, sub 90 degrees to the current angle to meet the expected abduction range [0,-180] and not only [0,-90]
        if angle_abd_add < -abd_add_motion_thresh:
            angle_abd_add -= 90

    # Set Flexion/Extension to 0.0° when angle-vector is close to X-Axis.
    # -> Flexion/Extension angles get very sensitive and error prone when close to X-Axis because it represents a rotation around it.
    full_absolute_abd_add = 90.0
    if abs(full_absolute_abd_add - abs(angle_abd_add)) < ignore_flex_abd90_delta:
        angle_flex_ex = 0.0

    return angle_flex_ex, angle_abd_add
