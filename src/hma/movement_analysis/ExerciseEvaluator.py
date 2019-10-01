from .Exercise import Exercise
from .AngleTargetStates import AngleTargetStates
from .AngleTypes import AngleTypes
from .Sequence import Sequence
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, savgol_filter


class ExerciseEvaluator:

    def __init__(self, exercise: Exercise):
        # The Exercise to evaluate
        self.exercise = exercise
        # The target_angles for each body part
        self.target_angles = None
        # The prioritised body parts and angles: [(<body_part_index>, <AngleType.KEY>)]
        self.prio_angles = None

        self.iterations = np.array([])
        self.global_minima = []
        self.global_maxima = []
        self.global_sequence = []
        self.global_prio_angles = []

    def _get_prio_angles(self, ex: Exercise, seq: Sequence) -> list:
        """
        Returns a list of tuples containing a body part mapped in Sequence.body_parts and the AngleType for that body part which is prioritised.
        Example: [(4, AngleType.FLEX_EX), (4, AngleType.AB_AD)]
        """
        HIGH_PRIO = 1.0
        prio_angles = []
        # Shoulders
        if ex.angles[AngleTargetStates.START.value]["shoulder_left"]["flexion_extension"]["priority"] == HIGH_PRIO:
            prio_angles.append((seq.body_parts["LeftShoulder"], AngleTypes.FLEX_EX))
        if ex.angles[AngleTargetStates.START.value]["shoulder_left"]["abduction_adduction"]["priority"] == HIGH_PRIO:
            prio_angles.append((seq.body_parts["LeftShoulder"], AngleTypes.AB_AD))
        if ex.angles[AngleTargetStates.START.value]["shoulder_right"]["flexion_extension"]["priority"] == HIGH_PRIO:
            prio_angles.append((seq.body_parts["RightShoulder"], AngleTypes.FLEX_EX))
        if ex.angles[AngleTargetStates.START.value]["shoulder_right"]["abduction_adduction"]["priority"] == HIGH_PRIO:
            prio_angles.append((seq.body_parts["RightShoulder"], AngleTypes.AB_AD))
        # Hips
        if ex.angles[AngleTargetStates.START.value]["hip_left"]["flexion_extension"]["priority"] == HIGH_PRIO:
            prio_angles.append((seq.body_parts["LeftHip"], AngleTypes.FLEX_EX))
        if ex.angles[AngleTargetStates.START.value]["hip_left"]["abduction_adduction"]["priority"] == HIGH_PRIO:
            prio_angles.append((seq.body_parts["LeftHip"], AngleTypes.AB_AD))
        if ex.angles[AngleTargetStates.START.value]["hip_right"]["flexion_extension"]["priority"] == HIGH_PRIO:
            prio_angles.append((seq.body_parts["RightHip"], AngleTypes.FLEX_EX))
        if ex.angles[AngleTargetStates.START.value]["hip_right"]["abduction_adduction"]["priority"] == HIGH_PRIO:
            prio_angles.append((seq.body_parts["RightHip"], AngleTypes.AB_AD))
        # Elbows
        if ex.angles[AngleTargetStates.START.value]["elbow_left"]["flexion_extension"]["priority"] == HIGH_PRIO:
            prio_angles.append((seq.body_parts["LeftElbow"], AngleTypes.FLEX_EX))
        if ex.angles[AngleTargetStates.START.value]["elbow_right"]["flexion_extension"]["priority"] == HIGH_PRIO:
            prio_angles.append((seq.body_parts["RightElbow"], AngleTypes.FLEX_EX))
        # Knees
        if ex.angles[AngleTargetStates.START.value]["knee_left"]["flexion_extension"]["priority"] == HIGH_PRIO:
            prio_angles.append((seq.body_parts["LeftKnee"], AngleTypes.FLEX_EX))
        if ex.angles[AngleTargetStates.START.value]["knee_right"]["flexion_extension"]["priority"] == HIGH_PRIO:
            prio_angles.append((seq.body_parts["RightKnee"], AngleTypes.FLEX_EX))

        self.prio_angles = prio_angles
        return prio_angles

    def _get_target_angles(self, ex: Exercise, seq: Sequence) -> list:
        """
        Returns a 4-D ndarray which contain the Exercises' range of target angles for all body_parts, angle types and target states as minimum/maximum . 
        The position of each body part in the returned list is mapped in the body_part attribute of the given sequence (seq.body_parts).
        The positions of the second dimension represents an angle type of the AngleTypes enum.
        The position in the third dimension represents the START (0) and END (1) state.
        The position in the fourth dimension represent the minimum value (0) and maximum value (1)
        """
        target_angles = np.zeros((len(seq.body_parts), len(AngleTypes), len(AngleTargetStates), 2))

        # Shoulders
        target_angles[seq.body_parts["LeftShoulder"]][AngleTypes.FLEX_EX.value][0] = ex.angles[AngleTargetStates.START.value]["shoulder_left"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["LeftShoulder"]][AngleTypes.FLEX_EX.value][1] = ex.angles[AngleTargetStates.END.value]["shoulder_left"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["LeftShoulder"]][AngleTypes.AB_AD.value][0] = ex.angles[AngleTargetStates.START.value]["shoulder_left"]["abduction_adduction"]["angle"]
        target_angles[seq.body_parts["LeftShoulder"]][AngleTypes.AB_AD.value][1] = ex.angles[AngleTargetStates.END.value]["shoulder_left"]["abduction_adduction"]["angle"]
        target_angles[seq.body_parts["RightShoulder"]][AngleTypes.FLEX_EX.value][0] = ex.angles[AngleTargetStates.START.value]["shoulder_right"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["RightShoulder"]][AngleTypes.FLEX_EX.value][1] = ex.angles[AngleTargetStates.END.value]["shoulder_right"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["RightShoulder"]][AngleTypes.AB_AD.value][0] = ex.angles[AngleTargetStates.START.value]["shoulder_right"]["abduction_adduction"]["angle"]
        target_angles[seq.body_parts["RightShoulder"]][AngleTypes.AB_AD.value][1] = ex.angles[AngleTargetStates.END.value]["shoulder_right"]["abduction_adduction"]["angle"]
        # Hips
        target_angles[seq.body_parts["LeftHip"]][AngleTypes.FLEX_EX.value][0] = ex.angles[AngleTargetStates.START.value]["hip_left"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["LeftHip"]][AngleTypes.FLEX_EX.value][1] = ex.angles[AngleTargetStates.END.value]["hip_left"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["LeftHip"]][AngleTypes.AB_AD.value][0] = ex.angles[AngleTargetStates.START.value]["hip_left"]["abduction_adduction"]["angle"]
        target_angles[seq.body_parts["LeftHip"]][AngleTypes.AB_AD.value][1] = ex.angles[AngleTargetStates.END.value]["hip_left"]["abduction_adduction"]["angle"]
        target_angles[seq.body_parts["RightHip"]][AngleTypes.FLEX_EX.value][0] = ex.angles[AngleTargetStates.START.value]["hip_right"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["RightHip"]][AngleTypes.FLEX_EX.value][1] = ex.angles[AngleTargetStates.END.value]["hip_right"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["RightHip"]][AngleTypes.AB_AD.value][0] = ex.angles[AngleTargetStates.START.value]["hip_right"]["abduction_adduction"]["angle"]
        target_angles[seq.body_parts["RightHip"]][AngleTypes.AB_AD.value][1] = ex.angles[AngleTargetStates.END.value]["hip_right"]["abduction_adduction"]["angle"]
        # Elbow
        target_angles[seq.body_parts["LeftElbow"]][AngleTypes.FLEX_EX.value][0] = ex.angles[AngleTargetStates.START.value]["hip_left"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["LeftElbow"]][AngleTypes.FLEX_EX.value][1] = ex.angles[AngleTargetStates.END.value]["hip_left"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["RightElbow"]][AngleTypes.FLEX_EX.value][0] = ex.angles[AngleTargetStates.START.value]["hip_left"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["RightElbow"]][AngleTypes.FLEX_EX.value][1] = ex.angles[AngleTargetStates.END.value]["hip_left"]["flexion_extension"]["angle"]
        # Knee
        target_angles[seq.body_parts["LeftKnee"]][AngleTypes.FLEX_EX.value][0] = ex.angles[AngleTargetStates.START.value]["hip_left"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["LeftKnee"]][AngleTypes.FLEX_EX.value][1] = ex.angles[AngleTargetStates.END.value]["hip_left"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["RightKnee"]][AngleTypes.FLEX_EX.value][0] = ex.angles[AngleTargetStates.START.value]["hip_left"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["RightKnee"]][AngleTypes.FLEX_EX.value][1] = ex.angles[AngleTargetStates.END.value]["hip_left"]["flexion_extension"]["angle"]

        self.target_angles = target_angles
        return target_angles

    def find_iteration_keypoints(self, sequence: Sequence):
        prio_angles = self.get_prio_angles(self.exercise, sequence)

        if len(self.global_minima) == 0:
            for i in range(0, len(prio_angles)):
                self.global_maxima.append([])
                self.global_minima.append([])

        for prio_joint_idx, prio_joint in enumerate(prio_angles):
            joint_angles = prio_joint[0]
            ex_target_start = prio_joint[1]
            ex_target_end = prio_joint[2]

            # TODO: Find best value for window size
            savgol_window_max = 51
            savgol_window_generic = int(math.floor(len(joint_angles)/1.5)+1 if math.floor(len(joint_angles)/1.5) % 2 == 0 else math.floor(len(joint_angles)/1.5))
            # savgol_window = min(savgol_window_max, savgol_window_generic)
            savgol_window = 51
            joint_angles_smooth = savgol_filter(joint_angles, savgol_window, 3, mode="nearest")

            # TODO: Find best value for order parameter
            maxima = argrelextrema(joint_angles_smooth, np.greater, order=10)[0]
            minima = argrelextrema(joint_angles_smooth, np.less, order=10)[0]

            ### Filter Extrema ###
            # 1. Check whether local maxima/minima are closer to their desired target point than to the other target point
            # target_end_greater_start -> Maxima = END state frames; Minima = START state frames
            # target_end_less_start -> Minima = END state frames; Maxima = START state frames
            target_end_greater_start = ex_target_end > ex_target_start
            target_end_less_start = ex_target_end < ex_target_start
            # target_end_is_zero = ex_target_end == ex_target_start
            # TODO: Solve with map/lambda ?
            for rel_max in maxima:
                diff_to_start = abs(joint_angles_smooth[rel_max] - min(ex_target_start))
                diff_to_end = abs(joint_angles_smooth[rel_max] - min(ex_target_end))
                if target_end_greater_start:
                    if diff_to_start > diff_to_end:
                        self.global_maxima[prio_joint_idx].append(rel_max + len(self.global_sequence))
                if target_end_less_start:
                    if diff_to_start < diff_to_end:
                        self.global_maxima[prio_joint_idx].append(rel_max + len(self.global_sequence))
            for rel_min in minima:
                diff_to_start = abs(joint_angles_smooth[rel_min] - min(ex_target_start))
                diff_to_end = abs(joint_angles_smooth[rel_min] - min(ex_target_end))
                if target_end_greater_start:
                    if diff_to_start < diff_to_end:
                        self.global_minima[prio_joint_idx].append(rel_min + len(self.global_sequence))
                if target_end_less_start:
                    if diff_to_start > diff_to_end:
                        self.global_minima[prio_joint_idx].append(rel_min + len(self.global_sequence))

            # Compare indices of min/max. The altering list must contain increasing index values. One sequence of min < max < min is one iteration.
            if target_end_greater_start:
                minmax_altering = []
                for i in range(0, max(len(self.global_minima[prio_joint_idx]), len(self.global_maxima[prio_joint_idx]))):
                    if len(self.global_minima[prio_joint_idx]) > i:
                        minmax_altering.append((self.global_minima[prio_joint_idx][i], "min"))
                    if len(self.global_maxima[prio_joint_idx]) > i:
                        minmax_altering.append((self.global_maxima[prio_joint_idx][i], "max"))
            if target_end_less_start:
                minmax_altering = np.array([])
                for i in range(0, max(len(self.global_minima[prio_joint_idx]), len(self.global_maxima[prio_joint_idx]))):
                    if len(self.global_maxima[prio_joint_idx]) > i:
                        minmax_altering.append((self.global_maxima[prio_joint_idx][i], "max"))
                    if len(self.global_minima[prio_joint_idx]) > i:
                        minmax_altering.append((self.global_minima[prio_joint_idx][i], "min"))

            # print(minmax_altering)
            print(minmax_altering[0:3])

        if len(self.global_sequence) == 0:
            self.global_sequence = sequence
            self.global_prio_angles = prio_angles
        else:
            self.global_sequence = self.global_sequence.merge(sequence)
            for i in range(0, len(prio_angles)):
                np.append(self.global_prio_angles[i], prio_angles[i])

        # plt.plot(range(0, len(joint_angles)), joint_angles, zorder=1)
        # plt.plot(range(0, len(joint_angles)), joint_angles_smooth, color='red', zorder=1)
        # plt.scatter(maxima, joint_angles_smooth[maxima], color='green', marker="^", zorder=2)
        # plt.scatter(minima, joint_angles_smooth[minima], color='green', marker="v", zorder=2)
        # plt.show()

        # plt.plot(range(0, len(self.global_prio_angles[prio_joint_idx][0])), self.global_prio_angles[prio_joint_idx][0], zorder=1)
        # if len(self.global_maxima[prio_joint_idx]) > 0:
        #     plt.scatter(np.array(self.global_maxima[prio_joint_idx]), np.array(self.global_prio_angles[prio_joint_idx][0])[np.array(self.global_maxima[prio_joint_idx])], color='green', marker="^", zorder=2)
        # if len(self.global_minima[prio_joint_idx]) > 0:
        #     plt.scatter(np.array(self.global_minima[prio_joint_idx]), np.array(self.global_prio_angles[prio_joint_idx][0])[np.array(self.global_minima[prio_joint_idx])], color='green', marker="v", zorder=2)
        # plt.show()

    def evaluate(self, seq: Sequence, switch_state_idx: int):
        ex = self.exercise
        bp = seq.body_parts

        if self.prio_angles == None:
            self._get_prio_angles(ex, seq)

        if self.target_angles == None:
            self._get_target_angles(ex, seq)

        # results = np.zeros((len(seq), len(seq.body_parts), len(AngleTypes)))
        results = [[[None] * len(AngleTypes)] * len(bp)] * len(seq)
        current_target_state = AngleTargetStates.END
        for frame in range(0, len(seq)):
            if frame == switch_state_idx:
                current_target_state = AngleTargetStates.START

            # Shoulders
            shoulder_left_angle_flex_ex, shoulder_left_angle_abd_add = self.process_ball_joint_angles(
                seq.joint_angles[frame][bp["LeftShoulder"]][AngleTypes.FLEX_EX.value],
                seq.joint_angles[frame][bp["LeftShoulder"]][AngleTypes.AB_AD.value],
                self.target_angles[bp["LeftShoulder"]][AngleTypes.FLEX_EX.value],
                self.target_angles[bp["LeftShoulder"]][AngleTypes.AB_AD.value])
            results[frame][bp["LeftShoulder"]] = ex.check_angles_shoulder_left(shoulder_left_angle_flex_ex, shoulder_left_angle_abd_add, current_target_state, 10)
            # shoulder_right_angle_flex_ex, shoulder_right_angle_abd_add = self.process_ball_joint_angles(
            #     seq.joint_angles[frame][bp["RightShoulder"]][AngleTypes.FLEX_EX.value],
            #     seq.joint_angles[frame][bp["RightShoulder"]][AngleTypes.AB_AD.value],
            #     self.target_angles[bp["RightShoulder"]][AngleTypes.FLEX_EX.value],
            #     self.target_angles[bp["RightShoulder"]][AngleTypes.AB_AD.value])
            # shoulder_right_results.append(ex.check_angles_shoulder_right(shoulder_right_angle_flex_ex, shoulder_right_angle_abd_add, current_target_state, 10))
            # # Hips
            # hip_left_angle_flex_ex, hip_left_angle_abd_add = self.process_ball_joint_angles(
            #     seq.joint_angles[frame][bp["LeftHip"]][AngleTypes.FLEX_EX.value],
            #     seq.joint_angles[frame][bp["LeftHip"]][AngleTypes.AB_AD.value],
            #     self.target_angles[bp["LeftHip"]][AngleTypes.FLEX_EX.value],
            #     self.target_angles[bp["LeftHip"]][AngleTypes.AB_AD.value])
            # hip_left_results.append(ex.check_angles_shoulder_left(hip_left_angle_flex_ex, hip_left_angle_abd_add, current_target_state, 10))
            # hip_right_angle_flex_ex, hip_right_angle_abd_add = self.process_ball_joint_angles(
            #     seq.joint_angles[frame][bp["RightHip"]][AngleTypes.FLEX_EX.value],
            #     seq.joint_angles[frame][bp["RightHip"]][AngleTypes.AB_AD.value],
            #     self.target_angles[bp["RightHip"]][AngleTypes.FLEX_EX.value],
            #     self.target_angles[bp["RightHip"]][AngleTypes.AB_AD.value])
            # hip_right_results.append(ex.check_angles_shoulder_right(hip_right_angle_flex_ex, hip_right_angle_abd_add, current_target_state, 10))
            # # Elbows
            # elbow_left_angle_flex_ex = seq.joint_angles[frame][bp["LeftElbow"]][AngleTypes.FLEX_EX.value]
            # elbow_left_results.append(ex.check_angles_elbow_left(elbow_left_angle_flex_ex, current_target_state, 10))
            # elbow_right_angle_flex_ex = seq.joint_angles[frame][bp["RightElbow"]][AngleTypes.AB_AD.value]
            # elbow_right_results.append(ex.check_angles_elbow_right(elbow_right_angle_flex_ex, current_target_state, 10))
            # # Knees
            # knee_left_angle_flex_ex = seq.joint_angles[frame][bp["LeftKnee"]][AngleTypes.FLEX_EX.value]
            # knee_left_results.append(ex.check_angles_knee_left(knee_left_angle_flex_ex, current_target_state, 10))
            # knee_right_angle_flex_ex = seq.joint_angles[frame][bp["RightKnee"]][AngleTypes.AB_AD.value]
            # knee_right_results.append(ex.check_angles_knee_right(knee_right_angle_flex_ex, current_target_state, 10))

            # results = [None] * len(seq.body_parts)
            # results[bp["LeftShoulder"]] = shoulder_left_results
            # results[bp["RightShoulder"]] = shoulder_right_results
            # results[bp["LeftHip"]] = hip_left_results
            # results[bp["RightHip"]] = hip_right_results
            # results[bp["LeftElbow"]] = elbow_left_results
            # results[bp["RightElbow"]] = elbow_right_results
            # results[bp["LeftKnee"]] = knee_left_results
            # results[bp["RightKnee"]] = knee_right_results
            return results

    def process_ball_joint_angles(
            self,
            angle_flex_ex: float,
            angle_abd_add: float,
            target_angle_range_flex_ex: list,
            target_angle_range_abd_add: list,
            ignore_flex_abd90_delta: int = 20,
            abd_add_motion_thresh: int = 45) -> tuple:
        # TODO: This function needs a review, whether the processing is correct.
        # => Always altering abduction/adduction angles might be incorrect because in case of a flexion->abduction rotation order,
        #    the abduction represents the horizontal abduction, which is actually limited to [-90, 90°]. In this case, adding 90° would be wrong.
        #    Possible solution: Add 90° only if the prioritised angle is an abduction/adduction.

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
