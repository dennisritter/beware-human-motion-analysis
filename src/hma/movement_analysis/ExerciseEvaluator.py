from .Exercise import Exercise
from .AngleTargetStates import AngleTargetStates
from .AngleTypes import AngleTypes
from .AngleAnalysisResultStates import AngleAnalysisResultStates
from .Sequence import Sequence
import warnings
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, savgol_filter


class ExerciseEvaluator:

    HIGH_PRIO = 1.0
    MEDIUM_PRIO = 0.5
    LOW_PRIO = 0.0

    def __init__(self, exercise: Exercise):
        # The Exercise to evaluate
        self.exercise = exercise
        # The target_angles for each body part
        self.target_angles = None
        # The prioritised body parts and angles: [(<body_part_index>, <AngleType.KEY>)]
        self.prio_angles = None
        # A Dictionary that maps body part indices of a sequences positions to names body parts represented by String values (Sequence.body_parts attribute)
        self.body_part_indices = None

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
        if ex.angles["start"]["shoulder_left"]["flexion_extension"]["priority"] == HIGH_PRIO:
            prio_angles.append((seq.body_parts["LeftShoulder"], AngleTypes.FLEX_EX))
        if ex.angles["start"]["shoulder_left"]["abduction_adduction"]["priority"] == HIGH_PRIO:
            prio_angles.append((seq.body_parts["LeftShoulder"], AngleTypes.AB_AD))
        if ex.angles["start"]["shoulder_right"]["flexion_extension"]["priority"] == HIGH_PRIO:
            prio_angles.append((seq.body_parts["RightShoulder"], AngleTypes.FLEX_EX))
        if ex.angles["start"]["shoulder_right"]["abduction_adduction"]["priority"] == HIGH_PRIO:
            prio_angles.append((seq.body_parts["RightShoulder"], AngleTypes.AB_AD))
        # Hips
        if ex.angles["start"]["hip_left"]["flexion_extension"]["priority"] == HIGH_PRIO:
            prio_angles.append((seq.body_parts["LeftHip"], AngleTypes.FLEX_EX))
        if ex.angles["start"]["hip_left"]["abduction_adduction"]["priority"] == HIGH_PRIO:
            prio_angles.append((seq.body_parts["LeftHip"], AngleTypes.AB_AD))
        if ex.angles["start"]["hip_right"]["flexion_extension"]["priority"] == HIGH_PRIO:
            prio_angles.append((seq.body_parts["RightHip"], AngleTypes.FLEX_EX))
        if ex.angles["start"]["hip_right"]["abduction_adduction"]["priority"] == HIGH_PRIO:
            prio_angles.append((seq.body_parts["RightHip"], AngleTypes.AB_AD))
        # Elbows
        if ex.angles["start"]["elbow_left"]["flexion_extension"]["priority"] == HIGH_PRIO:
            prio_angles.append((seq.body_parts["LeftElbow"], AngleTypes.FLEX_EX))
        if ex.angles["start"]["elbow_right"]["flexion_extension"]["priority"] == HIGH_PRIO:
            prio_angles.append((seq.body_parts["RightElbow"], AngleTypes.FLEX_EX))
        # Knees
        if ex.angles["start"]["knee_left"]["flexion_extension"]["priority"] == HIGH_PRIO:
            prio_angles.append((seq.body_parts["LeftKnee"], AngleTypes.FLEX_EX))
        if ex.angles["start"]["knee_right"]["flexion_extension"]["priority"] == HIGH_PRIO:
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
        target_angles[seq.body_parts["LeftShoulder"]][AngleTypes.FLEX_EX.value][0] = ex.angles["start"]["shoulder_left"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["LeftShoulder"]][AngleTypes.FLEX_EX.value][1] = ex.angles["end"]["shoulder_left"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["LeftShoulder"]][AngleTypes.AB_AD.value][0] = ex.angles["start"]["shoulder_left"]["abduction_adduction"]["angle"]
        target_angles[seq.body_parts["LeftShoulder"]][AngleTypes.AB_AD.value][1] = ex.angles["end"]["shoulder_left"]["abduction_adduction"]["angle"]
        target_angles[seq.body_parts["RightShoulder"]][AngleTypes.FLEX_EX.value][0] = ex.angles["start"]["shoulder_right"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["RightShoulder"]][AngleTypes.FLEX_EX.value][1] = ex.angles["end"]["shoulder_right"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["RightShoulder"]][AngleTypes.AB_AD.value][0] = ex.angles["start"]["shoulder_right"]["abduction_adduction"]["angle"]
        target_angles[seq.body_parts["RightShoulder"]][AngleTypes.AB_AD.value][1] = ex.angles["end"]["shoulder_right"]["abduction_adduction"]["angle"]
        # Hips
        target_angles[seq.body_parts["LeftHip"]][AngleTypes.FLEX_EX.value][0] = ex.angles["start"]["hip_left"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["LeftHip"]][AngleTypes.FLEX_EX.value][1] = ex.angles["end"]["hip_left"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["LeftHip"]][AngleTypes.AB_AD.value][0] = ex.angles["start"]["hip_left"]["abduction_adduction"]["angle"]
        target_angles[seq.body_parts["LeftHip"]][AngleTypes.AB_AD.value][1] = ex.angles["end"]["hip_left"]["abduction_adduction"]["angle"]
        target_angles[seq.body_parts["RightHip"]][AngleTypes.FLEX_EX.value][0] = ex.angles["start"]["hip_right"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["RightHip"]][AngleTypes.FLEX_EX.value][1] = ex.angles["end"]["hip_right"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["RightHip"]][AngleTypes.AB_AD.value][0] = ex.angles["start"]["hip_right"]["abduction_adduction"]["angle"]
        target_angles[seq.body_parts["RightHip"]][AngleTypes.AB_AD.value][1] = ex.angles["end"]["hip_right"]["abduction_adduction"]["angle"]
        # Elbow
        target_angles[seq.body_parts["LeftElbow"]][AngleTypes.FLEX_EX.value][0] = ex.angles["start"]["hip_left"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["LeftElbow"]][AngleTypes.FLEX_EX.value][1] = ex.angles["end"]["hip_left"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["RightElbow"]][AngleTypes.FLEX_EX.value][0] = ex.angles["start"]["hip_left"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["RightElbow"]][AngleTypes.FLEX_EX.value][1] = ex.angles["end"]["hip_left"]["flexion_extension"]["angle"]
        # Knee
        target_angles[seq.body_parts["LeftKnee"]][AngleTypes.FLEX_EX.value][0] = ex.angles["start"]["hip_left"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["LeftKnee"]][AngleTypes.FLEX_EX.value][1] = ex.angles["end"]["hip_left"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["RightKnee"]][AngleTypes.FLEX_EX.value][0] = ex.angles["start"]["hip_left"]["flexion_extension"]["angle"]
        target_angles[seq.body_parts["RightKnee"]][AngleTypes.FLEX_EX.value][1] = ex.angles["end"]["hip_left"]["flexion_extension"]["angle"]

        self.target_angles = target_angles
        return target_angles

    def find_iteration_keypoints(self, seq: Sequence):
        ex = self.exercise

        if self.prio_angles == None:
            self._get_prio_angles(ex, seq)

        if self.target_angles == None:
            self._get_target_angles(ex, seq)

        if self.body_part_indices == None:
            self.body_part_indices = seq.body_parts

        # TODO: Refactor
        if len(self.global_minima) == 0:
            for i in range(0, len(self.prio_angles)):
                self.global_maxima.append([])
                self.global_minima.append([])

        for body_part_idx, angle_type in self.prio_angles:
            # (idx, AngleType)
            # (1, AngleType.FLEX_EX)
            
            # Get calculated angles of a specific type for a specific body part for all frames  
            angles = seq.joint_angles[:, body_part_idx, angle_type.value]

            # TODO: Find best value for window size
            # Apply a Savitzky-Golay Filter to get a list of 'smoothed' angles.
            # savgol_window_max = 51
            # savgol_window_generic = int(math.floor(len(angles)/1.5)+1 if math.floor(len(angles)/1.5) % 2 == 0 else math.floor(len(angles)/1.5))
            # savgol_window = min(savgol_window_max, savgol_window_generic)
            savgol_window = 51
            angles_savgol = savgol_filter(angles, savgol_window, 3, mode="nearest")

            # TODO: Find best value for order parameter
            # Find Minima and Maxima of angles after applying a Savitzky-Golay Filter filter
            maxima = argrelextrema(angles_savgol, np.greater, order=10)[0]
            minima = argrelextrema(angles_savgol, np.less, order=10)[0]

            # Get Exercise targets for the current angle type 
            ex_targets = self.target_angles[body_part_idx][angle_type.value]
            # Check if Exercise targets of target state END are greater than targets of START
            # We need this information to identify whether local MAXIMA or MINIMA represent start/end of a subsequence
            target_end_greater_start = min(ex_targets[AngleTargetStates.END.value]) > min(ex_targets[AngleTargetStates.START.value])

            # Check if distance to Exercise START target angle is greater than Exercise END target angle
            # in order to remove falsy maxima/minima
            _dist_filter = lambda x: abs(angles_savgol[x] - min(ex_targets[AngleTargetStates.START.value])) > abs(angles_savgol[x] - min(ex_targets[AngleTargetStates.END.value]))
            if target_end_greater_start:
                maxima = maxima[_dist_filter(maxima)]
                minima = minima[np.invert(_dist_filter(minima))]
            else:
                maxima = maxima[np.invert(_dist_filter(maxima))]
                minima = minima[_dist_filter(minima)]


            plt.plot(range(0, len(angles)), angles, zorder=1)
            plt.plot(range(0, len(angles)), angles_savgol, color='red', zorder=1)
            plt.scatter(maxima, angles_savgol[maxima], color='green', marker="^", zorder=2)
            plt.scatter(minima, angles_savgol[minima], color='green', marker="v", zorder=2)
            plt.show()

    def evaluate(self, seq: Sequence, switch_state_idx: int):
        ex = self.exercise
        bp = seq.body_parts

        if self.prio_angles == None:
            self._get_prio_angles(ex, seq)

        if self.target_angles == None:
            self._get_target_angles(ex, seq)

        if self.body_part_indices == None:
            self.body_part_indices = seq.body_parts

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
            results[frame][bp["LeftShoulder"]] = self._get_results_shoulder_left(shoulder_left_angle_flex_ex, shoulder_left_angle_abd_add, current_target_state, 10)
            shoulder_right_angle_flex_ex, shoulder_right_angle_abd_add = self.process_ball_joint_angles(
                seq.joint_angles[frame][bp["RightShoulder"]][AngleTypes.FLEX_EX.value],
                seq.joint_angles[frame][bp["RightShoulder"]][AngleTypes.AB_AD.value],
                self.target_angles[bp["RightShoulder"]][AngleTypes.FLEX_EX.value],
                self.target_angles[bp["RightShoulder"]][AngleTypes.AB_AD.value])
            results[frame][bp["RightShoulder"]] = self._get_results_shoulder_right(shoulder_right_angle_flex_ex, shoulder_right_angle_abd_add, current_target_state, 10)
            # # Hips
            hip_left_angle_flex_ex, hip_left_angle_abd_add = self.process_ball_joint_angles(
                seq.joint_angles[frame][bp["LeftHip"]][AngleTypes.FLEX_EX.value],
                seq.joint_angles[frame][bp["LeftHip"]][AngleTypes.AB_AD.value],
                self.target_angles[bp["LeftHip"]][AngleTypes.FLEX_EX.value],
                self.target_angles[bp["LeftHip"]][AngleTypes.AB_AD.value])
            results[frame][bp["LeftHip"]] = self._get_results_hip_left(hip_left_angle_flex_ex, hip_left_angle_abd_add, current_target_state, 10)
            hip_right_angle_flex_ex, hip_right_angle_abd_add = self.process_ball_joint_angles(
                seq.joint_angles[frame][bp["RightHip"]][AngleTypes.FLEX_EX.value],
                seq.joint_angles[frame][bp["RightHip"]][AngleTypes.AB_AD.value],
                self.target_angles[bp["RightHip"]][AngleTypes.FLEX_EX.value],
                self.target_angles[bp["RightHip"]][AngleTypes.AB_AD.value])
            results[frame][bp["RightHip"]] = self._get_results_hip_right(hip_right_angle_flex_ex, hip_right_angle_abd_add, current_target_state, 10)
            # # Elbows
            elbow_left_angle_flex_ex = seq.joint_angles[frame][bp["LeftElbow"]][AngleTypes.FLEX_EX.value]
            results[frame][bp["LeftElbow"]] = self._get_results_elbow_left(elbow_left_angle_flex_ex, current_target_state, 10)
            elbow_right_angle_flex_ex = seq.joint_angles[frame][bp["RightElbow"]][AngleTypes.FLEX_EX.value]
            results[frame][bp["RightElbow"]] = self._get_results_elbow_right(elbow_right_angle_flex_ex, current_target_state, 10)
            # # Knees
            knee_left_angle_flex_ex = seq.joint_angles[frame][bp["LeftKnee"]][AngleTypes.FLEX_EX.value]
            results[frame][bp["LeftKnee"]] = self._get_results_knee_left(knee_left_angle_flex_ex, current_target_state, 10)
            knee_right_angle_flex_ex = seq.joint_angles[frame][bp["RightKnee"]][AngleTypes.FLEX_EX.value]
            results[frame][bp["RightKnee"]] = self._get_results_knee_right(knee_right_angle_flex_ex, current_target_state, 10)

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

    def _get_results_shoulder_left(self, angle_flex_ex: float, angle_abd_add: float, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        result = [None] * len(AngleTypes)
        result[AngleTypes.FLEX_EX.value] = self._check_angle_shoulder_left(angle_flex_ex, AngleTypes.FLEX_EX, target_state, tolerance)
        result[AngleTypes.AB_AD.value] = self._check_angle_shoulder_left(angle_abd_add, AngleTypes.AB_AD, target_state, tolerance)
        return result

    def _get_results_shoulder_right(self, angle_flex_ex: float, angle_abd_add: float, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        result = [None] * len(AngleTypes)
        result[AngleTypes.FLEX_EX.value] = self._check_angle_shoulder_right(angle_flex_ex, AngleTypes.FLEX_EX, target_state, tolerance)
        result[AngleTypes.AB_AD.value] = self._check_angle_shoulder_right(angle_abd_add, AngleTypes.AB_AD, target_state, tolerance)
        return result

    def _get_results_hip_left(self, angle_flex_ex: float, angle_abd_add: float, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        result = [None] * len(AngleTypes)
        result[AngleTypes.FLEX_EX.value] = self._check_angle_hip_left(angle_flex_ex, AngleTypes.FLEX_EX, target_state, tolerance)
        result[AngleTypes.AB_AD.value] = self._check_angle_hip_left(angle_abd_add, AngleTypes.AB_AD, target_state, tolerance)
        return result

    def _get_results_hip_right(self, angle_flex_ex: float, angle_abd_add: float, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        result = [None] * len(AngleTypes)
        result[AngleTypes.FLEX_EX.value] = self._check_angle_hip_right(angle_flex_ex, AngleTypes.FLEX_EX, target_state, tolerance)
        result[AngleTypes.AB_AD.value] = self._check_angle_hip_right(angle_abd_add, AngleTypes.AB_AD, target_state, tolerance)
        return result

    def _get_results_elbow_left(self, angle_flex_ex: float, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        result = [None] * len(AngleTypes)
        result[AngleTypes.FLEX_EX.value] = self._check_angle_elbow_left(angle_flex_ex, AngleTypes.FLEX_EX, target_state, tolerance)
        return result

    def _get_results_elbow_right(self, angle_flex_ex: float, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        result = [None] * len(AngleTypes)
        result[AngleTypes.FLEX_EX.value] = self._check_angle_elbow_right(angle_flex_ex, AngleTypes.FLEX_EX, target_state, tolerance)
        return result

    def _get_results_knee_left(self, angle_flex_ex: float, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        result = [None] * len(AngleTypes)
        result[AngleTypes.FLEX_EX.value] = self._check_angle_knee_left(angle_flex_ex, AngleTypes.FLEX_EX, target_state, tolerance)
        return result

    def _get_results_knee_right(self, angle_flex_ex: float, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        result = [None] * len(AngleTypes)
        result[AngleTypes.FLEX_EX.value] = self._check_angle_knee_right(angle_flex_ex, AngleTypes.FLEX_EX, target_state, tolerance)
        return result

    def _check_angle_shoulder_left(self, angle: float, angle_type: AngleTypes, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        target_start = self.target_angles[self.body_part_indices["LeftShoulder"]][angle_type.value][AngleTargetStates.START.value]
        target_end = self.target_angles[self.body_part_indices["LeftShoulder"]][angle_type.value][AngleTargetStates.END.value]
        target_min = min(self.target_angles[self.body_part_indices["LeftShoulder"]][angle_type.value][target_state.value])
        target_max = max(self.target_angles[self.body_part_indices["LeftShoulder"]][angle_type.value][target_state.value])

        result = {
            "angle": angle,
            "body_part": "LeftShoulder",
            "angle_type": angle_type,
            "target_min": target_min,
            "target_max": target_max,
            "target_state": target_state,
            "result_state": self._get_angle_analysis_result_state(angle, target_state, target_start, target_end, target_min, target_max, tolerance),
        }
        return result

    def _check_angle_shoulder_right(self, angle: float, angle_type: AngleTypes, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        target_start = self.target_angles[self.body_part_indices["RightShoulder"]][angle_type.value][AngleTargetStates.START.value]
        target_end = self.target_angles[self.body_part_indices["RightShoulder"]][angle_type.value][AngleTargetStates.END.value]
        target_min = min(self.target_angles[self.body_part_indices["RightShoulder"]][angle_type.value][target_state.value])
        target_max = max(self.target_angles[self.body_part_indices["RightShoulder"]][angle_type.value][target_state.value])

        result = {
            "angle": angle,
            "body_part": "RightShoulder",
            "angle_type": angle_type,
            "target_min": target_min,
            "target_max": target_max,
            "target_state": target_state,
            "result_state": self._get_angle_analysis_result_state(angle, target_state, target_start, target_end, target_min, target_max, tolerance),
        }
        return result

    def _check_angle_hip_left(self, angle: float, angle_type: AngleTypes, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        target_start = self.target_angles[self.body_part_indices["LeftHip"]][angle_type.value][AngleTargetStates.START.value]
        target_end = self.target_angles[self.body_part_indices["LeftHip"]][angle_type.value][AngleTargetStates.END.value]
        target_min = min(self.target_angles[self.body_part_indices["LeftHip"]][angle_type.value][target_state.value])
        target_max = max(self.target_angles[self.body_part_indices["LeftHip"]][angle_type.value][target_state.value])

        result = {
            "angle": angle,
            "body_part": "LeftHip",
            "angle_type": angle_type,
            "target_min": target_min,
            "target_max": target_max,
            "target_state": target_state,
            "result_state": self._get_angle_analysis_result_state(angle, target_state, target_start, target_end, target_min, target_max, tolerance),
        }
        return result

    def _check_angle_hip_right(self, angle: float, angle_type: AngleTypes, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        target_start = self.target_angles[self.body_part_indices["RightHip"]][angle_type.value][AngleTargetStates.START.value]
        target_end = self.target_angles[self.body_part_indices["RightHip"]][angle_type.value][AngleTargetStates.END.value]
        target_min = min(self.target_angles[self.body_part_indices["RightHip"]][angle_type.value][target_state.value])
        target_max = max(self.target_angles[self.body_part_indices["RightHip"]][angle_type.value][target_state.value])

        result = {
            "angle": angle,
            "body_part": "RightHip",
            "angle_type": angle_type,
            "target_min": target_min,
            "target_max": target_max,
            "target_state": target_state,
            "result_state": self._get_angle_analysis_result_state(angle, target_state, target_start, target_end, target_min, target_max, tolerance),
        }
        return result

    def _check_angle_elbow_left(self, angle: float, angle_type: AngleTypes, target_state: AngleTargetStates, tolerance: int = 10):
        target_start = self.target_angles[self.body_part_indices["LeftElbow"]][angle_type.value][AngleTargetStates.START.value]
        target_end = self.target_angles[self.body_part_indices["LeftElbow"]][angle_type.value][AngleTargetStates.END.value]
        target_min = min(self.target_angles[self.body_part_indices["LeftElbow"]][angle_type.value][target_state.value])
        target_max = max(self.target_angles[self.body_part_indices["LeftElbow"]][angle_type.value][target_state.value])

        result = {
            "angle": angle,
            "body_part": "LeftElbow",
            "angle_type": angle_type,
            "target_min": target_min,
            "target_max": target_max,
            "target_state": target_state,
            "result_state": self._get_angle_analysis_result_state(angle, target_state, target_start, target_end, target_min, target_max, tolerance),
        }
        return result

    def _check_angle_elbow_right(self, angle: float, angle_type: AngleTypes, target_state: AngleTargetStates, tolerance: int = 10):
        target_start = self.target_angles[self.body_part_indices["RightElbow"]][angle_type.value][AngleTargetStates.START.value]
        target_end = self.target_angles[self.body_part_indices["RightElbow"]][angle_type.value][AngleTargetStates.END.value]
        target_min = min(self.target_angles[self.body_part_indices["RightElbow"]][angle_type.value][target_state.value])
        target_max = max(self.target_angles[self.body_part_indices["RightElbow"]][angle_type.value][target_state.value])

        result = {
            "angle": angle,
            "body_part": "RightElbow",
            "angle_type": angle_type,
            "target_min": target_min,
            "target_max": target_max,
            "target_state": target_state,
            "result_state": self._get_angle_analysis_result_state(angle, target_state, target_start, target_end, target_min, target_max, tolerance),
        }
        return result

    def _check_angle_knee_left(self, angle: float, angle_type: AngleTypes, target_state: AngleTargetStates, tolerance: int = 10):
        target_start = self.target_angles[self.body_part_indices["LeftKnee"]][angle_type.value][AngleTargetStates.START.value]
        target_end = self.target_angles[self.body_part_indices["LeftKnee"]][angle_type.value][AngleTargetStates.END.value]
        target_min = min(self.target_angles[self.body_part_indices["LeftKnee"]][angle_type.value][target_state.value])
        target_max = max(self.target_angles[self.body_part_indices["LeftKnee"]][angle_type.value][target_state.value])

        result = {
            "angle": angle,
            "body_part": "LeftKnee",
            "angle_type": angle_type,
            "target_min": target_min,
            "target_max": target_max,
            "target_state": target_state,
            "result_state": self._get_angle_analysis_result_state(angle, target_state, target_start, target_end, target_min, target_max, tolerance),
        }
        return result

    def _check_angle_knee_right(self, angle: float, angle_type: AngleTypes, target_state: AngleTargetStates, tolerance: int = 10):
        target_start = self.target_angles[self.body_part_indices["RightKnee"]][angle_type.value][AngleTargetStates.START.value]
        target_end = self.target_angles[self.body_part_indices["RightKnee"]][angle_type.value][AngleTargetStates.END.value]
        target_min = min(self.target_angles[self.body_part_indices["RightKnee"]][angle_type.value][target_state.value])
        target_max = max(self.target_angles[self.body_part_indices["RightKnee"]][angle_type.value][target_state.value])

        result = {
            "angle": angle,
            "body_part": "RightKnee",
            "angle_type": angle_type,
            "target_min": target_min,
            "target_max": target_max,
            "target_state": target_state,
            "result_state": self._get_angle_analysis_result_state(angle, target_state, target_start, target_end, target_min, target_max, tolerance),
        }
        return result

    def _get_angle_analysis_result_state(self,
                                         angle: float,
                                         target_state: AngleTargetStates,
                                         target_start: list,
                                         target_end: list,
                                         target_min: float,
                                         target_max: float,
                                         tolerance: int) -> AngleAnalysisResultStates:

        target_end_is_flexion = target_end[1] > target_start[1]
        target_end_is_extension = target_end[1] < target_start[1]
        target_end_is_zero = target_end[1] == target_start[1]

        result_state = AngleAnalysisResultStates.NONE.value
        if target_state.value == AngleTargetStates.START.value:
            if angle < target_min - tolerance:
                result_state = AngleAnalysisResultStates.TARGET_UNDERCUT.value if target_end_is_flexion else result_state
                result_state = AngleAnalysisResultStates.TARGET_EXCEEDED.value if target_end_is_extension else result_state
                result_state = AngleAnalysisResultStates.TARGET_EXCEEDED.value if target_end_is_zero else result_state
            if angle > target_max + tolerance:
                result_state = AngleAnalysisResultStates.TARGET_EXCEEDED.value if target_end_is_flexion else result_state
                result_state = AngleAnalysisResultStates.TARGET_UNDERCUT.value if target_end_is_extension else result_state
                result_state = AngleAnalysisResultStates.TARGET_EXCEEDED.value if target_end_is_zero else result_state
            if angle <= target_max + tolerance and angle >= target_min - tolerance:
                result_state = AngleAnalysisResultStates.IN_TARGET_RANGE.value

        if target_state.value == AngleTargetStates.END.value:
            if angle < target_min - tolerance:
                result_state = AngleAnalysisResultStates.TARGET_UNDERCUT.value if target_end_is_flexion else result_state
                result_state = AngleAnalysisResultStates.TARGET_EXCEEDED.value if target_end_is_extension else result_state
                result_state = AngleAnalysisResultStates.TARGET_EXCEEDED.value if target_end_is_zero else result_state
            if angle > target_max + tolerance:
                result_state = AngleAnalysisResultStates.TARGET_EXCEEDED.value if target_end_is_flexion else result_state
                result_state = AngleAnalysisResultStates.TARGET_UNDERCUT.value if target_end_is_extension else result_state
                result_state = AngleAnalysisResultStates.TARGET_EXCEEDED.value if target_end_is_zero else result_state
            if angle <= target_max + tolerance and angle >= target_min - tolerance:
                result_state = AngleAnalysisResultStates.IN_TARGET_RANGE.value

        return result_state
