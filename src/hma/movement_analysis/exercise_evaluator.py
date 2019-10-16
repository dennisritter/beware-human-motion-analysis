from .exercise import Exercise
from .enums.angle_target_states import AngleTargetStates
from .enums.angle_types import AngleTypes
from .enums.angle_analysis_result_states import AngleAnalysisResultStates
from .sequence import Sequence
import warnings
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import argrelextrema, savgol_filter

# TODO: Check whether Sequence parameter for some function are really necessary.
# Some of them only use the sequences body_part attribute, which is also a attribute in the ExerciseEvaluator class.
# Either remove the ExerciseEvaluator.body_part_indices attribute or use it instead of sequence.body_parts in some functions.


class ExerciseEvaluator:
    """This class evaluates analyses motion sequences with respect to an Exercise.

    Attributes:
        exercise (Exercise):    The exercise to evaluate motion sequences for.
        sequence (Sequence):    The sequence to evaluate.
        target_angles (list):   A list of target angles defined in the given exercise.
        prio_angles(list):      A list of prioritised body parts and angletypes for the given exercise.
    """

    # Definitions of high, mid, low priorities as floats
    HIGH_PRIO = 1.0
    MEDIUM_PRIO = 0.5
    LOW_PRIO = 0.0

    def __init__(self, exercise: Exercise, sequence: Sequence):
        """Inits ExerciseEvaluator class with the given Exercise and Sequence"""

        # The Exercise to evaluate
        self.exercise = exercise
        # The sequence to evaluate
        self.sequence = sequence
        # The target_angles for each body part
        self.target_angles = self._get_target_angles()
        # The prioritised body parts and angles: [(<body_part_index>, <AngleType.KEY>)]
        self.prio_angles = self._get_prio_angles()
        
        # Process all ball joint angles of the sequence attribute
        # NOTE: Will change the Sequences angles!
        self._process_sequence_ball_joint_angles()

    def set_sequence(self, seq: Sequence):
        """Assigns the given sequence to the sequence attribute and performs necessary recalculations.

        Args:
            sequence (Sequence): The new sequence to set as sequence attribute.
        """
        self.sequence = seq
        # As the sequence attribute has changed, we have to recalculate target_angles and prio_angles.
        self.target_angles = self._get_target_angles()
        self.prio_angles = self._get_prio_angles()
        # And finally process the sequences' ball joint angles again.
        # NOTE: Will change the Sequences angles!
        self._process_sequence_ball_joint_angles()

    def _get_prio_angles(self) -> list:
        """Returns a list of tuples containing a body part mapped in Sequence.body_parts and the AngleType for that body part which is prioritised.
           Example: [(4, AngleType.FLEX_EX), (4, AngleType.AB_AD)]

        Returns:
            A list of tuples containing body part indices and angle types. 
            Example: [(4, AngleType.FLEX_EX), (4, AngleType.AB_AD)]
        """
        ex = self.exercise
        seq = self.sequence
        prio_angles = []
        # Shoulders
        if ex.angles["start"]["shoulder_left"]["flexion_extension"]["priority"] == self.HIGH_PRIO:
            prio_angles.append((seq.body_parts["LeftShoulder"], AngleTypes.FLEX_EX))
        if ex.angles["start"]["shoulder_left"]["abduction_adduction"]["priority"] == self.HIGH_PRIO:
            prio_angles.append((seq.body_parts["LeftShoulder"], AngleTypes.AB_AD))
        if ex.angles["start"]["shoulder_right"]["flexion_extension"]["priority"] == self.HIGH_PRIO:
            prio_angles.append((seq.body_parts["RightShoulder"], AngleTypes.FLEX_EX))
        if ex.angles["start"]["shoulder_right"]["abduction_adduction"]["priority"] == self.HIGH_PRIO:
            prio_angles.append((seq.body_parts["RightShoulder"], AngleTypes.AB_AD))
        # Hips
        if ex.angles["start"]["hip_left"]["flexion_extension"]["priority"] == self.HIGH_PRIO:
            prio_angles.append((seq.body_parts["LeftHip"], AngleTypes.FLEX_EX))
        if ex.angles["start"]["hip_left"]["abduction_adduction"]["priority"] == self.HIGH_PRIO:
            prio_angles.append((seq.body_parts["LeftHip"], AngleTypes.AB_AD))
        if ex.angles["start"]["hip_right"]["flexion_extension"]["priority"] == self.HIGH_PRIO:
            prio_angles.append((seq.body_parts["RightHip"], AngleTypes.FLEX_EX))
        if ex.angles["start"]["hip_right"]["abduction_adduction"]["priority"] == self.HIGH_PRIO:
            prio_angles.append((seq.body_parts["RightHip"], AngleTypes.AB_AD))
        # Elbows
        if ex.angles["start"]["elbow_left"]["flexion_extension"]["priority"] == self.HIGH_PRIO:
            prio_angles.append((seq.body_parts["LeftElbow"], AngleTypes.FLEX_EX))
        if ex.angles["start"]["elbow_right"]["flexion_extension"]["priority"] == self.HIGH_PRIO:
            prio_angles.append((seq.body_parts["RightElbow"], AngleTypes.FLEX_EX))
        # Knees
        if ex.angles["start"]["knee_left"]["flexion_extension"]["priority"] == self.HIGH_PRIO:
            prio_angles.append((seq.body_parts["LeftKnee"], AngleTypes.FLEX_EX))
        if ex.angles["start"]["knee_right"]["flexion_extension"]["priority"] == self.HIGH_PRIO:
            prio_angles.append((seq.body_parts["RightKnee"], AngleTypes.FLEX_EX))

        self.prio_angles = prio_angles
        return prio_angles

    def _get_target_angles(self) -> list:
        """Returns target angles for the exercise of this ExerciseEvaluator instance.

        Returns a 4-D ndarray which contain the Exercises' range of target angles for all body_parts, angle types and target states as minimum/maximum.
        The position of each body part in the returned list is mapped in the body_part attribute of the given sequence (seq.body_parts).
        The positions of the second dimension represents an angle type of the AngleTypes enum.
        The position in the third dimension represents the START (0) and END (1) state.
        The position in the fourth dimension represent the minimum value (0) and maximum value (1).

        Returns:
            Returns a 4-D ndarray which contain the Exercises' range of target angles for all body_parts, angle types and target states as minimum/maximum.
        """
        ex = self.exercise
        seq = self.sequence
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

    # TODO: Identify anad create sub functions to shrink the length of this function and increase overview.
    def find_iteration_keypoints(self, start_frame_min_dist: int=5, end_frame_min_dist: int=5, plot=False) -> list:
        """Finds iterations of a movement in a motion sequence.

        The function identifies Start, Mid and End positions of iterations in a motion sequence 
        based on shared minimum and maximum values of prioritised body part angles.

        Args:
            # NOTE: Probably changes again
            # TODO: Add doctring after finished
            start_frame_min_dist (int): 
            end_frame_min_dist (int):
            plot (Boolean): Determines whether to plot partial results of an execution of this function. 

        Returns:
            A 2-D list of frame indices of the given sequence that represent start, mid and end frames of an iteration. 
        """
        seq = self.sequence
        # Store all minima/maxima in a matrix of [0,1]
        # Rows represent prioritised angle (bodypart and angle type)
        # Columns represent Frames
        # 0 means no minimum/maximum; 1 means minimum/maximum found
        start_frame_matrix = np.zeros((len(self.prio_angles), len(seq)))
        turning_frame_matrix = np.zeros((len(self.prio_angles), len(seq)))
        # Gather information for plotting a summary graph
        angles_savgol_all_bps = np.zeros((len(self.prio_angles), len(seq)))
        minima_all_bps = [None]*len(self.prio_angles)
        maxima_all_bps = [None]*len(self.prio_angles)
        # minima_all_bps = np.zeros((len(self.prio_angles)))
        # maxima_all_bps = np.zeros((len(self.prio_angles)))

        # self.prio_angles element example format (idx, AngleType) -> (1, AngleType.FLEX_EX)
        for prio_idx, (body_part_idx, angle_type) in enumerate(self.prio_angles):

            # Get calculated angles of a specific type for a specific body part for all frames
            angles = seq.joint_angles[:, body_part_idx, angle_type.value]

            # TODO: Find best value for window size (51 seems to work well)
            # Apply a Savitzky-Golay Filter to get a list of 'smoothed' angles.
            # savgol_window_max = 51
            # savgol_window_generic = int(math.floor(len(angles)/1.5)+1 if math.floor(len(angles)/1.5) % 2 == 0 else math.floor(len(angles)/1.5))
            # savgol_window = min(savgol_window_max, savgol_window_generic)
            savgol_window = 51
            angles_savgol = savgol_filter(angles, savgol_window, 3, mode="nearest")
            angles_savgol_all_bps[prio_idx] = angles_savgol

            # TODO: Find best value for order parameter (10 seems to work well)
            # Find Minima and Maxima of angles after applying a Savitzky-Golay Filter filter
            maxima = argrelextrema(angles_savgol, np.greater, order=10)[0]
            minima = argrelextrema(angles_savgol, np.less, order=10)[0]

            # Add minimum to first and last frame if start_frame_min_dist/end_frame_min_dist
            # param value is not less than the actual distance to the target angle
            angles_savgol = np.array(angles_savgol)
            target_start_range = self.target_angles[body_part_idx][angle_type.value][AngleTargetStates.START.value]
            if (min(target_start_range) < angles_savgol[0] < max(target_start_range) or
                start_frame_min_dist > abs(angles_savgol[0]-target_start_range[0]) or
                    start_frame_min_dist > abs(angles_savgol[0]-target_start_range[1])):
                minima = np.insert(minima, 0, 0)
            if (min(target_start_range) < angles_savgol[-1] < max(target_start_range) or
                end_frame_min_dist > abs(angles_savgol[-1]-target_start_range[0]) or
                    end_frame_min_dist > abs(angles_savgol[-1]-target_start_range[1])):
                minima = np.append(minima, len(angles_savgol)-1)

            # Get Exercise targets for the current angle type
            ex_targets = self.target_angles[body_part_idx][angle_type.value]
            # Check if Exercise targets of target state END are greater than targets of START
            # We need this information to identify whether local MAXIMA or MINIMA represent start/end of a subsequence
            target_end_greater_start = min(ex_targets[AngleTargetStates.END.value]) > min(ex_targets[AngleTargetStates.START.value])

            # Check if distance to Exercise START target angle is greater than Exercise END target angle
            # in order to remove falsy maxima/minima
            def _dist_filter(x): return abs(angles_savgol[x] - min(ex_targets[AngleTargetStates.START.value])) > abs(angles_savgol[x] - min(ex_targets[AngleTargetStates.END.value]))
            if target_end_greater_start:
                maxima = maxima[_dist_filter(maxima)]
                minima = minima[np.invert(_dist_filter(minima))]
            else:
                maxima = maxima[np.invert(_dist_filter(maxima))]
                minima = minima[_dist_filter(minima)]

            maxima_all_bps[prio_idx] = maxima
            minima_all_bps[prio_idx] = minima
            # Add 1 values to frames where a minimum/maximum has been found.
            # If movement type is a extension/adduction (target_end_greater_start == false),
            # add minima to turning frames matrix and maxima to starting frames matrix
            for minimum in minima:
                if target_end_greater_start:
                    start_frame_matrix[prio_idx][minimum] = 1.0
                else:
                    turning_frame_matrix[prio_idx][minimum] = 1.0
            for maximum in maxima:
                if target_end_greater_start:
                    turning_frame_matrix[prio_idx][maximum] = 1.0
                else:
                    start_frame_matrix[prio_idx][maximum] = 1.0

            # if plot:
            #     plt.plot(range(0, len(angles)), angles, zorder=1, linewidth="1.0")
            #     plt.plot(range(0, len(angles)), angles_savgol, color='red', zorder=1, linewidth="1.0")
            #     plt.scatter(maxima, angles_savgol[maxima], color='green', marker="^", zorder=2, facecolors='none')
            #     plt.scatter(minima, angles_savgol[minima], color='green', marker="v", zorder=2, facecolors='none')
            #     plt.show()

        # TODO: What if we have only two prioritised angles? -> 100% must be correct? 50% must be correct? Something better?
        confirm_extrema_thresh = len(self.prio_angles) - 1
        # Window size that determines the range of frames minima/maxima of different body parts belong to each other.
        w_size = 30
        confirmed_start_frames = self._confirm_extrema(start_frame_matrix, w_size, confirm_extrema_thresh)
        confirmed_turning_frames = self._confirm_extrema(turning_frame_matrix, w_size, confirm_extrema_thresh)

        if plot:
            for prio_idx in range(len(angles_savgol_all_bps)):
                maxima = maxima_all_bps[prio_idx].astype(int)
                minima = minima_all_bps[prio_idx].astype(int)
                plt.plot(range(0, len(angles_savgol_all_bps[prio_idx])), angles_savgol_all_bps[prio_idx], color='red', zorder=1, linewidth="1.0")
                plt.scatter(maxima, angles_savgol_all_bps[prio_idx][maxima], color='green', marker="^", zorder=2, facecolors='none')
                plt.scatter(minima, angles_savgol_all_bps[prio_idx][minima], color='green', marker="v", zorder=2, facecolors='none')
            plt.scatter(confirmed_start_frames, np.zeros(confirmed_start_frames.shape), color='blue', marker="v", zorder=3)
            plt.scatter(confirmed_turning_frames, np.full(confirmed_turning_frames.shape, 180), color='blue', marker="^", zorder=3)
            plt.show()

        iterations = []
        last_end_frame = None
        # We Don't need to iterate over the last element because this can't be the start of a iteration anymore.
        for sf_idx in range(0, len(confirmed_start_frames)-1):
            # Ensure that next start frame is greater equal than last end frame
            if last_end_frame is not None and last_end_frame > confirmed_start_frames[sf_idx]:
                continue
            # Only keep turning frames that occur later than the current start frame
            start_frame = confirmed_start_frames[sf_idx]
            confirmed_turning_frames = confirmed_turning_frames[start_frame < confirmed_turning_frames]
            # If there is no element in confirmed_turning_frames that is greater the current start_frame,
            # we can exit the loop since following start_frames will be even higher
            if len(confirmed_turning_frames) == 0:
                break
            # If there are still elements left in confirmed_turning_frames, take the smallest one as our turning point
            else:
                turning_frame = confirmed_turning_frames[0]
                confirmed_end_frames = confirmed_start_frames[turning_frame < confirmed_start_frames]
                if len(confirmed_end_frames) == 0:
                    break
                # If there are still elements left in confirmed_end_frames, take the smallest one as our end point
                else:
                    end_frame = confirmed_end_frames[0]
                    last_end_frame = end_frame
                    iterations.append([start_frame, turning_frame, end_frame])

        return iterations

    def _confirm_extrema(self, extrema_matrix: np.ndarray, w_size: int, confirm_extrema_thresh: int) -> np.ndarray:
        """Returns a 1-D numpy ndarray of extrema (minima or maxima).

        Args: 
            extrema_matrix: np.ndarray - A 2-D matrix of 0 and 1 where each row represents all frames for some angle.
                 Each 1 represents a extremum for the respective angle and frame.
            w_size: int - The window size. Determines how many frames are going to be summarized 
                when determining whether a extremum can be accounted to all bodyparts (can be confirmed).
            confirm_extrema_thresh: int - Determines how many angles (rows) of the window must contain at least
                one extremum so the extremum will be confirmed.

        Returns:
            A 1-D numpy ndarray of extrema (minima or maxima).
        """
        confirmed_extrema = []

        for column_idx in range(0, extrema_matrix.shape[1]):
            w_start = column_idx
            w_end = w_start + w_size
            window = extrema_matrix[:, w_start:w_end]

            # Check how many window rows include at least one extremum
            w_row_extrema = 0
            for w_row in window:
                w_row_extrema += 1 if (np.sum(w_row) >= 1.0) else 0
            # If enough window rows include an extremum
            if w_row_extrema >= confirm_extrema_thresh:
                # Find first extremum in window
                extrema_in_window = np.argwhere(window > 0)
                first_window_extremum = np.min(extrema_in_window[:, 1])
                last_window_extremum = np.max(extrema_in_window[:, 1])
                # Use average index between first and last extremum in window as confirmed extremum index
                confirmed_extremum = int((first_window_extremum + last_window_extremum)/2) + w_start
                confirmed_extrema.append(confirmed_extremum)
                # Remove all 1.0 values from the current window slice of the extrema_matrix
                extrema_matrix[:, w_start:w_end] = np.zeros(window.shape)

        return np.array(confirmed_extrema)

    def evaluate(self, switch_state_idx: int) -> list:
        """Evaluates the Sequence of this ExerciseEvaluator instance with respect to the Exercise.

        Args:
            switch_state_idx (int): The sequence index where the target state to evaluate against changes from END to START 

        Returns:
            A list of evaluation results containing: Frames -> body_parts -> AngleTypes -> Result dictionaries
            Example indexing of a specific result: result[<frame>][<body_part_index>][<angle_type.value>]
        """
        seq = self.sequence
        bp = seq.body_parts

        results = []
        current_target_state = AngleTargetStates.END
        for frame in range(len(seq)):
            if frame == switch_state_idx + 1:
                current_target_state = AngleTargetStates.START
            # Shoulders
            shoulder_left_angle_flex_ex = seq.joint_angles[frame][bp["LeftShoulder"]][AngleTypes.FLEX_EX.value]
            shoulder_left_angle_abd_add = seq.joint_angles[frame][bp["LeftShoulder"]][AngleTypes.AB_AD.value]
            shoulder_right_angle_flex_ex = seq.joint_angles[frame][bp["RightShoulder"]][AngleTypes.FLEX_EX.value]
            shoulder_right_angle_abd_add = seq.joint_angles[frame][bp["RightShoulder"]][AngleTypes.AB_AD.value]
            # Hips
            hip_left_angle_flex_ex = seq.joint_angles[frame][bp["LeftHip"]][AngleTypes.FLEX_EX.value]
            hip_left_angle_abd_add = seq.joint_angles[frame][bp["LeftHip"]][AngleTypes.AB_AD.value]
            hip_right_angle_flex_ex = seq.joint_angles[frame][bp["RightHip"]][AngleTypes.FLEX_EX.value]
            hip_right_angle_abd_add = seq.joint_angles[frame][bp["RightHip"]][AngleTypes.AB_AD.value]
            # Elbows
            elbow_left_angle_flex_ex = seq.joint_angles[frame][bp["LeftElbow"]][AngleTypes.FLEX_EX.value]
            elbow_right_angle_flex_ex = seq.joint_angles[frame][bp["RightElbow"]][AngleTypes.FLEX_EX.value]
            # Knees
            knee_left_angle_flex_ex = seq.joint_angles[frame][bp["LeftKnee"]][AngleTypes.FLEX_EX.value]
            knee_right_angle_flex_ex = seq.joint_angles[frame][bp["RightKnee"]][AngleTypes.FLEX_EX.value]

            # Everything gets overridden? WHY?
            frame_result = [None]*len(bp)
            frame_result[bp["LeftShoulder"]] = self._get_results_shoulder_left(shoulder_left_angle_flex_ex, shoulder_left_angle_abd_add, current_target_state, 10)
            frame_result[bp["RightShoulder"]] = self._get_results_shoulder_right(shoulder_right_angle_flex_ex, shoulder_right_angle_abd_add, current_target_state, 10)
            frame_result[bp["LeftHip"]] = self._get_results_hip_left(hip_left_angle_flex_ex, hip_left_angle_abd_add, current_target_state, 10)
            frame_result[bp["RightHip"]] = self._get_results_hip_right(hip_right_angle_flex_ex, hip_right_angle_abd_add, current_target_state, 10)
            frame_result[bp["LeftElbow"]] = self._get_results_elbow_left(elbow_left_angle_flex_ex, current_target_state, 10)
            frame_result[bp["RightElbow"]] = self._get_results_elbow_right(elbow_right_angle_flex_ex, current_target_state, 10)
            frame_result[bp["LeftKnee"]] = self._get_results_knee_left(knee_left_angle_flex_ex, current_target_state, 10)
            frame_result[bp["RightKnee"]] = self._get_results_knee_right(knee_right_angle_flex_ex, current_target_state, 10)
            results.append(frame_result)

        return results

    def _process_sequence_ball_joint_angles(self, ignore_flex_abd90_delta: int = 20):
        """Processes this ExerciseEvaluators sequences' ball joint angles for all ball joints and applies changes to the sequence. 
        
        Args:
            ignore_flex_abd90_delta (int):  Determines the maximum distance to a 90 degrees abduction/adduction angle, from where the flexion/extension angle is ignored.
                        	                Default=20;
        """
        seq = self.sequence
        bp = seq.body_parts
        for i in range(len(seq)):
            processed_ls = self._process_ball_joint_angles(
                seq.joint_angles[i][bp["LeftShoulder"]][AngleTypes.FLEX_EX.value],
                seq.joint_angles[i][bp["LeftShoulder"]][AngleTypes.AB_AD.value], bp["LeftShoulder"])
            seq.joint_angles[i][bp["LeftShoulder"]] = [processed_ls[0], processed_ls[1], seq.joint_angles[i][bp["LeftShoulder"]][AngleTypes.IN_EX_ROT.value]]
            processed_rs = self._process_ball_joint_angles(
                seq.joint_angles[i][bp["RightShoulder"]][AngleTypes.FLEX_EX.value],
                seq.joint_angles[i][bp["RightShoulder"]][AngleTypes.AB_AD.value], bp["RightShoulder"])
            seq.joint_angles[i][bp["RightShoulder"]] = [processed_rs[0], processed_rs[1], seq.joint_angles[i][bp["RightShoulder"]][AngleTypes.IN_EX_ROT.value]]
            processed_lh = self._process_ball_joint_angles(
                seq.joint_angles[i][bp["LeftHip"]][AngleTypes.FLEX_EX.value],
                seq.joint_angles[i][bp["LeftHip"]][AngleTypes.AB_AD.value], bp["LeftHip"])
            seq.joint_angles[i][bp["LeftHip"]] = [processed_lh[0], processed_lh[1], seq.joint_angles[i][bp["LeftHip"]][AngleTypes.IN_EX_ROT.value]]
            processed_rh = self._process_ball_joint_angles(
                seq.joint_angles[i][bp["RightHip"]][AngleTypes.FLEX_EX.value],
                seq.joint_angles[i][bp["RightHip"]][AngleTypes.AB_AD.value], bp["RightHip"])
            seq.joint_angles[i][bp["RightHip"]] = [processed_rh[0], processed_rh[1], seq.joint_angles[i][bp["RightHip"]][AngleTypes.IN_EX_ROT.value]]

    def _process_ball_joint_angles(self,
                                  angle_flex_ex: float,
                                  angle_abd_add: float,
                                  bp_idx: int,
                                  ignore_flex_abd90_delta: int = 20) -> tuple:
        """Processes ball joint angles to improve clinical representation of those.

        (1) Abduction/Adduction angles range after initial calculations is [-90, 90] degrees, 
        but the range of motion is [-180,180] degrees, the value must be extended
        by -90/90 degrees if a prioritised angle of the exercise is an abduction/adduction and 
        corresponding the flexion angle is greater 90 or less than -90.

        (2) The second operation resets the flexion/extension angle to zero(0) whenever the abduction/adduction
        angles distance is less than the ignore_flex_abd90_delta parameters value from 90 degrees.
        Because flexion/extension angles are very sensitive and error prone when close to the X-Axis
        because it represents a rotation around it.

        Args:
            angle_flex_ex (float):
                The flexion/extension angle to process.
            angle_abd_add (float):
                The abduction/adduction angle to process.
            bp_idx (int): The sequences body_part index of the processed angles.
            ignore_flex_abd90_delta (int):
                Determines the maximum distance to a 90 degrees abduction/adduction angle, from where the flexion/extension angle is ignored.
                Default=20;

        Returns:
            A tuple containing the processed flexion/extension angle as first element and the processed
            abduction/adduction angle as second element.
            Example: (85.0, 11.0)
        """
        # Check if angle-vector.y is higher than origin and adjust abduction/adduction angles if conditions are met.
        # If flexion angle is >90.0, angle-vector.y is higher than origin because flexion angle represents a rotation about the X-Axis
        if angle_flex_ex > 90.0 or angle_flex_ex < -90:
            for pa in self.prio_angles:
                # TODO: Maybe it is better to check for the expected range of motion instead of priority
                # If Abduction/Adduction is a prio angle type, fix the 90° limitation
                if pa[0] == bp_idx and pa[1] == AngleTypes.AB_AD:
                    if angle_abd_add < 0:
                        angle_abd_add -= 90

                    if angle_abd_add > 0:
                        angle_abd_add += 90

        # Set Flexion/Extension to 0.0° when angle-vector is close to X-Axis.
        # -> Flexion/Extension angles get very sensitive and error prone when close to X-Axis because it represents a rotation around it.
        full_absolute_abd_add = 90.0
        if abs(full_absolute_abd_add - abs(angle_abd_add)) < ignore_flex_abd90_delta:
            angle_flex_ex = 0.0

        return angle_flex_ex, angle_abd_add

    def _get_results_shoulder_left(self, angle_flex_ex: float, angle_abd_add: float, target_state: AngleTargetStates, tolerance: int = 10) -> list:
        """Returns a list of evaluation result dictionaries of all AngleTypes for the left shoulder joint.

            Args:
                angle_flex_ex (float):              The flexion/extension angle.
                angle_abd_add (float):              The abduction/adduction angle.
                target_state (AngleTargetStates):   The target state of the class instances Exercise to evaluate against.
                tolerance (int):                    The tolerance of angles to be treated as in range of the target (Default=10).

            Returns: A list of evaluation result dictionaries of all AngleTypes for the left shoulder joint.
        """
        result = [None] * len(AngleTypes)
        result[AngleTypes.FLEX_EX.value] = self._check_angle_shoulder_left(angle_flex_ex, AngleTypes.FLEX_EX, target_state, tolerance)
        result[AngleTypes.AB_AD.value] = self._check_angle_shoulder_left(angle_abd_add, AngleTypes.AB_AD, target_state, tolerance)
        return result

    def _get_results_shoulder_right(self, angle_flex_ex: float, angle_abd_add: float, target_state: AngleTargetStates, tolerance: int = 10) -> list:
        """Returns a list of evaluation result dictionaries of all AngleTypes for the right shoulder joint.

            Args:
                angle_flex_ex (float):              The flexion/extension angle.
                angle_abd_add (float):              The abduction/adduction angle.
                target_state (AngleTargetStates):   The target state of the class instances Exercise to evaluate against.
                tolerance (int):                    The tolerance of angles to be treated as in range of the target (Default=10).

            Returns: A list of evaluation result dictionaries of all AngleTypes for the right shoulder joint.
        """
        result = [None] * len(AngleTypes)
        result[AngleTypes.FLEX_EX.value] = self._check_angle_shoulder_right(angle_flex_ex, AngleTypes.FLEX_EX, target_state, tolerance)
        result[AngleTypes.AB_AD.value] = self._check_angle_shoulder_right(angle_abd_add, AngleTypes.AB_AD, target_state, tolerance)
        return result

    def _get_results_hip_left(self, angle_flex_ex: float, angle_abd_add: float, target_state: AngleTargetStates, tolerance: int = 10) -> list:
        """Returns a list of evaluation result dictionaries of all AngleTypes for the left hip joint.

            Args:
                angle_flex_ex (float):              The flexion/extension angle.
                angle_abd_add (float):              The abduction/adduction angle.
                target_state (AngleTargetStates):   The target state of the class instances Exercise to evaluate against.
                tolerance (int):                    The tolerance of angles to be treated as in range of the target (Default=10).

            Returns: A list of evaluation result dictionaries of all AngleTypes for the left hip joint.
        """
        result = [None] * len(AngleTypes)
        result[AngleTypes.FLEX_EX.value] = self._check_angle_hip_left(angle_flex_ex, AngleTypes.FLEX_EX, target_state, tolerance)
        result[AngleTypes.AB_AD.value] = self._check_angle_hip_left(angle_abd_add, AngleTypes.AB_AD, target_state, tolerance)
        return result

    def _get_results_hip_right(self, angle_flex_ex: float, angle_abd_add: float, target_state: AngleTargetStates, tolerance: int = 10) -> list:
        """Returns a list of evaluation result dictionaries of all AngleTypes for the right hip joint.

            Args:
                angle_flex_ex (float):              The flexion/extension angle.
                angle_abd_add (float):              The abduction/adduction angle.
                target_state (AngleTargetStates):   The target state of the class instances Exercise to evaluate against.
                tolerance (int):                    The tolerance of angles to be treated as in range of the target (Default=10).

            Returns: A list of evaluation result dictionaries of all AngleTypes for the right hip joint.
        """
        result = [None] * len(AngleTypes)
        result[AngleTypes.FLEX_EX.value] = self._check_angle_hip_right(angle_flex_ex, AngleTypes.FLEX_EX, target_state, tolerance)
        result[AngleTypes.AB_AD.value] = self._check_angle_hip_right(angle_abd_add, AngleTypes.AB_AD, target_state, tolerance)
        return result

    def _get_results_elbow_left(self, angle_flex_ex: float, target_state: AngleTargetStates, tolerance: int = 10) -> list:
        """Returns a list of evaluation result dictionaries of all AngleTypes for the left elbow joint.

            Args:
                angle_flex_ex (float):              The flexion/extension angle.
                target_state (AngleTargetStates):   The target state of the class instances Exercise to evaluate against.
                tolerance (int):                    The tolerance of angles to be treated as in range of the target (Default=10).

            Returns: A list of evaluation result dictionaries of all AngleTypes for the left elbow joint.
        """
        result = [None] * len(AngleTypes)
        result[AngleTypes.FLEX_EX.value] = self._check_angle_elbow_left(angle_flex_ex, AngleTypes.FLEX_EX, target_state, tolerance)
        return result

    def _get_results_elbow_right(self, angle_flex_ex: float, target_state: AngleTargetStates, tolerance: int = 10) -> list:
        """Returns a list of evaluation result dictionaries of all AngleTypes for the right elbow joint.

            Args:
                angle_flex_ex (float):              The flexion/extension angle.
                target_state (AngleTargetStates):   The target state of the class instances Exercise to evaluate against.
                tolerance (int):                    The tolerance of angles to be treated as in range of the target (Default=10).

            Returns: A list of evaluation result dictionaries of all AngleTypes for the right elbow joint.
        """
        result = [None] * len(AngleTypes)
        result[AngleTypes.FLEX_EX.value] = self._check_angle_elbow_right(angle_flex_ex, AngleTypes.FLEX_EX, target_state, tolerance)
        return result

    def _get_results_knee_left(self, angle_flex_ex: float, target_state: AngleTargetStates, tolerance: int = 10) -> list:
        """Returns a list of evaluation result dictionaries of all AngleTypes for the left knee joint.

            Args:
                angle_flex_ex (float):              The flexion/extension angle.
                target_state (AngleTargetStates):   The target state of the class instances Exercise to evaluate against.
                tolerance (int):                    The tolerance of angles to be treated as in range of the target (Default=10).

            Returns: A list of evaluation result dictionaries of all AngleTypes for the left knee joint.
        """
        result = [None] * len(AngleTypes)
        result[AngleTypes.FLEX_EX.value] = self._check_angle_knee_left(angle_flex_ex, AngleTypes.FLEX_EX, target_state, tolerance)
        return result

    def _get_results_knee_right(self, angle_flex_ex: float, target_state: AngleTargetStates, tolerance: int = 10) -> list:
        """Returns a list of evaluation result dictionaries of all AngleTypes for the right knee joint.

            Args:
                angle_flex_ex (float):              The flexion/extension angle.
                target_state (AngleTargetStates):   The target state of the class instances Exercise to evaluate against.
                tolerance (int):                    The tolerance of angles to be treated as in range of the target (Default=10).

            Returns: A list of evaluation result dictionaries of all AngleTypes for the right knee joint.
        """
        result = [None] * len(AngleTypes)
        result[AngleTypes.FLEX_EX.value] = self._check_angle_knee_right(angle_flex_ex, AngleTypes.FLEX_EX, target_state, tolerance)
        return result

    def _check_angle_shoulder_left(self, angle: float, angle_type: AngleTypes, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        """Returns a dictionary that represents the result evaluation of the given AngleTypes for the left shoulder joint with respect to the given AngleTargetState.

            Args:
                angle (float):                      The angle to evaluate.
                angle_type (AngleTypes):            The type of angle.
                target_state (AngleTargetStates):   The target state of the class instances Exercise to evaluate against.
                tolerance (int):                    The tolerance of angles to be treated as in range of the target (Default=10).

            Returns: A dictionary that represents the result evaluation of the given AngleTypes for the left shoulder joint with respect to the given AngleTargetState.
        """
        target_start = self.target_angles[self.sequence.body_parts["LeftShoulder"]][angle_type.value][AngleTargetStates.START.value]
        target_end = self.target_angles[self.sequence.body_parts["LeftShoulder"]][angle_type.value][AngleTargetStates.END.value]
        target_min = min(self.target_angles[self.sequence.body_parts["LeftShoulder"]][angle_type.value][target_state.value])
        target_max = max(self.target_angles[self.sequence.body_parts["LeftShoulder"]][angle_type.value][target_state.value])

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
        """Returns a dictionary that represents the result evaluation of the given AngleTypes for the right shoulder joint with respect to the given AngleTargetState.

            Args:
                angle (float):                      The angle to evaluate.
                angle_type (AngleTypes):            The type of angle.
                target_state (AngleTargetStates):   The target state of the class instances Exercise to evaluate against.
                tolerance (int):                    The tolerance of angles to be treated as in range of the target (Default=10).

            Returns: A dictionary that represents the result evaluation of the given AngleTypes for the right shoulder joint with respect to the given AngleTargetState.
        """
        target_start = self.target_angles[self.sequence.body_parts["RightShoulder"]][angle_type.value][AngleTargetStates.START.value]
        target_end = self.target_angles[self.sequence.body_parts["RightShoulder"]][angle_type.value][AngleTargetStates.END.value]
        target_min = min(self.target_angles[self.sequence.body_parts["RightShoulder"]][angle_type.value][target_state.value])
        target_max = max(self.target_angles[self.sequence.body_parts["RightShoulder"]][angle_type.value][target_state.value])

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
        """Returns a dictionary that represents the result evaluation of the given AngleTypes for the left hip joint with respect to the given AngleTargetState.

            Args:
                angle (float):                      The angle to evaluate.
                angle_type (AngleTypes):            The type of angle.
                target_state (AngleTargetStates):   The target state of the class instances Exercise to evaluate against.
                tolerance (int):                    The tolerance of angles to be treated as in range of the target (Default=10).

            Returns: A dictionary that represents the result evaluation of the given AngleTypes for the left hip joint with respect to the given AngleTargetState.
        """
        target_start = self.target_angles[self.sequence.body_parts["LeftHip"]][angle_type.value][AngleTargetStates.START.value]
        target_end = self.target_angles[self.sequence.body_parts["LeftHip"]][angle_type.value][AngleTargetStates.END.value]
        target_min = min(self.target_angles[self.sequence.body_parts["LeftHip"]][angle_type.value][target_state.value])
        target_max = max(self.target_angles[self.sequence.body_parts["LeftHip"]][angle_type.value][target_state.value])

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
        """Returns a dictionary that represents the result evaluation of the given AngleTypes for the right hip joint with respect to the given AngleTargetState.

            Args:
                angle (float):                      The angle to evaluate.
                angle_type (AngleTypes):            The type of angle.
                target_state (AngleTargetStates):   The target state of the class instances Exercise to evaluate against.
                tolerance (int):                    The tolerance of angles to be treated as in range of the target (Default=10).

            Returns: A dictionary that represents the result evaluation of the given AngleTypes for the left right hip with respect to the given AngleTargetState.
        """
        target_start = self.target_angles[self.sequence.body_parts["RightHip"]][angle_type.value][AngleTargetStates.START.value]
        target_end = self.target_angles[self.sequence.body_parts["RightHip"]][angle_type.value][AngleTargetStates.END.value]
        target_min = min(self.target_angles[self.sequence.body_parts["RightHip"]][angle_type.value][target_state.value])
        target_max = max(self.target_angles[self.sequence.body_parts["RightHip"]][angle_type.value][target_state.value])

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
        """Returns a dictionary that represents the result evaluation of the given AngleTypes for the left elbow joint with respect to the given AngleTargetState.

            Args:
                angle (float):                      The angle to evaluate.
                angle_type (AngleTypes):            The type of angle.
                target_state (AngleTargetStates):   The target state of the class instances Exercise to evaluate against.
                tolerance (int):                    The tolerance of angles to be treated as in range of the target (Default=10).

            Returns: A dictionary that represents the result evaluation of the given AngleTypes for the left elbow joint with respect to the given AngleTargetState.
        """
        target_start = self.target_angles[self.sequence.body_parts["LeftElbow"]][angle_type.value][AngleTargetStates.START.value]
        target_end = self.target_angles[self.sequence.body_parts["LeftElbow"]][angle_type.value][AngleTargetStates.END.value]
        target_min = min(self.target_angles[self.sequence.body_parts["LeftElbow"]][angle_type.value][target_state.value])
        target_max = max(self.target_angles[self.sequence.body_parts["LeftElbow"]][angle_type.value][target_state.value])

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
        """Returns a dictionary that represents the result evaluation of the given AngleTypes for the right elbow joint with respect to the given AngleTargetState.

            Args:
                angle (float):                      The angle to evaluate.
                angle_type (AngleTypes):            The type of angle.
                target_state (AngleTargetStates):   The target state of the class instances Exercise to evaluate against.
                tolerance (int):                    The tolerance of angles to be treated as in range of the target (Default=10).

            Returns: A dictionary that represents the result evaluation of the given AngleTypes for the right elbow joint with respect to the given AngleTargetState.
        """
        target_start = self.target_angles[self.sequence.body_parts["RightElbow"]][angle_type.value][AngleTargetStates.START.value]
        target_end = self.target_angles[self.sequence.body_parts["RightElbow"]][angle_type.value][AngleTargetStates.END.value]
        target_min = min(self.target_angles[self.sequence.body_parts["RightElbow"]][angle_type.value][target_state.value])
        target_max = max(self.target_angles[self.sequence.body_parts["RightElbow"]][angle_type.value][target_state.value])

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
        """Returns a dictionary that represents the result evaluation of the given AngleTypes for the left knee joint with respect to the given AngleTargetState.

            Args:
                angle (float):                      The angle to evaluate.
                angle_type (AngleTypes):            The type of angle.
                target_state (AngleTargetStates):   The target state of the class instances Exercise to evaluate against.
                tolerance (int):                    The tolerance of angles to be treated as in range of the target (Default=10).

            Returns: A dictionary that represents the result evaluation of the given AngleTypes for the left knee joint with respect to the given AngleTargetState.
        """
        target_start = self.target_angles[self.sequence.body_parts["LeftKnee"]][angle_type.value][AngleTargetStates.START.value]
        target_end = self.target_angles[self.sequence.body_parts["LeftKnee"]][angle_type.value][AngleTargetStates.END.value]
        target_min = min(self.target_angles[self.sequence.body_parts["LeftKnee"]][angle_type.value][target_state.value])
        target_max = max(self.target_angles[self.sequence.body_parts["LeftKnee"]][angle_type.value][target_state.value])

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
        """Returns a dictionary that represents the result evaluation of the given AngleTypes for the right shoulder joint with respect to the given AngleTargetState.

            Args:
                angle (float):                      The angle to evaluate.
                angle_type (AngleTypes):            The type of angle.
                target_state (AngleTargetStates):   The target state of the class instances Exercise to evaluate against.
                tolerance (int):                    The tolerance of angles to be treated as in range of the target (Default=10).

            Returns: A dictionary that represents the result evaluation of the given AngleTypes for the right shoulder joint with respect to the given AngleTargetState.
        """
        target_start = self.target_angles[self.sequence.body_parts["RightKnee"]][angle_type.value][AngleTargetStates.START.value]
        target_end = self.target_angles[self.sequence.body_parts["RightKnee"]][angle_type.value][AngleTargetStates.END.value]
        target_min = min(self.target_angles[self.sequence.body_parts["RightKnee"]][angle_type.value][target_state.value])
        target_max = max(self.target_angles[self.sequence.body_parts["RightKnee"]][angle_type.value][target_state.value])

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
        """Returns a AngleAnalysisResultStates value that represents the result of the evaluation of a specific angle.

            Args:
                angle (float):                      The angle to evaluate.
                target_state (AngleTargetStates):   The target state of the class instances Exercise to evaluate against.
                target_start (list):                The specified target angle range for the START target state.
                target_end (list):                  The specified target angle range for the END target state.2
                target_min (float):                 The minimum value of the target angle range. 
                target_max (float):                 The maximum value of the target angle range.
                tolerance (int):                    The tolerance of angles to be treated as in range of the target (Default=10).

            Returns: A AngleAnalysisResultStates value that represents the result of the evaluation of a specific angle.       	
        """

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
