import json
import tslearn.metrics as ts
from hma.movement_analysis.models.sequence import Sequence
from hma.movement_analysis import exercise_loader
from hma.movement_analysis.helpers import reformat_angles_dtw
from hma.movement_analysis.exercise_evaluator import ExerciseEvaluator
from hma.movement_analysis.skeleton_visualiser import SkeletonVisualiser


def load_sequence_example():
    """Loading a sequence from file"""
    filename = "data/sequences/191024_tracking/single/squat/user-2/191024__single__squat__user-2__1.json"
    sequence = Sequence.from_mocap_file(filename)
    return sequence


# load_sequence_example()


def visualise_sequence_example(filename: str = "data/sequences/191024_tracking/single/squat/user-2/191024__single__squat__user-2__1.json",
                               frames_from_to: list = [0, 1]):
    """Loading and visualising a sequence from file"""
    sequence = Sequence.from_mocap_file(filename)
    skeleton_vis = SkeletonVisualiser(sequence[frames_from_to[0]:frames_from_to[1]])
    skeleton_vis.show()


# visualise_sequence_example()


def load_exercise_example():
    """Loading an Exercise"""
    return exercise_loader.load("data/exercises/squat.json")


# load_exercise_example()


def identify_exercise_type_example():
    """Identify the type of an exercise"""
    filename_q = "data/sequences/191024_tracking/single/squat/user-3/191024__single__squat__user-3__1.json"
    filename_squat_gt = "data/sequences/191024_tracking/single/squat/user-2/191024__single__squat__user-2__1.json"
    filename_biceps_curl_left_gt = "data/sequences/191024_tracking/single/biceps_curl_left/user-2/191024__single__biceps_curl_left__user-2__1.json"

    sequence_q = Sequence.from_mocap_file(filename_q, str(filename_q).split('\\')[-1])
    sequence_squat_gt = Sequence.from_mocap_file(filename_squat_gt, "Squat Ground-Truth")
    sequence_biceps_curl_left_gt = Sequence.from_mocap_file(filename_biceps_curl_left_gt, "Biceps Curl Left Ground-Truth")

    angles_q = reformat_angles_dtw(sequence_q)
    angles_squat_gt = reformat_angles_dtw(sequence_squat_gt)
    angles_biceps_curl_left_gt = reformat_angles_dtw(sequence_biceps_curl_left_gt)

    path_squat_gt, dist_squat_gt = ts.dtw_path(angles_q, angles_squat_gt)
    path_biceps_curl_left_gt, dist_biceps_curl_left_gt = ts.dtw_path(angles_q, angles_biceps_curl_left_gt)

    # The Ground-Truth Sequences Exercise type that results in the lowest distance is assumed to be the same exercise as performed in the Query Sequence.
    print("-- Listing distances between Query Sequence and Ground-Truth Sequences --")
    print(f"Query Sequences DTW distance to {sequence_squat_gt.name}: {dist_squat_gt}")
    print(f"path {path_squat_gt}")
    print(f"-----")
    print(f"Query Sequences DTW distance to {sequence_biceps_curl_left_gt.name}: {dist_biceps_curl_left_gt}")
    print(f"path: {path_biceps_curl_left_gt}")
    print(f"-----")


# identify_exercise_type_example()


def identify_single_iteration_subsequences(plot=False):
    """Identify subsequences consisting of a single exercise repitition"""
    # Load Sequence and Exercise
    filename = "data/sequences/191024_tracking/multi/squat/user-3/191024__multi__squat__user-3__0.json"
    sequence = Sequence.from_mocap_file(filename, str(filename).split('\\')[-1])
    squat = exercise_loader.load("data/exercises/squat.json")
    # Init ExerciseEvaluiator with an Exercise and a Sequence
    EE = ExerciseEvaluator(squat, sequence)
    # Identify Single Iteration Subsequences
    iterations = EE.find_iteration_keypoints(plot)
    print(f"Subsequencing results for '{sequence.name}'")
    print(f"Identified Iterations: {len(iterations)}")
    print(f"IdentifiedSubsequences Start/Turn/End Frame Indices:\n{iterations}")


# identify_single_iteration_subsequences(plot=True)


def rate_exercise_performance(filename: str = "data/sequences/191024_tracking/multi/squat/user-2/191024__multi__squat__user-2__1.json",
                              output_name: str = "data/rating_result_",
                              plot=False):
    """Rating trainees exercise execution performance - Angle Comparison Method"""
    # Load Sequence and Exercise
    sequence = Sequence.from_mocap_file(filename, str(filename).split('\\')[-1])
    squat = exercise_loader.load("data/exercises/squat.json")
    # Init ExerciseEvaluiator with an Exercise and a Sequence
    EE = ExerciseEvaluator(squat, sequence)
    # Identify Single Iteration Subsequences
    # NOTE: The Angle Comparison rating method needs the Turning Frame index as parameter,
    #       thus we first have to utilise the find_iteration_keypoints method of the ExerciseEvaluator.
    iterations = EE.find_iteration_keypoints(plot)
    if len(iterations) >= 1:
        print(f"Subsequencing results for '{sequence.name}'")
        print(f"Identified Iterations: {len(iterations)}")
        print(f"IdentifiedSubsequences Start/Turn/End Frame Indices:\n{iterations}")
        for i, it in enumerate(iterations):
            # Set Sequence to one of the identified Subsequences
            EE.set_sequence(sequence[it[0]:it[2]])
            # The second element of an iteration is the turning frame.
            turning_frame = it[1]
            rating_result = EE.evaluate(turning_frame)
            # Create result JSON file
            with open(f"{output_name}{i}.json", 'w') as outfile:
                # The Results are structured in the following way:
                # 1. Dimension = Frame
                # 2. Dimension = Body Part (null means there are no angles for that body part)
                # 3. Dimension = Angle Type (0 = Flexion/Extension; 1 = Abduction/Adduction; 2 = Internal/External Rotation )
                json.dump({"result": rating_result}, outfile)
    else:
        print("No iterations identified. Unable to rate execution performance.")


# rate_exercise_performance(plot=True)
