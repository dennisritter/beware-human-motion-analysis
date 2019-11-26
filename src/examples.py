import tslearn.metrics as ts
import numpy as np
from hma.movement_analysis.pose_processor import PoseProcessor
from hma.movement_analysis.enums.pose_format_enum import PoseFormatEnum
from hma.movement_analysis.exercise import Exercise
from hma.movement_analysis import exercise_loader
from hma.movement_analysis.helpers import reformat_angles_dtw
from hma.movement_analysis.exercise_evaluator import ExerciseEvaluator
import json


def load_sequence_example(visualise=False):
    """Loading and visualising a sequence from file"""
    mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)
    filename = "data/sequences/191024_tracking/single/squat/user-2/191024__single__squat__user-2__1.json"
    sequence = mocap_poseprocessor.load(filename)
    if visualise:
        sequence.visualise()
    return sequence


# load_sequence_example(visualise=True)


def load_exercise_example():
    """Loading an Exercise"""
    return exercise_loader.load("data/exercises/squat.json")


# load_exercise_example()


def identify_exercise_type_example():
    """Identify the type of an exercise"""
    mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)
    filename_q = "data/sequences/191024_tracking/single/squat/user-3/191024__single__squat__user-3__1.json"
    filename_squat_gt = "data/sequences/191024_tracking/single/squat/user-2/191024__single__squat__user-2__1.json"
    filename_biceps_curl_left_gt = "data/sequences/191024_tracking/single/biceps_curl_left/user-2/191024__single__biceps_curl_left__user-2__1.json"

    sequence_q = mocap_poseprocessor.load(filename_q, str(filename_q).split('\\')[-1])
    sequence_squat_gt = mocap_poseprocessor.load(filename_squat_gt, "Squat Ground-Truth")
    sequence_biceps_curl_left_gt = mocap_poseprocessor.load(filename_biceps_curl_left_gt, "Biceps Curl Left Ground-Truth")

    angles_q = reformat_angles_dtw(sequence_q)
    angles_squat_gt = reformat_angles_dtw(sequence_squat_gt)
    angles_biceps_curl_left_gt = reformat_angles_dtw(sequence_biceps_curl_left_gt)

    path_squat_gt, dist_squat_gt = ts.dtw_path(angles_q, angles_squat_gt)[1]
    path_biceps_curl_left_gt, dist_biceps_curl_left_gt = ts.dtw_path(angles_q, angles_biceps_curl_left_gt)[1]

    # The Ground-Truth Sequences Exercise type that results in the lowest distance is assumed to be the same exercise as performed in the Query Sequence.
    print("-- Listing distances between Query Sequence and Ground-Truth Sequences --")
    print(f"Query Sequences DTW distance to {sequence_squat_gt.name}: {dist_squat_gt}")
    print(f"Query Sequences DTW distance to {sequence_biceps_curl_left_gt.name}: {dist_biceps_curl_left_gt}")


# identify_exercise_type_example()


def identify_single_iteration_subsequences(plot=False):
    """Identify subsequences consisting of a single exercise repitition"""
    # Load Sequence and Exercise
    mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)
    filename = "data/sequences/191024_tracking/multi/squat/user-3/191024__multi__squat__user-3__0.json"
    sequence = mocap_poseprocessor.load(filename, str(filename).split('\\')[-1])
    squat = exercise_loader.load("data/exercises/squat.json")
    # Init ExerciseEvaluiator with an Exercise and a Sequence
    EE = ExerciseEvaluator(squat, sequence)
    # Identify Single Iteration Subsequences
    iterations = EE.find_iteration_keypoints(plot)
    print(f"Subsequencing results for '{sequence.name}'")
    print(f"Identified Iterations: {len(iterations)}")
    print(f"IdentifiedSubsequences Start/Turn/End Frame Indices:\n{iterations}")


# identify_single_iteration_subsequences(plot=True)


def rate_exercise_performance(plot=False):
    """Rating trainees exercise execution performance - Angle Comparison Method"""
    # Load Sequence and Exercise
    mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)
    filename = "data/sequences/191024_tracking/multi/squat/user-2/191024__multi__squat__user-2__1.json"
    sequence = mocap_poseprocessor.load(filename, str(filename).split('\\')[-1])
    squat = exercise_loader.load("data/exercises/squat.json")
    # Init ExerciseEvaluiator with an Exercise and a Sequence
    EE = ExerciseEvaluator(squat, sequence)
    # Identify Single Iteration Subsequences
    # NOTE: The Angle Comparison rating method needs the Turning Frame index as parameter,
    #       thus we first have to utilise the find_iteration_keypoints method of the ExerciseEvaluator.
    iterations = EE.find_iteration_keypoints(plot)
    if len(iterations) > 0:
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
            with open(f"data/rating_result_{i}.json", 'w') as outfile:
                # The Results are structured in the following way:
                # 1. Dimension = Frame
                # 2. Dimension = Body Part (null means there are no angles for that body part)
                # 3. Dimension = Angle Type (0 = Flexion/Extension; 1 = Abduction/Adduction; 2 = Internal/External Rotation )
                json.dump({"result": rating_result}, outfile)
    else:
        print("No iterations identified. Unable to rate execution performance.")


# rate_exercise_performance()
