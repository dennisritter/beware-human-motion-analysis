import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tslearn.metrics as ts
import seaborn as sns
import numpy as np
from pathlib import Path
from hma.movement_analysis.sequence import Sequence
from hma.movement_analysis.exercise import Exercise
from hma.movement_analysis.pose_processor import PoseProcessor
from hma.movement_analysis.enums.pose_format_enum import PoseFormatEnum
from hma.movement_analysis import angle_calculations as acm
from hma.movement_analysis import distance
from hma.movement_analysis.enums.angle_types import AngleTypes
from hma.movement_analysis import exercise_loader
from hma.movement_analysis.exercise_evaluator import ExerciseEvaluator
import tslearn.metrics as ts
import json
import seaborn as sns


# Calculating joint angles for a MOCAP sequence and returning a 2D-list containing all angles for each frame in consecutive order
def get_dtw_angles_mocap(seq):
    bp = seq.body_parts
    dtw_angles = []
    for frame in range(0, len(seq)):
        seq_frame_angles = []
        seq_frame_angles.append(seq.joint_angles[frame][bp["LeftShoulder"]][AngleTypes.FLEX_EX.value])
        seq_frame_angles.append(seq.joint_angles[frame][bp["LeftShoulder"]][AngleTypes.AB_AD.value])
        seq_frame_angles.append(seq.joint_angles[frame][bp["RightShoulder"]][AngleTypes.FLEX_EX.value])
        seq_frame_angles.append(seq.joint_angles[frame][bp["RightShoulder"]][AngleTypes.AB_AD.value])
        seq_frame_angles.append(seq.joint_angles[frame][bp["LeftHip"]][AngleTypes.FLEX_EX.value])
        seq_frame_angles.append(seq.joint_angles[frame][bp["LeftHip"]][AngleTypes.AB_AD.value])
        seq_frame_angles.append(seq.joint_angles[frame][bp["RightHip"]][AngleTypes.FLEX_EX.value])
        seq_frame_angles.append(seq.joint_angles[frame][bp["RightHip"]][AngleTypes.AB_AD.value])
        seq_frame_angles.append(seq.joint_angles[frame][bp["LeftElbow"]][AngleTypes.FLEX_EX.value])
        seq_frame_angles.append(seq.joint_angles[frame][bp["RightElbow"]][AngleTypes.FLEX_EX.value])
        seq_frame_angles.append(seq.joint_angles[frame][bp["LeftKnee"]][AngleTypes.FLEX_EX.value])
        seq_frame_angles.append(seq.joint_angles[frame][bp["RightKnee"]][AngleTypes.FLEX_EX.value])
        dtw_angles.append(seq_frame_angles)
    return np.array(dtw_angles)


def get_distances_dtw(seqs_gt, seqs_query):
    dtw_results = []
    for seq_gt in seqs_gt:
        for seq_query in seqs_query:
            path, dist = ts.dtw_path(get_dtw_angles_mocap(seq_gt), get_dtw_angles_mocap(seq_query))
            result = {
                "distance": dist,
                # "path": path,
                "ground_truth_name": seq_gt.name,
                "query_name": seq_query.name
            }
            dtw_results.append(result)
            print(f"DTW distance result ::: {seq_gt.name} <-> {seq_query.name}")
            print(f"Distance: {dist}")
    return dtw_results


def get_dtw_results():
    # Get PoseProcessor instance for MOCAP sequences
    mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)

    squat = exercise_loader.load('data/exercises/squat.json')
    biceps_curl_left = exercise_loader.load('data/exercises/biceps-curl-left.json')
    biceps_curl_right = exercise_loader.load('data/exercises/biceps-curl-right.json')
    knee_lift_left = exercise_loader.load('data/exercises/knee-lift-left.json')
    knee_lift_right = exercise_loader.load('data/exercises/knee-lift-right.json')

    # Load query sequences
    seqs_query = []
    filenames_query = list(Path("data/evaluation/matching/test_matching_dataset/squat").rglob("*.json"))
    filenames_query += list(Path("data/evaluation/matching/test_matching_dataset/biceps_curl_left").rglob("*.json"))
    filenames_query += list(Path("data/evaluation/matching/test_matching_dataset/biceps_curl_right").rglob("*.json"))
    filenames_query += list(Path("data/evaluation/matching/test_matching_dataset/knee_lift_left").rglob("*.json"))
    filenames_query += list(Path("data/evaluation/matching/test_matching_dataset/knee_lift_right").rglob("*.json"))

    for filename in filenames_query:
        print(f"Loading Query Sequence file: {filename}")
        sequence = mocap_poseprocessor.load(filename, str(filename).split('\\')[-1])
        seqs_query.append(sequence)

    # Load ground-truth sequences
    seqs_gt = []
    filenames_gt = list(Path("data/evaluation/matching/groundtruth/").rglob("*.json"))
    for filename in filenames_gt:
        print(f"Loading Ground Truth Sequence file: {filename}")
        sequence = mocap_poseprocessor.load(filename, str(filename).split('\\')[-1])
        seqs_gt.append(sequence)

    dtw_results = {
        "exercises": ["squat", "biceps_curl_left", "biceps_curl_right", "knee_lift_left", "knee_lift_right"],
        "payload": get_distances_dtw(seqs_gt, seqs_query)
    }
    with open(f"data/evaluation/matching/dtw_results.json", 'w') as outfile:
        json.dump(dtw_results, outfile)


def plot_dtw_result():
    with open(f"data/evaluation/matching/dtw_results.json", 'r') as result_file:
        serialised_json_result = json.load(result_file)
        results = serialised_json_result["payload"]
        exercises = serialised_json_result["exercises"]

    for result in results:
        # TODO: Group results by query sequence and compare distances
        pass


plot_dtw_result()
