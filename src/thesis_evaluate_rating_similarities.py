import matplotlib.pyplot as plt
from hma.movement_analysis.exercise import Exercise
from hma.movement_analysis.enums.pose_format_enum import PoseFormatEnum
from hma.movement_analysis.enums.angle_target_states import AngleTargetStates
from hma.movement_analysis.sequence import Sequence
from hma.movement_analysis.pose_processor import PoseProcessor
from hma.movement_analysis import exercise_loader
from hma.movement_analysis import angle_calculations as acm
from hma.movement_analysis import transformations
from hma.movement_analysis.exercise_evaluator import ExerciseEvaluator
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema, savgol_filter
import glob
from pathlib import Path
import json
import seaborn as sns
import tslearn.metrics as ts


mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)
squat = exercise_loader.load('data/exercises/squat.json')

# Load Ground-Truth Sequence
filename_gt = "data/evaluation/rating/similarities/ground_truth/191024__single__squat__user-2__13.json"
seq_gt = mocap_poseprocessor.load(filename_gt, str(filename_gt).split('\\')[-1])


def get_tf_angle_dist_and_dtw_dist():
    results = {
        "exercise": "squat",
        "ground_truth": seq_gt.name,
        "query_seqs": [],
        "tf_angle_dif": {},
        "distances": {}
    }
    # Load Query Sequences
    filenames = list(Path("data/evaluation/rating/similarities/single_squats_query/").rglob("*.json"))
    seqs = []
    for filename in filenames:
        print(f"Loading Sequence file: {filename}")
        sequence = mocap_poseprocessor.load(filename, str(filename).split('\\')[-1])
        seqs.append(sequence)

    EE = ExerciseEvaluator(squat, seq_gt)
    prio_angles = EE.prio_angles
    iterations = EE.find_iteration_keypoints()
    gt_tf = iterations[0][1]
    results = {
        "exercise": "squat",
        "ground_truth": seq_gt.name,
        "query_seqs": [],
        "tf_angle_dif": {},
        "distances": {}
    }
    for seq in seqs:
        print(f"Evaluating:  {seq.name}")
        EE.set_sequence(seq)
        iterations = EE.find_iteration_keypoints()
        if len(iterations) == 0:
            EE.find_iteration_keypoints(plot=True)
            continue
        results["query_seqs"].append(seq.name)
        for angle in prio_angles:
            tf_angle = seq.joint_angles[iterations[0][1], angle[0], angle[1].value]
            gt_tf_angle = seq.joint_angles[gt_tf, angle[0], angle[1].value]
            tf_angle_dif = gt_tf_angle - tf_angle

            seq_gt_angles = seq_gt.joint_angles[:, angle[0], angle[1].value]
            seq_query_angles = seq.joint_angles[:, angle[0], angle[1].value]
            path, dist = ts.dtw_path(seq_gt_angles, seq_query_angles)
            for key, value in seq.body_parts.items():
                if value == angle[0]:
                    if key not in results["tf_angle_dif"].keys() and key not in results["distances"].keys():
                        results["tf_angle_dif"][key] = [tf_angle_dif]
                        results["distances"][key] = [dist]
                    else:
                        results["tf_angle_dif"][key].append(tf_angle_dif)
                        results["distances"][key].append(dist)

    with open(f"data/evaluation/rating/similarities/results.json", 'w') as outfile:
        json.dump(results, outfile)


def plot_correlation(bp: str):
    with open(f"data/evaluation/rating/similarities/results.json", 'r') as result_file:
        serialised_json_result = json.load(result_file)
        distances = serialised_json_result["distances"][bp]
        distances = serialised_json_result["tf_angle_dif"][bp]
