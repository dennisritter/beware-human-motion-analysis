import matplotlib.pyplot as plt
from hma.movement_analysis.exercise import Exercise
from hma.movement_analysis.enums.pose_format_enum import PoseFormatEnum
from hma.movement_analysis.enums.angle_target_states import AngleTargetStates
from hma.movement_analysis.sequence import Sequence
from hma.movement_analysis.pose_processor import PoseProcessor
from hma.movement_analysis import exercise_loader
from hma.movement_analysis import angle_calculations as acm
from hma.movement_analysis import transformations
from hma.movement_analysis.enums.angle_types import AngleTypes
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
seq_gt = mocap_poseprocessor.load(filename_gt, "191024__single__squat__user-2__13")


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
            tf_angle_dif = abs(gt_tf_angle - tf_angle)

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


# get_tf_angle_dist_and_dtw_dist()

def get_avg_angle_dif_and_dtw_dist():
    results = {
        "exercise": "squat",
        "ground_truth": seq_gt.name,
        "query_seqs": [],
        "avg_angle_dif": {},
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
    results = {
        "exercise": "squat",
        "ground_truth": seq_gt.name,
        "query_seqs": [],
        "avg_angle_difs": {},
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
            seq_gt_angles = seq_gt.joint_angles[:, angle[0], angle[1].value]
            seq_query_angles = seq.joint_angles[:, angle[0], angle[1].value]
            avg_angle = np.mean(seq_query_angles)
            gt_avg_angle = np.mean(seq_gt_angles)
            avg_angle_dif = abs(gt_avg_angle - avg_angle)
            path, dist = ts.dtw_path(seq_gt_angles, seq_query_angles)
            for key, value in seq.body_parts.items():
                if value == angle[0]:
                    if key not in results["avg_angle_difs"].keys() and key not in results["distances"].keys():
                        results["avg_angle_difs"][key] = [avg_angle_dif]
                        results["distances"][key] = [dist]
                    else:
                        results["avg_angle_difs"][key].append(avg_angle_dif)
                        results["distances"][key].append(dist)

    with open(f"data/evaluation/rating/similarities/results_avg_difs.json", 'w') as outfile:
        json.dump(results, outfile)


# get_avg_angle_dif_and_dtw_dist()

def get_minmax_angle_mean_dif_and_dtw_dist():
    results = {
        "exercise": "squat",
        "ground_truth": seq_gt.name,
        "query_seqs": [],
        "minmax_angle_mean_dif": {},
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
    results = {
        "exercise": "squat",
        "ground_truth": seq_gt.name,
        "query_seqs": [],
        "minmax_angle_mean_dif": {},
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
            seq_gt_angles = seq_gt.joint_angles[:, angle[0], angle[1].value]
            seq_query_angles = seq.joint_angles[:, angle[0], angle[1].value]
            minmax_mean_angle = np.mean([min(seq_query_angles), max(seq_query_angles)])
            gt_minmax_mean_angle = np.mean([min(seq_gt_angles), max(seq_gt_angles)])
            minmax_mean_angle_dif = abs(gt_minmax_mean_angle - minmax_mean_angle)
            path, dist = ts.dtw_path(seq_gt_angles, seq_query_angles)
            for key, value in seq.body_parts.items():
                if value == angle[0]:
                    if key not in results["minmax_angle_mean_dif"].keys() and key not in results["distances"].keys():
                        results["minmax_angle_mean_dif"][key] = [minmax_mean_angle_dif]
                        results["distances"][key] = [dist]
                    else:
                        results["minmax_angle_mean_dif"][key].append(minmax_mean_angle_dif)
                        results["distances"][key].append(dist)

    with open(f"data/evaluation/rating/similarities/results_minmax_angle_mean_dif.json", 'w') as outfile:
        json.dump(results, outfile)


# get_minmax_angle_mean_dif_and_dtw_dist()


def get_max_angle_dif_and_dtw_dist():
    results = {
        "exercise": "squat",
        "ground_truth": seq_gt.name,
        "query_seqs": [],
        "max_angle_dif": {},
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
    results = {
        "exercise": "squat",
        "ground_truth": seq_gt.name,
        "query_seqs": [],
        "max_angle_dif": {},
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
            seq_gt_angles = seq_gt.joint_angles[:, angle[0], angle[1].value]
            seq_query_angles = seq.joint_angles[:, angle[0], angle[1].value]
            max_angle = max(seq_query_angles)
            gt_max_angle = max(seq_gt_angles)
            max_angle_dif = abs(gt_max_angle - max_angle)
            path, dist = ts.dtw_path(seq_gt_angles, seq_query_angles)
            for key, value in seq.body_parts.items():
                if value == angle[0]:
                    if key not in results["max_angle_dif"].keys() and key not in results["distances"].keys():
                        results["max_angle_dif"][key] = [max_angle_dif]
                        results["distances"][key] = [dist]
                    else:
                        results["max_angle_dif"][key].append(max_angle_dif)
                        results["distances"][key].append(dist)

    with open(f"data/evaluation/rating/similarities/results_max_angle_dif.json", 'w') as outfile:
        json.dump(results, outfile)


# get_max_angle_dif_and_dtw_dist()


def plot_correlation_tf_angles(bp: str):
    with open(f"data/evaluation/rating/similarities/results.json", 'r') as result_file:
        serialised_json_result = json.load(result_file)
        distances = serialised_json_result["distances"][bp]
        tf_angle_difs = serialised_json_result["tf_angle_dif"][bp]
        seq_names = serialised_json_result["query_seqs"]

        np_distances = np.array(distances)
        np_tf_angle_difs = np.array(tf_angle_difs)
        correlation_coef = np.corrcoef(np_distances, np_tf_angle_difs)
        print(correlation_coef)
        sns.set_style("ticks")
        # sns.set_context("paper")

        plt.figure(figsize=(6, 3))
        fig = sns.regplot(x=distances, y=tf_angle_difs)
        fig.set_title(f"Correlation between DTW Distance and Turning Frame Angle Difference\nAngle: {bp} Flexion/Extension\nGround-Truth: {seq_gt.name}\nCorrelation Coefficient: {'%.3f' % correlation_coef[0][1]}")
        fig.set(xlabel="Distances", ylabel="Turning Frame Angles")
        # plt.savefig(f"rating_similarities_{bp}.png",
        #             bbox_inches="tight",
        #             dpi=300)
        plt.show()


# plot_correlation_tf_angles("LeftHip")


def plot_correlation_avg_angles(bp: str):
    with open(f"data/evaluation/rating/similarities/results_avg_difs.json", 'r') as result_file:
        serialised_json_result = json.load(result_file)
        distances = serialised_json_result["distances"][bp]
        avg_angle_difs = serialised_json_result["avg_angle_difs"][bp]
        seq_names = serialised_json_result["query_seqs"]

        np_distances = np.array(distances)
        bp_avg_angle_difs = np.array(avg_angle_difs)
        correlation_coef = np.corrcoef(np_distances, bp_avg_angle_difs)
        print(correlation_coef)
        sns.set_style("ticks")
        # sns.set_context("paper")

        plt.figure(figsize=(6, 3))
        fig = sns.regplot(x=distances, y=bp_avg_angle_difs)
        fig.set_title(f"Correlation between DTW Distance and Average Angle Difference\nAngle: {bp} Flexion/Extension\nGround-Truth: {seq_gt.name}\nCorrelation Coefficient: {'%.3f' % correlation_coef[0][1]}")
        fig.set(xlabel="Distances", ylabel="Turning Frame Angles")
        plt.savefig(f"rating_similarities_avg_angle_{bp}.png",
                    bbox_inches="tight",
                    dpi=300)
        plt.show()


# plot_correlation_avg_angles("RightHip")


def plot_correlation_minmax_mean_angles(bp: str):
    with open(f"data/evaluation/rating/similarities/results_minmax_angle_mean_dif.json", 'r') as result_file:
        serialised_json_result = json.load(result_file)
        distances = serialised_json_result["distances"][bp]
        minmax_angle_difs = serialised_json_result["minmax_angle_mean_dif"][bp]
        seq_names = serialised_json_result["query_seqs"]

        np_distances = np.array(distances)
        bp_minmax_angle_difs = np.array(minmax_angle_difs)
        correlation_coef = np.corrcoef(np_distances, bp_minmax_angle_difs)
        print(correlation_coef)
        sns.set_style("ticks")
        # sns.set_context("paper")

        plt.figure(figsize=(6, 3))
        fig = sns.regplot(x=distances, y=bp_minmax_angle_difs)
        fig.set_title(f"Correlation between DTW Distance and Min/Max Angle Difference\nAngle: {bp} Flexion/Extension\nGround-Truth: {seq_gt.name}\nCorrelation Coefficient: {'%.3f' % correlation_coef[0][1]}")
        fig.set(xlabel="Distances", ylabel="Turning Frame Angles")
        plt.savefig(f"rating_similarities_minmax_angle_{bp}.png",
                    bbox_inches="tight",
                    dpi=300)
        plt.show()


# plot_correlation_minmax_mean_angles("RightHip")

def plot_correlation_max_angles(bp: str):
    with open(f"data/evaluation/rating/similarities/results_max_angle_dif.json", 'r') as result_file:
        serialised_json_result = json.load(result_file)
        distances = serialised_json_result["distances"][bp]
        max_angle_difs = serialised_json_result["max_angle_dif"][bp]
        seq_names = serialised_json_result["query_seqs"]

        np_distances = np.array(distances)
        bp_max_angle_difs = np.array(max_angle_difs)
        correlation_coef = np.corrcoef(np_distances, bp_max_angle_difs)
        print(correlation_coef)
        sns.set_style("ticks")
        # sns.set_context("paper")

        plt.figure(figsize=(6, 3))
        fig = sns.regplot(x=distances, y=bp_max_angle_difs)
        fig.set_title(f"Correlation between DTW Distance and Max Angle Difference\nAngle: {bp} Flexion/Extension\nGround-Truth: {seq_gt.name}\nCorrelation Coefficient: {'%.3f' % correlation_coef[0][1]}")
        fig.set(xlabel="DTW Distance", ylabel="Max Angle Difference")
        plt.savefig(f"rating_similarities_max_angle_{bp}.png",
                    bbox_inches="tight",
                    dpi=300)


plot_correlation_max_angles("RightHip")
plot_correlation_max_angles("LeftHip")
plot_correlation_max_angles("LeftKnee")
plot_correlation_max_angles("RightKnee")


def plot_sequences(bp: str):

    with open(f"data/evaluation/rating/similarities/results.json", 'r') as result_file:
        serialised_json_result = json.load(result_file)
        distances = serialised_json_result["distances"][bp]
        tf_angle_difs = serialised_json_result["tf_angle_dif"][bp]
        seq_names = serialised_json_result["query_seqs"]

    seq_name = seq_names[distances.index(max(distances))]
    seq_q = mocap_poseprocessor.load(list(Path("data/evaluation/rating/similarities/single_squats_query/").rglob(f"{seq_name}"))[0], seq_name)

    sns.set_style("ticks")

    # sns.set_context("paper")
    fig = plt.figure(figsize=(18, 5))
    fig.subplots_adjust(top=0.8)
    ax = plt.subplot(111)

    # Major ticks every 20, minor ticks every 5
    angle_major_ticks = np.arange(-180, 180, 20)
    angle_minor_ticks = np.arange(-180, 180, 10)
    frame_major_ticks = np.arange(-100, max(len(seq_gt), len(seq_q)) + 100, 20)
    frame_minor_ticks = np.arange(-100, max(len(seq_gt), len(seq_q)))

    ax.set_xticks(frame_major_ticks)
    ax.set_xticks(frame_minor_ticks, minor=True)
    ax.set_yticks(angle_major_ticks)
    ax.set_yticks(angle_minor_ticks, minor=True)

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.3)
    ax.grid(which='major', alpha=0.85)
    ax.tick_params(which='both', direction='out')

    plt.plot(range(0, len(seq_gt)),
             seq_gt.joint_angles[:, seq_gt.body_parts[bp], AngleTypes.FLEX_EX.value],
             zorder=1,
             linewidth="2.0",
             label="Ground-Truth")

    plt.plot(range(0, len(seq_q)),
             seq_q.joint_angles[:, seq_q.body_parts[bp], AngleTypes.FLEX_EX.value],
             zorder=1,
             linewidth="2.0",
             label="Query")

    EE = ExerciseEvaluator(squat, seq_gt)
    tf_gt = EE.find_iteration_keypoints()[0][1]
    EE.set_sequence(seq_q)
    tf_q = EE.find_iteration_keypoints()[0][1]

    plt.scatter(tf_gt, seq_gt.joint_angles[tf_gt, seq_gt.body_parts[bp], AngleTypes.FLEX_EX.value], marker="x", color="green", zorder=2, label="Turning Frame Ground-Truth")
    plt.scatter(tf_q, seq_gt.joint_angles[tf_q, seq_gt.body_parts[bp], AngleTypes.FLEX_EX.value], marker="x", color="red", zorder=2, label="Turning Frame Query")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1.02), fontsize="small")
    plt.xlabel("Frame")
    plt.ylabel("Angle")
    plt.show()


# plot_sequences("LeftHip")
