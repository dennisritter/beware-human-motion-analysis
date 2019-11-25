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


def get_closest_distances_dtw_result():
    with open(f"data/evaluation/matching/dtw_results.json", 'r') as result_file:
        serialised_json_result = json.load(result_file)
        results = serialised_json_result["payload"]
        exercises = serialised_json_result["exercises"]

    q_grouped_results = {}
    for result in results:
        distance = result["distance"]
        gt_name = result["ground_truth_name"]
        q_name = result["query_name"]
        if q_name not in q_grouped_results:
            q_grouped_results[q_name] = []
        q_grouped_results[q_name].append((gt_name, distance))
    with open(f"data/evaluation/matching/dtw_results_q_grouped.json", 'w') as outfile:
        json.dump(q_grouped_results, outfile)

    closest_distances = []
    for q_group in q_grouped_results.keys():
        closest_distance = None
        for q_distance in q_grouped_results[f"{q_group}"]:
            if closest_distance is None:
                closest_distance = q_distance
            if closest_distance[1] > q_distance[1]:
                closest_distance = q_distance
        closest_distance = [q_group, closest_distance[0], closest_distance[1]]
        closest_distances.append(closest_distance)
    with open(f"data/evaluation/matching/dtw_results_closest_distances.json", 'w') as outfile:
        json.dump({"result": closest_distances}, outfile)
    return closest_distances


def plot_matching_global():
    exercises = [
        "squat",
        "biceps_curl_left",
        "biceps_curl_right",
        "knee_lift_left",
        "knee_lift_right"
    ]
    closest_distances_result = get_closest_distances_dtw_result()
    correct_matches = 0
    incorrect_matches = 0
    for result in closest_distances_result:
        for ex in exercises:
            if ex in result[0] and ex in result[1]:
                correct_matches += 1
                break
            if ex in result[0] and ex not in result[1]:
                incorrect_matches += 1
                break

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = ['correct', 'incorrect']
    sizes = [correct_matches, incorrect_matches]
    print(sizes)
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=["#2ecc71", "#e74c3c"])
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    fig1.suptitle(f"Overall Matching Results for {correct_matches+incorrect_matches} Query Sequences")
    plt.savefig("matching_overall_result.png",
                bbox_inches="tight",
                dpi=300)
    plt.show()


def plot_matching_per_exercise():
    exercises = [
        "squat",
        "biceps_curl_left",
        "biceps_curl_right",
        "knee_lift_left",
        "knee_lift_right"
    ]
    closest_distances_result = get_closest_distances_dtw_result()
    squat = [0, 0]
    biceps_curl_left = [0, 0]
    biceps_curl_right = [0, 0]
    knee_lift_left = [0, 0]
    knee_lift_right = [0, 0]
    for result in closest_distances_result:
        for ex in exercises:
            if ex in result[0] and ex in result[1]:
                if ex == "squat":
                    squat[0] += 1
                if ex == "biceps_curl_left":
                    biceps_curl_left[0] += 1
                if ex == "biceps_curl_right":
                    biceps_curl_right[0] += 1
                if ex == "knee_lift_left":
                    knee_lift_left[0] += 1
                if ex == "knee_lift_right":
                    knee_lift_right[0] += 1
                break
            if ex in result[0] and ex not in result[1]:
                if ex == "squat":
                    squat[1] += 1
                if ex == "biceps_curl_left":
                    biceps_curl_left[1] += 1
                if ex == "biceps_curl_right":
                    biceps_curl_right[1] += 1
                if ex == "knee_lift_left":
                    knee_lift_left[1] += 1
                if ex == "knee_lift_right":
                    knee_lift_right[1] += 1
                break

    groups = ['squat', 'biceps_curl_left', 'biceps_curl_right', 'knee_lift_left', 'knee_lift_right']
    categorical = ['Correct', 'Incorrect']
    categorical_label = ['Correct', 'Incorrect']
    # colors = ['green', 'red', 'blue', 'orange']
    numerical = [
        [squat[0], biceps_curl_left[0], biceps_curl_right[0], knee_lift_left[0], knee_lift_right[0]],
        [squat[1], biceps_curl_left[1], biceps_curl_right[1], knee_lift_left[1], knee_lift_right[1]]
    ]
    number_groups = len(categorical)
    bin_width = 1.0 / (number_groups + 1)
    fig, ax = plt.subplots(figsize=(10, 3))
    for i in range(number_groups):
        if i == 0:
            bars = ax.bar(x=np.arange(5) + i * bin_width, height=numerical[i], width=bin_width, align='center', color="#2ecc71")
        if i == 1:
            bars = ax.bar(x=np.arange(5) + i * bin_width, height=numerical[i], width=bin_width, align='center', color="#e74c3c")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., 3, '%d' % int(bar.get_height()), ha='center', va="bottom", color='black')
        ax.set_xticks(np.arange(len(groups)) + number_groups/(3*(number_groups+1)))  # number_groups/(2*(number_groups+1)): offset of xticklabelax.set_xticklabels(categorical)
        ax.set_xticklabels(groups)
        ax.legend(categorical_label, facecolor='w', loc='upper left', bbox_to_anchor=(1, 1.02), fontsize="small")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height * 0.9])
    fig.suptitle(f"Matching Results for All Query Sequences Grouped by Exercises", fontsize=13)
    plt.xlabel("Exercise")
    plt.ylabel("Matches")
    plt.savefig("matching_exercise_result.png",
                bbox_inches="tight",
                dpi=300)
    plt.show()


# plot_matching_per_exercise()


def plot_distribution(exercise: str):
    exercises = [
        "squat",
        "biceps_curl_left",
        "biceps_curl_right",
        "knee_lift_left",
        "knee_lift_right"
    ]
    closest_distances_result = get_closest_distances_dtw_result()
    squat = 0
    biceps_curl_left = 0
    biceps_curl_right = 0
    knee_lift_left = 0
    knee_lift_right = 0
    for result in closest_distances_result:
        if exercise in result[0]:
            ex = result[1]
            if "squat" in ex:
                squat += 1
            if "biceps_curl_left" in ex:
                biceps_curl_left += 1
                print(result[0], result[2])
            if "biceps_curl_right" in ex:
                biceps_curl_right += 1
            if "knee_lift_left" in ex:
                knee_lift_left += 1
            if "knee_lift_right" in ex:
                knee_lift_right += 1
    groups = ['squat', 'biceps_curl_left', 'biceps_curl_right', 'knee_lift_left', 'knee_lift_right']
    categorical = ['Matches']
    categorical_label = ['Matches']
    # colors = ['green', 'red', 'blue', 'orange']
    numerical = [[squat, biceps_curl_left, biceps_curl_right, knee_lift_left, knee_lift_right]]
    number_groups = len(categorical)
    bin_width = 1.0 / (number_groups + 1)
    fig, ax = plt.subplots(figsize=(10, 3))
    for i in range(number_groups):
        if i == 0:
            bars = ax.bar(x=np.arange(5) + i * bin_width, height=numerical[i], width=bin_width, align='center', color=["#e74c3c", "#e74c3c", "#e74c3c", "#e74c3c", "#2ecc71"])
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., 3, '%d' % int(bar.get_height()), ha='center', va="bottom", color='black')
        ax.set_xticks(np.arange(len(groups)) + number_groups/(3*(number_groups+1)))  # number_groups/(2*(number_groups+1)): offset of xticklabelax.set_xticklabels(categorical)
        ax.set_xticklabels(groups)
        # ax.legend(categorical_label, facecolor='w', loc='upper left', bbox_to_anchor=(1, 1.02), fontsize="small")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height * 0.9])
    fig.suptitle(f"Matching Result Distribution for Right Knee Lift Exercise Query Sequences", fontsize=13)
    plt.xlabel("Exercise")
    plt.ylabel("Matches")
    plt.savefig("matching_klr_distribution.png",
                bbox_inches="tight",
                dpi=300)
    plt.show()


plot_distribution("knee_lift_right")


def plot_compare_seqs(exercise: str):
    exercises = [
        "squat",
        "biceps_curl_left",
        "biceps_curl_right",
        "knee_lift_left",
        "knee_lift_right"
    ]
    closest_distances_result = get_closest_distances_dtw_result()
    squat = 0
    biceps_curl_left = 0
    biceps_curl_right = 0
    knee_lift_left = 0
    knee_lift_right = 0
    for result in closest_distances_result:
        if exercise in result[0]:
            ex = result[1]
            if "squat" in ex:
                squat += 1
            if "biceps_curl_left" in ex:
                biceps_curl_left += 1
            if "biceps_curl_right" in ex:
                biceps_curl_right += 1
            if "knee_lift_left" in ex:
                knee_lift_left += 1
            if "knee_lift_right" in ex:
                knee_lift_right += 1
    groups = ['squat', 'biceps_curl_left', 'biceps_curl_right', 'knee_lift_left', 'knee_lift_right']
    categorical = ['Matches']
    categorical_label = ['Matches']
    # colors = ['green', 'red', 'blue', 'orange']
    numerical = [[squat, biceps_curl_left, biceps_curl_right, knee_lift_left, knee_lift_right]]
    number_groups = len(categorical)
    bin_width = 1.0 / (number_groups + 1)
    fig, ax = plt.subplots(figsize=(10, 3))
    for i in range(number_groups):
        if i == 0:
            bars = ax.bar(x=np.arange(5) + i * bin_width, height=numerical[i], width=bin_width, align='center', color=["#e74c3c", "#e74c3c", "#e74c3c", "#e74c3c", "#2ecc71"])
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., 3, '%d' % int(bar.get_height()), ha='center', va="bottom", color='black')
        ax.set_xticks(np.arange(len(groups)) + number_groups/(3*(number_groups+1)))  # number_groups/(2*(number_groups+1)): offset of xticklabelax.set_xticklabels(categorical)
        ax.set_xticklabels(groups)
        # ax.legend(categorical_label, facecolor='w', loc='upper left', bbox_to_anchor=(1, 1.02), fontsize="small")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height * 0.9])
    fig.suptitle(f"Matching Result Distribution for Right Knee Lift Exercise Query Sequences", fontsize=13)
    plt.xlabel("Exercise")
    plt.ylabel("Matches")
    plt.savefig("matching_klr_distribution.png",
                bbox_inches="tight",
                dpi=300)
    plt.show()
