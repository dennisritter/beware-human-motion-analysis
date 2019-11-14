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

mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)
squat = exercise_loader.load('data/exercises/squat.json')
overhead_press = exercise_loader.load('data/exercises/overhead-press.json')
biceps_curl_left = exercise_loader.load('data/exercises/biceps-curl-left.json')
biceps_curl_right = exercise_loader.load('data/exercises/biceps-curl-right.json')
knee_lift_left = exercise_loader.load('data/exercises/knee-lift-left.json')
knee_lift_right = exercise_loader.load('data/exercises/knee-lift-right.json')


def get_subseq_result(exercise: Exercise, exercise_name: str):
    seqs = []
    EE = None

    # Get all multi iteration sequences of a squat
    filenames = list(Path("data/sequences/191024_tracking/multi/" + exercise_name + "/").rglob("*.json"))
    for filename in filenames:
        print(f"Loading Sequence file: {filename}")
        sequence = mocap_poseprocessor.load(filename, str(filename).split('\\')[-1])
        seqs.append(sequence)

    result = {
        "exercise": exercise_name,
        "params": {
            "savgol_window": 51,
            "savgol_order": 3,
            "argrelextrema_order": 20,
            "start_end_keyframe_tolerance": "dmean",
            "extrema_group_window_size": 30,
        },
        "n_sequences": len(seqs),
        "n_iterations_found": 0,
        "results": [None]*len(seqs),
    }

    seq_results = [None]*len(seqs)
    for i, seq in enumerate(seqs):
        if EE is None:
            EE = ExerciseEvaluator(exercise, seq)
        else:
            EE.set_sequence(seq)
        seq_results[i] = {
            "name": seq.name,
            "length": len(seq),
            "iterations": EE.find_iteration_keypoints().tolist(),
        }
        result["n_iterations_found"] += len(seq_results[i]["iterations"])
        print("----------------------------------------")
        print(f"{seq_results[i]['name']}")
        print(f"Sequence length: {seq_results[i]['length']}")
        print(f"{len(seq_results[i]['iterations'])} iterations found: {seq_results[i]['iterations']}")

    result["results"] = seq_results

    result_filename = f"subseq_result_{result['exercise']}_{result['params']['savgol_window']}_{result['params']['savgol_order']}_{result['params']['argrelextrema_order']}_{result['params']['start_end_keyframe_tolerance']}_{result['params']['extrema_group_window_size']}.json"
    with open(f"data/evaluation/subsequencing/{result_filename}", 'w') as outfile:
        json.dump(result, outfile)


# get_subseq_result(squat, 'squat')
# get_subseq_result(overhead_press, 'overhead_press')
# get_subseq_result(biceps_curl_left, 'biceps_curl_left')
# get_subseq_result(biceps_curl_right, 'biceps_curl_right')
# get_subseq_result(knee_lift_left, 'knee_lift_left')
# get_subseq_result(knee_lift_right, 'knee_lift_right')


def plot_evaluation_savgol_window():
    results = []
    filenames = list(Path("data/evaluation/subsequencing/").rglob("*_3_10_dmean_30.json"))
    for filename in filenames:
        with open(filename, 'r') as f:
            result = json.load(f)
            results.append(result)

    groups = ['squat', 'overhead_press', 'biceps_curl_left', 'biceps_curl_right', 'knee_lift_left', 'knee_lift_right']
    categorical = ['Expected iterations', 21, 51, 101]
    categorical_label = ['Expected iterations', f"savgol_window: {21}", f"savgol_window: {51}", f"savgol_window: {101}"]
    # colors = ['green', 'red', 'blue', 'orange']
    numerical = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    for result in results:
        expected_iterations = result["n_sequences"] * 10
        identified_iterations = result["n_iterations_found"]
        numerical[categorical.index(result["params"]["savgol_window"])][groups.index(result["exercise"])] = identified_iterations
        numerical[0][groups.index(result["exercise"])] = expected_iterations

    number_groups = len(categorical)
    bin_width = 1.0 / (number_groups + 1)
    fig, ax = plt.subplots(figsize=(15, 3))
    for i in range(number_groups):
        bars = ax.bar(x=np.arange(6) + i * bin_width, height=numerical[i], width=bin_width, align='center')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., 3, '%d' % int(bar.get_height()), ha='center', va="bottom", color='white')
        ax.set_xticks(np.arange(len(groups)) + number_groups/(3*(number_groups+1)))  # number_groups/(2*(number_groups+1)): offset of xticklabelax.set_xticklabels(categorical)
        ax.set_xticklabels(groups)
        ax.legend(categorical_label, facecolor='w', loc='upper left', bbox_to_anchor=(1, 1.02), fontsize="small")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height * 0.9])
    fig.suptitle(f"Subsequencing results using different Savitzky Golay window sizes\nsavgol_order: 3  |  argrelextrema_order: 10  |  extrema_group_window_size: 30", fontsize=13)
    plt.xlabel("Exercise")
    plt.ylabel("Iterations")
    plt.savefig("subsequencing_savgol_windows.png",
                bbox_inches="tight",
                dpi=300)
    plt.show()


# plot_evaluation_savgol_window()


def plot_evaluation_extrema_grouping_window():
    results = []
    filenames = list(Path("data/evaluation/subsequencing/").rglob("*51_3_10_dmean_*.json"))
    for filename in filenames:
        with open(filename, 'r') as f:
            result = json.load(f)
            results.append(result)

    groups = ['squat', 'overhead_press', 'biceps_curl_left', 'biceps_curl_right', 'knee_lift_left', 'knee_lift_right']
    categorical = ['Expected iterations', 10, 30, 50]
    categorical_label = ['Expected iterations', f"extrema_group_window: {10}", f"extrema_group_window: {30}", f"extrema_group_window: {50}"]
    # colors = ['green', 'red', 'blue', 'orange']
    numerical = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    for result in results:
        expected_iterations = result["n_sequences"] * 10
        identified_iterations = result["n_iterations_found"]
        numerical[categorical.index(result["params"]["extrema_group_window_size"])][groups.index(result["exercise"])] = identified_iterations
        numerical[0][groups.index(result["exercise"])] = expected_iterations

    number_groups = len(categorical)
    bin_width = 1.0 / (number_groups + 1)
    fig, ax = plt.subplots(figsize=(15, 3))
    for i in range(number_groups):
        bars = ax.bar(x=np.arange(6) + i * bin_width, height=numerical[i], width=bin_width, align='center')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., 3, '%d' % int(bar.get_height()), ha='center', va="bottom", color='white')
        ax.set_xticks(np.arange(len(groups)) + number_groups/(3*(number_groups+1)))  # number_groups/(2*(number_groups+1)): offset of xticklabelax.set_xticklabels(categorical)
        ax.set_xticklabels(groups)
        ax.legend(categorical_label, facecolor='w', loc='upper left', bbox_to_anchor=(1, 1.02), fontsize="small")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height * 0.9])
    fig.suptitle(f"Subsequencing results using different extrema grouping window sizes\nsavgol_window: 51  |  savgol_order: 3  |  argrelextrema_order: 10", fontsize=13)
    plt.xlabel("Exercise")
    plt.ylabel("Iterations")
    plt.savefig("subsequencing_extrema_grouping_window.png",
                bbox_inches="tight",
                dpi=300)
    plt.show()


# plot_evaluation_extrema_grouping_window()

def plot_evaluation_savgol_order():
    results = []
    filenames = list(Path("data/evaluation/subsequencing/").rglob("*51_*_10_dmean_30.json"))
    for filename in filenames:
        with open(filename, 'r') as f:
            result = json.load(f)
            results.append(result)

    groups = ['squat', 'overhead_press', 'biceps_curl_left', 'biceps_curl_right', 'knee_lift_left', 'knee_lift_right']
    categorical = ['Expected iterations', 2, 3, 4]
    categorical_label = ['Expected iterations', f"savgol_order: {2}", f"savgol_order: {3}", f"savgol_order: {4}"]
    # colors = ['green', 'red', 'blue', 'orange']
    numerical = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    for result in results:
        expected_iterations = result["n_sequences"] * 10
        identified_iterations = result["n_iterations_found"]
        numerical[categorical.index(result["params"]["savgol_order"])][groups.index(result["exercise"])] = identified_iterations
        numerical[0][groups.index(result["exercise"])] = expected_iterations

    number_groups = len(categorical)
    bin_width = 1.0 / (number_groups + 1)
    fig, ax = plt.subplots(figsize=(15, 3))
    for i in range(number_groups):
        bars = ax.bar(x=np.arange(6) + i * bin_width, height=numerical[i], width=bin_width, align='center')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., 3, '%d' % int(bar.get_height()), ha='center', va="bottom", color='white')
        ax.set_xticks(np.arange(len(groups)) + number_groups/(3*(number_groups+1)))  # number_groups/(2*(number_groups+1)): offset of xticklabelax.set_xticklabels(categorical)
        ax.set_xticklabels(groups)
        ax.legend(categorical_label, facecolor='w', loc='upper left', bbox_to_anchor=(1, 1.02), fontsize="small")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height * 0.9])
    fig.suptitle(f"Subsequencing results using different Savitky Golay Filter order values\nsavgol_window: 51  |  argrelextrema_order: 10  |  extrema_group_window_size: 30", fontsize=13)
    plt.xlabel("Exercise")
    plt.ylabel("Iterations")
    plt.savefig("subsequencing_savgol_order.png",
                bbox_inches="tight",
                dpi=300)
    plt.show()


# plot_evaluation_savgol_order()


def plot_evaluation_argrelextrema_order():
    results = []
    filenames = list(Path("data/evaluation/subsequencing/").rglob("*51_3_*_dmean_30.json"))
    for filename in filenames:
        with open(filename, 'r') as f:
            result = json.load(f)
            results.append(result)

    groups = ['squat', 'overhead_press', 'biceps_curl_left', 'biceps_curl_right', 'knee_lift_left', 'knee_lift_right']
    categorical = ['Expected iterations', 5, 10, 20]
    categorical_label = ['Expected iterations', f"argrelextrema_order: {5}", f"argrelextrema_order: {10}", f"argrelextrema_order: {20}"]
    # colors = ['green', 'red', 'blue', 'orange']
    numerical = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    for result in results:
        expected_iterations = result["n_sequences"] * 10
        identified_iterations = result["n_iterations_found"]
        numerical[categorical.index(result["params"]["argrelextrema_order"])][groups.index(result["exercise"])] = identified_iterations
        numerical[0][groups.index(result["exercise"])] = expected_iterations

    number_groups = len(categorical)
    bin_width = 1.0 / (number_groups + 1)
    fig, ax = plt.subplots(figsize=(15, 3))
    for i in range(number_groups):
        bars = ax.bar(x=np.arange(6) + i * bin_width, height=numerical[i], width=bin_width, align='center')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., 3, '%d' % int(bar.get_height()), ha='center', va="bottom", color='white')
        ax.set_xticks(np.arange(len(groups)) + number_groups/(3*(number_groups+1)))  # number_groups/(2*(number_groups+1)): offset of xticklabelax.set_xticklabels(categorical)
        ax.set_xticklabels(groups)
        ax.legend(categorical_label, facecolor='w', loc='upper left', bbox_to_anchor=(1, 1.02), fontsize="small")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height * 0.9])
    fig.suptitle(f"Subsequencing results using different argrelextrema order values\nsavgol_window: 51  |  savgol_order: 3  |  extrema_group_window_size: 30", fontsize=13)
    plt.xlabel("Exercise")
    plt.ylabel("Iterations")
    plt.savefig("subsequencing_argrelextrema_order.png",
                bbox_inches="tight",
                dpi=300)
    plt.show()


plot_evaluation_argrelextrema_order()
