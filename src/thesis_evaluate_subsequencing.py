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
            "argrelextrema_order": 10,
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


get_subseq_result(squat, 'squat')
get_subseq_result(overhead_press, 'overhead_press')
get_subseq_result(biceps_curl_left, 'biceps_curl_left')
get_subseq_result(biceps_curl_right, 'biceps_curl_right')
get_subseq_result(knee_lift_left, 'knee_lift_left')
get_subseq_result(knee_lift_right, 'knee_lift_right')


# if plot:
#             sns.set_style("ticks")
#             # sns.set_context("paper")
#             fig = plt.figure(figsize=(12, 5))
#             ax = plt.subplot(111)

#             # Major ticks every 20, minor ticks every 5
#             angle_major_ticks = np.arange(-180, 180, 20)
#             angle_minor_ticks = np.arange(-180, 180, 10)
#             frame_major_ticks = np.arange(-100, len(seq) + 100, 100)
#             frame_minor_ticks = np.arange(-100, len(seq) + 100, 20)

#             ax.set_xticks(frame_major_ticks)
#             ax.set_xticks(frame_minor_ticks, minor=True)
#             ax.set_yticks(angle_major_ticks)
#             ax.set_yticks(angle_minor_ticks, minor=True)

#             # Or if you want different settings for the grids:
#             ax.grid(which='minor', alpha=0.3)
#             ax.grid(which='major', alpha=0.85)
#             ax.tick_params(which='both', direction='out')

#             for prio_idx in range(len(angles_savgol_all_bps)):
#                 savgol_angles_label, min_label, max_label = (None, None, None) if prio_idx != 0 else ("Body part angles", "Minimum", "Maximum")
#                 maxima = maxima_all_bps[prio_idx].astype(int)
#                 minima = minima_all_bps[prio_idx].astype(int)
#                 # Savgol angles
#                 sns.color_palette()
#                 plt.plot(range(0, len(angles_savgol_all_bps[prio_idx])),
#                          angles_savgol_all_bps[prio_idx],
#                          zorder=1,
#                          linewidth="2.0",
#                          label=self.get_label(angles_legend[prio_idx]))
#                 # Minima/Maxima
#                 plt.scatter(maxima, angles_savgol_all_bps[prio_idx][maxima], color='black', marker="^", zorder=2, facecolors='b', label=max_label)
#                 plt.scatter(minima, angles_savgol_all_bps[prio_idx][minima], color='black', marker="v", zorder=2, facecolors='b', label=min_label)
#             # # Confirmed Extrema
#             # plt.scatter(confirmed_start_frames, np.full(confirmed_start_frames.shape, angles_savgol_all_bps.min() - 10), color='r', marker="v", s=20, zorder=3, label="Removed Turning Frame")
#             # plt.scatter(confirmed_turning_frames, np.full(confirmed_turning_frames.shape, angles_savgol_all_bps.max() + 10), color='r', marker="^", s=20, zorder=3, label="Removed Start/End Frame")
#             plt.scatter(confirmed_start_frames, np.full(confirmed_start_frames.shape, angles_savgol_all_bps.min() - 10), color='r', marker="v", s=20, zorder=3, label="Removed Turning Frame")
#             plt.scatter(confirmed_turning_frames, np.full(confirmed_turning_frames.shape, angles_savgol_all_bps.max() + 10), color='r', marker="^", s=20, zorder=3, label="Removed Start/End Frame")
#             # # Iterations
#             plt.scatter(iterations[:, 1], np.full((len(iterations), ), angles_savgol_all_bps.max() + 10), color="g", zorder=4, marker="^", s=50, label="Turning Frame")
#             plt.scatter(iterations[:, 0:3:2], np.full((len(iterations), 2), angles_savgol_all_bps.min() - 10), color="g", zorder=4, marker="v", s=50, label="Start/End Frame")

#             box = ax.get_position()
#             ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
#             plt.legend(loc='upper left', bbox_to_anchor=(1, 1.02), fontsize="small")
#             plt.xlabel("Frame")
#             plt.ylabel("Angle")
#             # plt.savefig("subsequencing_extrema_distance_filter.png",
#             #             bbox_inches="tight",
#             #             dpi=300)
#             plt.show()
