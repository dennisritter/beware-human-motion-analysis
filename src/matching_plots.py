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
import matplotlib.pyplot as plt
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


# Calculates dtw_distance of sequences in seqs_angles compared to ground_truth_angles
# Returns a list of floats
def get_distances_dtw(ground_truth_angles, seqs_angles):
    distances = []
    path = []
    for seq_angles in seqs_angles:
        path.append(ts.dtw_path(ground_truth_angles, seq_angles)[0])
        distances.append(ts.dtw_path(ground_truth_angles, seq_angles)[1])

    return path, distances


# Get PoseProcessor instance for MOCAP sequences
mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)
biceps_curl_left = exercise_loader.load('data/exercises/biceps-curl-left.json')
biceps_curl_right = exercise_loader.load('data/exercises/biceps-curl-right.json')

seq1 = mocap_poseprocessor.load(
    'data/sequences/191024_tracking/single/biceps_curl_right/user-2/191024__single__biceps_curl_right__user-2__0.json',
    '191024__single__biceps_curl_right__user-2__0')
seq2 = mocap_poseprocessor.load(
    'data/sequences/191024_tracking/single/biceps_curl_right/user-2/191024__single__biceps_curl_right__user-2__1.json',
    '191024__single__biceps_curl_right__user-3__0')

sequences_single_iterations = [seq1, seq2]

sequences_dtw_angles = []
for s in sequences_single_iterations:
    print(f"Restructuring angles for DTW: {s.name}")
    sequences_dtw_angles.append(get_dtw_angles_mocap(s))

dtw_distances = sorted(get_distances_dtw(sequences_dtw_angles[0], sequences_dtw_angles)[1])
for i in range(len(dtw_distances)):
    print(f"{[dtw_distances[i]]} {sequences_single_iterations[i].name}")


bp = seq1.body_parts
path = get_distances_dtw(sequences_dtw_angles[0], sequences_dtw_angles)[0][1]
print(path)
sns.set_style("ticks")
# sns.set_context("paper")
fig = plt.figure(figsize=(18, 5))
fig.subplots_adjust(top=0.8)
ax = plt.subplot(111)

# Major ticks every 20, minor ticks every 5
angle_major_ticks = np.arange(-180, 180, 20)
angle_minor_ticks = np.arange(-180, 180, 10)
frame_major_ticks = np.arange(-100, max(len(seq1), len(seq2)) + 100, 20)
frame_minor_ticks = np.arange(-100, max(len(seq1), len(seq2)))

ax.set_xticks(frame_major_ticks)
ax.set_xticks(frame_minor_ticks, minor=True)
ax.set_yticks(angle_major_ticks)
ax.set_yticks(angle_minor_ticks, minor=True)

# Or if you want different settings for the grids:
ax.grid(which='minor', alpha=0.3)
ax.grid(which='major', alpha=0.85)
ax.tick_params(which='both', direction='out')

plt.plot(range(0, len(seq1)),
         seq1.joint_angles[:, bp["RightElbow"], AngleTypes.FLEX_EX.value],
         zorder=1,
         linewidth="2.0",
         label="Biceps Curl Seq 1")

plt.plot(range(0, len(seq2)),
         seq2.joint_angles[:, bp["RightElbow"], AngleTypes.FLEX_EX.value],
         zorder=1,
         linewidth="2.0",
         label="Biceps Curl Seq 2")
for p in path:
    print(p[0], p[1])
    plt.plot([p[0], p[1]], [seq1.joint_angles[p[0], bp["RightElbow"], AngleTypes.FLEX_EX.value], seq2.joint_angles[p[1], bp["RightElbow"], AngleTypes.FLEX_EX.value]], color="black", linewidth=1)

# plt.scatter(confirmed_start_frames, np.full(confirmed_start_frames.shape, angles_savgol_all_bps.min() - 10), color='r', marker="v", s=20, zorder=3, label="Removed Turning Frame")
# plt.scatter(confirmed_turning_frames, np.full(confirmed_turning_frames.shape, angles_savgol_all_bps.max() + 10), color='r', marker="^", s=20, zorder=3, label="Removed Start/End Frame")
# # # Iterations
# plt.scatter(iterations[:, 1], np.full((len(iterations), ), angles_savgol_all_bps.max() + 10), color="g", zorder=4, marker="^", s=50, label="Turning Frame")
# plt.scatter(iterations[:, 0:3:2], np.full((len(iterations), 2), angles_savgol_all_bps.min() - 10), color="g", zorder=4, marker="v", s=50, label="Start/End Frame")

# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
plt.legend(loc='upper left', bbox_to_anchor=(1, 1.02), fontsize="small")
plt.xlabel("Frame")
plt.ylabel("Right Elbow Flexion Angle")
fig.suptitle(f"Sequence 1: {seq1.name}.json \nSequence 2: {seq2.name}.json \nTotal distance: {max(dtw_distances)}")
plt.savefig("matching.png",
            bbox_inches="tight",
            dpi=300)
plt.show()
