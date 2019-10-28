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
    for seq_angles in seqs_angles:
        distances.append(ts.dtw_path(ground_truth_angles, seq_angles)[1])
    return distances


def get_sequence_iterations(folder_path: str, ex: Exercise):
    EE = None
    # Load sequence files
    filenames = list(Path(folder_path).rglob("*.json"))
    sequences = []
    for file in filenames:
        sequences.append(mocap_poseprocessor.load(file, name=str(file)))

    # Split them into single iterations and expand the name for identification
    sequences_single_iterations = []
    for s in sequences:
        if EE == None:
            EE = ExerciseEvaluator(ex, s)
        else:
            EE.set_sequence(s)
        iterations = EE.find_iteration_keypoints(plot=True)
        for idx, iteration in enumerate(iterations):
            start, turn, end = iteration
            sequence_iteration = s[start:end]
            sequence_iteration.name += f"--iteration-{idx}"
            sequences_single_iterations.append(sequence_iteration)
    return sequences_single_iterations


# Get PoseProcessor instance for MOCAP sequences
mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)

squat = exercise_loader.load('data/exercises/squat.json')
overheadpress = exercise_loader.load('data/exercises/overhead-press.json')
biceps_curl_left = exercise_loader.load('data/exercises/biceps-curl-left.json')
biceps_curl_right = exercise_loader.load('data/exercises/biceps-curl-right.json')
lungeleft = exercise_loader.load('data/exercises/lunge-left.json')
lungeright = exercise_loader.load('data/exercises/lunge-right.json')

sequences_single_iterations = []
# sequences_single_iterations += get_sequence_iterations("data/sequences/dennis/kniebeuge", squat)
# sequences_single_iterations += get_sequence_iterations("data/sequences/levente/kniebeuge", squat)
# sequences_single_iterations += get_sequence_iterations("data/sequences/philippe/kniebeuge", squat)
sequences_single_iterations += get_sequence_iterations("data/sequences/191024 Tracking/multi/biceps_curl_left/user-1", biceps_curl_left)
sequences_single_iterations += get_sequence_iterations("data/sequences/191024 Tracking/multi/biceps_curl_right/user-1", biceps_curl_right)
# sequences_single_iterations += get_sequence_iterations("data/sequences/philippe/overhead-press", overheadpress)
# sequences_single_iterations += get_sequence_iterations("data/sequences/dennis/wechsel-lunges", lungeleft)
# sequences_single_iterations += get_sequence_iterations("data/sequences/levente/wechsel-lunges", lungeleft)
# sequences_single_iterations += get_sequence_iterations("data/sequences/philippe/wechsel-lunges", lungeleft)

# filenames = list(Path("data/sequences/dennis/lunge-left").rglob("complete-session.json"))
# for file in filenames:
#     print(f"Loading file: {file}")
#     sequences_single_iterations.append(mocap_poseprocessor.load(file, name=str(file)))
# filenames = list(Path("data/sequences/levente/lunge-right").rglob("complete-session.json"))
# for file in filenames:
#     print(f"Loading file: {file}")
#     sequences_single_iterations.append(mocap_poseprocessor.load(file, name=str(file)))
# filenames = list(Path("data/sequences/philippe/lunge-left").rglob("complete-session.json"))
# for file in filenames:
#     print(f"Loading file: {file}")
#     sequences_single_iterations.append(mocap_poseprocessor.load(file, name=str(file)))
# filenames = list(Path("data/sequences/dennis/lunge-right").rglob("complete-session.json"))
# for file in filenames:
#     print(f"Loading file: {file}")
#     sequences_single_iterations.append(mocap_poseprocessor.load(file, name=str(file)))
# filenames = list(Path("data/sequences/levente/lunge-left").rglob("complete-session.json"))
# for file in filenames:
#     print(f"Loading file: {file}")
#     sequences_single_iterations.append(mocap_poseprocessor.load(file, name=str(file)))
# filenames = list(Path("data/sequences/philippe/lunge-right").rglob("complete-session.json"))
# for file in filenames:
#     print(f"Loading file: {file}")
#     sequences_single_iterations.append(mocap_poseprocessor.load(file, name=str(file)))

sequences_dtw_angles = []
for s in sequences_single_iterations:
    print(f"Restructuring angles for DTW: {s.name}")
    sequences_dtw_angles.append(get_dtw_angles_mocap(s))

dtw_distances = sorted(get_distances_dtw(sequences_dtw_angles[0], sequences_dtw_angles))
for i in range(len(dtw_distances)):
    print(f"{[dtw_distances[i]]} {sequences_single_iterations[i].name}")
