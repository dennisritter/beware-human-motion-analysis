import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tslearn.metrics as ts
import seaborn as sns
import numpy as np
from hma.movement_analysis.Sequence import Sequence
from hma.movement_analysis.PoseProcessor import PoseProcessor
from hma.movement_analysis.PoseFormatEnum import PoseFormatEnum
from hma.movement_analysis.AngleTypes import AngleTypes
from hma.movement_analysis import angle_calculations as acm
from hma.movement_analysis import distance
import tslearn.metrics as ts


# Calculating joint angles for a MOCAP sequence and returning a 2D-list containing all angles for each frame in consecutive order
def get_minmax_angles_mocap(seq):
    bp = seq.body_parts
    minmax_angles = []
    minmax_angles.append([np.min(seq.joint_angles[:, bp["LeftShoulder"], AngleTypes.FLEX_EX.value]), np.max(seq.joint_angles[:, bp["LeftShoulder"], AngleTypes.FLEX_EX.value])])
    minmax_angles.append([np.min(seq.joint_angles[:, bp["LeftShoulder"], AngleTypes.AB_AD.value]), np.max(seq.joint_angles[:, bp["LeftShoulder"], AngleTypes.AB_AD.value])])
    minmax_angles.append([np.min(seq.joint_angles[:, bp["RightShoulder"], AngleTypes.FLEX_EX.value]), np.max(seq.joint_angles[:, bp["RightShoulder"], AngleTypes.FLEX_EX.value])])
    minmax_angles.append([np.min(seq.joint_angles[:, bp["RightShoulder"], AngleTypes.AB_AD.value]), np.max(seq.joint_angles[:, bp["RightShoulder"], AngleTypes.AB_AD.value])])
    minmax_angles.append([np.min(seq.joint_angles[:, bp["LeftHip"], AngleTypes.FLEX_EX.value]), np.max(seq.joint_angles[:, bp["LeftHip"], AngleTypes.FLEX_EX.value])])
    minmax_angles.append([np.min(seq.joint_angles[:, bp["LeftHip"], AngleTypes.AB_AD.value]), np.max(seq.joint_angles[:, bp["LeftHip"], AngleTypes.AB_AD.value])])
    minmax_angles.append([np.min(seq.joint_angles[:, bp["RightHip"], AngleTypes.FLEX_EX.value]), np.max(seq.joint_angles[:, bp["RightHip"], AngleTypes.FLEX_EX.value])])
    minmax_angles.append([np.min(seq.joint_angles[:, bp["RightHip"], AngleTypes.AB_AD.value]), np.max(seq.joint_angles[:, bp["RightHip"], AngleTypes.AB_AD.value])])
    minmax_angles.append([np.min(seq.joint_angles[:, bp["LeftElbow"], AngleTypes.FLEX_EX.value]), np.max(seq.joint_angles[:, bp["LeftElbow"], AngleTypes.FLEX_EX.value])])
    minmax_angles.append([np.min(seq.joint_angles[:, bp["RightElbow"], AngleTypes.FLEX_EX.value]), np.max(seq.joint_angles[:, bp["RightElbow"], AngleTypes.FLEX_EX.value])])
    minmax_angles.append([np.min(seq.joint_angles[:, bp["LeftKnee"], AngleTypes.FLEX_EX.value]), np.max(seq.joint_angles[:, bp["LeftKnee"], AngleTypes.FLEX_EX.value])])
    minmax_angles.append([np.min(seq.joint_angles[:, bp["RightKnee"], AngleTypes.FLEX_EX.value]), np.max(seq.joint_angles[:, bp["RightKnee"], AngleTypes.FLEX_EX.value])])
    return np.array(minmax_angles)


# Calculates dtw_distance of sequences in seqs_angles compared to ground_truth_angles
# Returns a list of floats
def get_distances_minmax(ground_truth_angles, seqs_angles):
    distances = []
    for seq_angles in seqs_angles:
        distances.append(distance.hausdorff(ground_truth_angles, seq_angles)[0])
    return distances


# Get PoseProcessor instance for MOCAP sequences
mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)

# Convert json sequence file to Sequence object and calculate joint angles for dtw comparison
seq_gt_angles = get_minmax_angles_mocap(mocap_poseprocessor.load('data/sequences/squat_3/complete-session.json', 'Squat_Ground_Truth'))
seq_1_angles = get_minmax_angles_mocap(mocap_poseprocessor.load('data/sequences/squat_1/complete-session.json', 'Squat 1'))
seq_2_angles = get_minmax_angles_mocap(mocap_poseprocessor.load('data/sequences/squat_2/complete-session.json', 'Squat 2'))
seq_3_angles = get_minmax_angles_mocap(mocap_poseprocessor.load('data/sequences/squat_4/complete-session.json', 'Squat 4'))
seq_4_angles = get_minmax_angles_mocap(mocap_poseprocessor.load('data/sequences/squat_false/complete-session.json', 'False Squat'))
seq_5_angles = get_minmax_angles_mocap(mocap_poseprocessor.load('data/sequences/no_squat/complete-session.json', 'No Squat'))

minmax_result = get_distances_minmax(seq_gt_angles, [seq_1_angles, seq_2_angles, seq_3_angles, seq_4_angles, seq_5_angles])
print(minmax_result)
