import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tslearn.metrics as ts
import seaborn as sns
import numpy as np
from hma.movement_analysis.Sequence import Sequence
from hma.movement_analysis.pose_processor import PoseProcessor
from hma.movement_analysis.enums.pose_format_enum import PoseFormatEnum
from hma.movement_analysis import angle_calculations as acm
from hma.movement_analysis import distance
from hma.movement_analysis.enums.angle_types import AngleTypes
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
        print(seq_frame_angles)
    return np.array(dtw_angles)


# Calculates dtw_distance of sequences in seqs_angles compared to ground_truth_angles
# Returns a list of floats
def get_distances_dtw(ground_truth_angles, seqs_angles):
    distances = []
    for seq_angles in seqs_angles:
        distances.append(ts.dtw_path(ground_truth_angles, seq_angles)[1])
    return distances


# Get PoseProcessor instance for MOCAP sequences
mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)

# Convert json sequence file to Sequence object and calculate joint angles for dtw comparison
seq_gt_angles = get_dtw_angles_mocap(mocap_poseprocessor.load('data/sequences/squat_3/complete-session.json', 'Squat_Ground_Truth'))
seq_1_angles = get_dtw_angles_mocap(mocap_poseprocessor.load('data/sequences/squat_1/complete-session.json', 'Squat 1'))
seq_2_angles = get_dtw_angles_mocap(mocap_poseprocessor.load('data/sequences/squat_2/complete-session.json', 'Squat 2'))
seq_3_angles = get_dtw_angles_mocap(mocap_poseprocessor.load('data/sequences/squat_4/complete-session.json', 'Squat 4'))
seq_4_angles = get_dtw_angles_mocap(mocap_poseprocessor.load('data/sequences/squat_false/complete-session.json', 'False Squat'))
seq_5_angles = get_dtw_angles_mocap(mocap_poseprocessor.load('data/sequences/no_squat/complete-session.json', 'No Squat'))

# Compare sequences to ground truth sequence (squat)
dtw_result = get_distances_dtw(seq_gt_angles, [seq_1_angles, seq_2_angles, seq_3_angles, seq_4_angles, seq_5_angles])
print(dtw_result)
