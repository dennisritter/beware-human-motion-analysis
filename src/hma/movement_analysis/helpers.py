import numpy as np
from .enums.angle_types import AngleTypes


def reformat_angles_dtw(seq):
    """Returns the sequences reformatted joint angles to use them for the tslearn.metrics.dtw_path function.
    """
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
