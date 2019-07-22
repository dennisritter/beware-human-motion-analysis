from Sequence import Sequence
import numpy as np


def calc_angles_lefthip_flexion_extension(seq: Sequence, joints: dict) -> list:
    """ Calculates the hips flexion/extension angles for each frame of the Sequence.
    Parameters
    ----------
    seq : Sequence
        A Motion Sequence
    joints : dict
        The joints to use for angle calculation.
        Attributes:
            angle_vertex : int
            rays : list<int>
        Example: { "angle_vertex": 1, "rays": [0, 2] }
    """
    # print(seq.positions[:, 6, :])
    left_hip = seq.positions[:, joints["angle_vertex"], :]
    left_knee = seq.positions[:, joints["rays"][0], :]
    torso = seq.positions[:, joints["rays"][1], :]
    angles = []
    for i in range(len(left_hip)):
        l_hip_l_knee = left_knee[i] - left_hip[i]
        l_hip_torso = torso[i] - left_hip[i]
        # TODO: Understand
        cos_angle = np.dot(l_hip_l_knee, l_hip_torso) / (np.linalg.norm(l_hip_l_knee) * np.linalg.norm(l_hip_torso))
        angle = np.arccos(cos_angle)
        angles.append(180 - np.degrees(angle))

    print(angles[50])
    return angles
