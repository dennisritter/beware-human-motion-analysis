from Sequence import Sequence
import numpy as np


def calc_angle_hip_flexion_extension(seq: Sequence, joints: dict) -> list:
    # NOTE: Observations and Potential Problems:
    #       * Torso position might be incorrect for heavy people with big upper body
    #           => Possible solution: Calibration to calculate a bias? [Tell user to Stand (start) -> Track pose -> calculate bias]
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
    hip = seq.positions[:, joints["angle_vertex"], 1:]
    knee = seq.positions[:, joints["rays"][0], 1:]
    torso = seq.positions[:, joints["rays"][1], 1:]
    angles = []
    for i in range(len(hip)):
        hip_knee = knee[i] - hip[i]
        hip_torso = torso[i] - hip[i]
        cos_angle = np.dot(hip_knee, hip_torso) / (np.linalg.norm(hip_knee) * np.linalg.norm(hip_torso))
        angle = np.arccos(cos_angle)
        angles.append(180 - np.degrees(angle))

    print(angles)
    return angles
