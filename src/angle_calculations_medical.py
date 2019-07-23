from Sequence import Sequence
import numpy as np

def calc_angle_2d(angle_vertex_2d: list, ray_vertex_a: list, ray_vertex_b: list) -> float:
    """ Calculates the angle between angle_vertex_2d-ray_vertex_a and angle_vertex_2d-ray_vertex_b in 2D space.
    Returns
    ----------
    float
        Angle between angle_vertex_2d-ray_vertex_a and angle_vertex_2d-ray_vertex_b in degrees
    """
    ray_a = ray_vertex_a - angle_vertex_2d
    ray_b = ray_vertex_b - angle_vertex_2d
    cos_angle = np.dot(ray_a, ray_b) / (np.linalg.norm(ray_a) * np.linalg.norm(ray_b))
    return np.degrees(np.arccos(cos_angle))

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
        angles.append(180 - calc_angle_2d(hip[i], knee[i], torso[i]))

    return angles


def calc_angle_knee_flexion_extension(seq: Sequence, joints: dict) -> list:
    # NOTE: Observations and Potential Problems:
    """ Calculates the Knees flexion/extension angles for each frame of the Sequence.
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
    knee = seq.positions[:, joints["angle_vertex"], :]
    hip = seq.positions[:, joints["rays"][0], :]
    ankle = seq.positions[:, joints["rays"][1], :]
    angles = []
    for i in range(len(knee)):
        angles.append(calc_angle_2d(knee[i], hip[i], ankle[i]))

    return angles
