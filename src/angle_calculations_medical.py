from Sequence import Sequence
import numpy as np


def calc_angle(angle_vertex: list, ray_vertex_a: list, ray_vertex_b: list) -> float:
    """ Calculates the angle between angle_vertex_2d-ray_vertex_a and angle_vertex_2d-ray_vertex_b in 2D space.
    Returns
    ----------
    float
        Angle between angle_vertex_2d-ray_vertex_a and angle_vertex_2d-ray_vertex_b in degrees
    """
    ray_a = ray_vertex_a - angle_vertex
    ray_b = ray_vertex_b - angle_vertex
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
        # Substract angle from 180 because 'Normal Standing' is defined as 0°
        angles.append(180 - calc_angle(hip[i], knee[i], torso[i]))

    return angles


def calc_angle_hip_abduction_adduction(seq: Sequence, joints: dict) -> list:
    # NOTE: Observations and Potential Problems:
    #   * 'Normal Standing' angle will never be 0° because using left/right shoulder-hip ray for calculation.
    #       Possible solution:  -> Use a bias of 5-10°
    #                           -> Don't use shoulder-hip ray but direct shoulder-floor ray for calculation
    """ Calculates the hips abduction/adduction angles for each frame of the Sequence.
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
    # Ignore Z Axis for abduction/adduction
    hip = seq.positions[:, joints["angle_vertex"], :2]
    knee = seq.positions[:, joints["rays"][0], :2]
    torso = seq.positions[:, joints["rays"][1], :2]
    angles = []
    for i in range(len(hip)):
        # Substract angle from 180 because 'Normal Standing' is defined as 0°
        angles.append(180 - calc_angle(hip[i], knee[i], torso[i]))

    return angles


def calc_angle_knee_flexion_extension(seq: Sequence, joints: dict) -> list:
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
        # Substract angle from 180 because 'Normal Standing' is defined as 0°
        angles.append(180 - calc_angle(knee[i], hip[i], ankle[i]))

    return angles


def calc_angle_shoulder_flexion_extension(seq: Sequence, joints: dict) -> list:
    """ Calculates the Shoulders flexion/extension angles for each frame of the Sequence.
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
    shoulder = seq.positions[:, joints["angle_vertex"], 1:]
    elbow = seq.positions[:, joints["rays"][0], 1:]
    hip = seq.positions[:, joints["rays"][1], 1:]
    angles = []
    for i in range(len(shoulder)):
        # Substract angle from 180 because 'Normal Standing' (hanging arm) is defined as 0°
        angles.append(calc_angle(shoulder[i], elbow[i], hip[i]))

    return angles


def calc_angle_shoulder_abduction_adduction(seq: Sequence, joints: dict) -> list:
    # NOTE: Observations and Potential Problems:
    #   * 'Normal Standing' angle will never be 0° because using left/right shoulder-hip ray for calculation.
    #       Possible solution:  -> Use a bias of 5-10°
    #                           -> Don't use shoulder-hip ray but direct shoulder-floor ray for calculation
    """ Calculates the hips abduction/adduction angles for each frame of the Sequence.
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
    # Ignore Z Axis for abduction/adduction
    shoulder = seq.positions[:, joints["angle_vertex"], :2]
    elbow = seq.positions[:, joints["rays"][0], :2]
    hip = seq.positions[:, joints["rays"][1], :2]
    angles = []
    for i in range(len(shoulder)):
        # Substract angle from 180 because 'Normal Standing' is defined as 0°
        angles.append(180 - calc_angle(shoulder[i], elbow[i], hip[i]))

    return angles


def calc_angle_shoulder_innerrotation_outerrotation(seq: Sequence, joints: dict) -> list:
    """ Calculates the hips inner/outer rotation angles for each frame of the Sequence.
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

    # Euler transformation from wrist-xyz -> wrist-xyz' of rotation vector Shoulder-Elbow
    # Get Shoulder-Elbow Vector -> get 0° Wrist position wrist-xyz (imagined position) -> transform to wrist-xyz' (real position) -> Calc Angle from transformation
    #
    # How to get wrist-xyz 0° reference vertex?
    # -> Depends on shoulder Flexion/Extension angles of rotation vector Shoulder-Elbow
    # Front View:
    # -> If FlexEx angle = 0° -> wrist-0° is on orthogonal vector from Shoulder-Elbow to (X=SE.x, Y=W.y, Z=0)
    # -> If FlexEx angle = 90° -> wrist-0° is on orthogonal vector from Shoulder-Elbow to (X=SE.x, Y=1.0, Z=W.z)
    # ==> For other camera positions: Need angle of rotation around Y-World Axis
    return []


def calc_angle_elbow_flexion_extension(seq: Sequence, joints: dict) -> list:
    """ Calculates the Elbows flexion/extension angles for each frame of the Sequence.
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
    elbow = seq.positions[:, joints["angle_vertex"], :]
    wrist = seq.positions[:, joints["rays"][0], :]
    shoulder = seq.positions[:, joints["rays"][1], :]
    angles = []
    for i in range(len(elbow)):
        # Substract angle from 180 because 'Normal Standing' (straight arm) is defined as 0°
        angles.append(180 - calc_angle(elbow[i], wrist[i], shoulder[i]))

    return angles
