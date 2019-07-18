from PoseFormatEnum import PoseFormatEnum
import json
import numpy as np
from Sequence import Sequence

class ExercisePoseSegmentor:
    """ 
    """

    def __init__(self, poseformat: PoseFormatEnum):
        """ PoseMapper Constructor
        Parameters
        ----------
        mapping : PoseFormatEnum
            A PoseMapping Enumeration member defining input format of the PoseMapper instance.

        Returns
        ----------
        PoseMapper
            A PoseMapper instance.
        """
        if(not isinstance(poseformat, PoseFormatEnum)):
            raise ValueError(
                "'mapping' parameter must be a member of 'PoseFormatEnum'")
        self.poseformat = poseformat

    def map(self, input: str, name: str = 'sequence') -> Sequence:
        """
        Parameters
        ----------
        input : str
            The motion sequence input string to convert.
        Returns
        ----------
        Sequence
           The Sequence Object instance representing the motion sequence of the input string.
        """
        if (self.poseformat == PoseFormatEnum.MOCAP):
            return self.map_sequence_mocap(input, name=name)

    def map_sequence_mocap(self, input: str, name='sequence') -> Sequence:
        """
        Parameters
        ----------
        input : str
            The mocap motion sequence input string to convert.
        Returns
        ----------
        Sequence
           The Sequence Object instance representing the motion sequence of the input string.
        """
        MOCAP_BODY_PARTS = {"Head": 0, "Neck": 1, "RightShoulder": 2, "RightElbow": 3, "RightWrist": 4,
                            "LeftShoulder": 5, "LeftElbow": 6, "LeftWrist": 7, "RightHip": 8, "RightKnee": 9,
                            "RightAnkle": 10, "LeftHip": 11, "LeftKnee": 12, "LeftAnkle": 13, "Torso": 14, "Waist": 15, }
        mocap_sequence = json.loads(input)

        body_parts = mocap_sequence["format"]
        positions = np.array(mocap_sequence["frames"])
        timestamps = np.array(mocap_sequence["timestamps"])

        # reshape positions to 3d array
        positions = np.reshape(positions, (np.shape(positions)[0], int(np.shape(positions)[1]/3), 3))

        # Center Positions by subtracting the mean of each coordinate
        positions[:, :,
                  0] -= np.mean(positions[:, :, 0])
        positions[:, :,
                  1] -= np.mean(positions[:, :, 1])
        positions[:, :,
                  2] -= np.mean(positions[:, :, 2])
        return Sequence(body_parts, positions, timestamps, name=name)
