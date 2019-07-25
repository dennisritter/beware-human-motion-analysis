from PoseFormatEnum import PoseFormatEnum
import json
import numpy as np
from Sequence import Sequence


class PoseMapper:
    # TODO: Remove?
    OPENPOSE_MPI_BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                               "Background": 15}
    # TODO: Remove?
    OPENPOSE_MPI_PAIRS = [["Head", "Neck"],
                          ["Neck", "RShoulder"],
                          ["RShoulder", "RElbow"],
                          ["RElbow", "RWrist"],
                          ["Neck", "LShoulder"],
                          ["LShoulder", "LElbow"],
                          ["LElbow", "LWrist"],
                          ["Neck", "Chest"],
                          ["Chest", "RHip"],
                          ["RHip", "RKnee"],
                          ["RKnee", "RAnkle"],
                          ["Chest", "LHip"],
                          ["LHip", "LKnee"],
                          ["LKnee", "LAnkle"]]

    def __init__(self, poseformat: PoseFormatEnum):
        """ PoseMapper Constructor
        Parameters
        ----------
        poseformat : PoseFormatEnum
            A PoseFormat Enumeration member defining input format of the PoseMapper instance.

        Returns
        ----------
        PoseMapper
            A PoseMapper instance.
        """
        if(not isinstance(poseformat, PoseFormatEnum)):
            raise ValueError(
                "'poseformat' parameter must be a member of 'PoseFormat' enumeration.")
        self.poseformat = poseformat

    def load(self, path: str, name: str = 'Some Sequence') -> Sequence:
        """ Loads a Sequence from path and maps it to the specified poseformat.
        """
        with open(path, 'r') as myfile:
            seq = myfile.read()
        return self.map(seq, name)

    def map(self, input: str, name: str = 'Some Sequence') -> Sequence:
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

        # Find b 

        # Center Positions by subtracting the mean of each coordinate
        positions[:, :,
                  0] -= np.mean(positions[:, :, 0])
        positions[:, :,
                  1] -= np.mean(positions[:, :, 1])
        positions[:, :,
                  2] -= np.mean(positions[:, :, 2])
        return Sequence(body_parts, positions, timestamps, self.poseformat, name=name)
