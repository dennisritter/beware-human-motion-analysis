from hma.movement_analysis.enums.pose_format_enum import PoseFormatEnum
import json
import numpy as np
from hma.movement_analysis.models.sequence import Sequence


class PoseProcessor:
    """Loads and Processes sequence file representations according to the defined poseformat.

    Loads and Processes sequence file representations according to the defined poseformat. 
    Each available poseformat needs a condition in the "self.process" function and a separate
    "process_sequence_<poseformat>" function in order to load sequences of that poseformat.

    Attributes:
        poseformat (PoseFormatEnum): Specifies the pose format of sequences that will are imported. 
    """
    def __init__(self, poseformat: PoseFormatEnum):
        """ PoseProcessor Constructor
        Args:
            poseformat (PoseFormatEnum): A PoseFormat Enumeration member defining input format of the PoseProcessor instance.

        Returns:
            (PoseProcessor) A PoseProcessor instance.
        """
        if (not isinstance(poseformat, PoseFormatEnum)):
            raise ValueError("'poseformat' parameter must be a member of 'PoseFormat' enumeration.")
        self.poseformat = poseformat

    def load(self, path: str, name: str = 'Some Sequence') -> 'Sequence':
        """ Loads a Sequence from path and maps it to the specified poseformat.

        Args:
            path (str): The path to a sequences JSON file.
            name (str): Specifies the name of the returned sequence.

        Returns:
            (Sequence) A Sequence object.
        """
        with open(path, 'r') as myfile:
            seq = myfile.read()
        return self.process(seq, name)

    def process(self, input: str, name: str = 'Some Sequence') -> 'Sequence':
        """
        Args:
            input (str): The motion sequence input string to convert.
            name (str): Specifies the name of the returned sequence.
        Returns
            (Sequence): The Sequence Object instance representing the motion sequence of the input string.
        """
        if (self.poseformat == PoseFormatEnum.MOCAP):
            return self.process_sequence_mocap(input, name=name)

    def process_sequence_mocap(self, input: str, name='Some Sequence') -> 'Sequence':
        """
        Args
            input (str): The MoCap motion sequence input string to convert.
            name (str): Specifies the name of the returned sequence.
        Returns
            (Sequence): The Sequence Object instance representing the MoCap motion sequence of the input string.
        """
        mocap_sequence = json.loads(input)
        body_parts = mocap_sequence["format"]
        positions = np.array(mocap_sequence["frames"])
        timestamps = np.array(mocap_sequence["timestamps"])

        # reshape positions to 3d array
        positions = np.reshape(positions, (np.shape(positions)[0], int(np.shape(positions)[1] / 3), 3))

        # Center Positions by subtracting the mean of each coordinate
        positions[:, :, 0] -= np.mean(positions[:, :, 0])
        positions[:, :, 1] -= np.mean(positions[:, :, 1])
        positions[:, :, 2] -= np.mean(positions[:, :, 2])

        # MoCap coordinate system is left handed -> flip x-axis to adjust data for right handed coordinate system
        positions[:, :, 0] *= -1
        # Switch Y and Z axis.
        # In Mocap Y points up and Z to the back -> We want Z to point up and Y to the front,
        y_positions_mocap = positions[:, :, 1].copy()
        z_positions_mocap = positions[:, :, 2].copy()
        positions[:, :, 1] = z_positions_mocap
        positions[:, :, 2] = y_positions_mocap

        return Sequence(body_parts, positions, timestamps, body_parts, name=name)
