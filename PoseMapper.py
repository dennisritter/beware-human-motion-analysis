from enum import Enum
import json
import numpy
from Sequence import Sequence


class PoseMappingEnum(Enum):
    MOCAP = 1


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

    def __init__(self, mapping: PoseMappingEnum):
        """ PoseMapper Constructor
        Parameters
        ----------
        mapping : PoseMappingEnum
            A PoseMapping Enumeration member defining input format of the PoseMapper instance.

        Returns
        ----------
        PoseMapper
            A PoseMapper instance.
        """
        if(not isinstance(mapping, PoseMappingEnum)):
            raise ValueError(
                "'mapping' parameter must be a member of 'PoseMapping' enumeration.")
        self.mapping = mapping

    def map(self, input: str) -> Sequence:
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
        if (self.mapping == PoseMappingEnum.MOCAP):
            return self.map_sequence_mocap(input)

    def map_sequence_mocap(self, input: str) -> Sequence:
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
        body_parts = numpy.empty(len(MOCAP_BODY_PARTS), dtype=object)
        positions = []
        timestamps = []

        # Add Body Parts to body_parts array in the correct order
        for (key, item) in MOCAP_BODY_PARTS.items():
            body_parts[item] = key
            positions.append([])

        # For each keypoint in the keypoints Array
        for keypoint in mocap_sequence["keypoints"]:
            # Accessing the actual keypoint through the timestamp object key
            for timestamp in keypoint:
                # Store timestamp in timestamp List
                timestamps.append(float(timestamp))
                # Append 1st level list to positions
                part_positions = keypoint[timestamp]
                for body_part in part_positions:
                    # Append position in [x_pos, y_pos, z_pos] format to the 1st level last list (the last timestamp index)
                    position = part_positions[body_part].split(",")
                    positions[MOCAP_BODY_PARTS[body_part]].append([float(position[0]), float(
                        position[1]), float(position[2])])

        positions = numpy.array(positions)
        timestamps = numpy.array(timestamps)
        body_parts = numpy.array(body_parts)

        # Center Positions by subtracting the mean of each coordinate
        positions[:, :,
                  0] -= numpy.mean(numpy.ndarray.flatten(positions[:, :, 0]))
        positions[:, :,
                  1] -= numpy.mean(numpy.ndarray.flatten(positions[:, :, 1]))
        positions[:, :,
                  2] -= numpy.mean(numpy.ndarray.flatten(positions[:, :, 2]))

        return Sequence(body_parts, positions, timestamps)
