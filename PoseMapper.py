from enum import Enum
import json
from Sequence import Sequence

# TODO: Optimize returned values
# TODO: Implement Sequence class
# {
#   body_parts: ["Head", "Neck", "RShoulder", "RElbow", ...],
#   body_pairs: [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"], ["RElbow", "RWrist"], ...],
#   positions: [
#                 [[part-i.x, part-i.y, part-i.z], [part-i.x, part-i.y, part-i.z], [part-i.x, part-i.y, part-i.z]],
#                 [[part-i+1.x, part-i+1.y, part-i+1.z], [part-i+1.x, part-i+1.y, part-i+1.z], [part-i+1.x, part-i+1.y, part-i+1.z]],
#                 ...
#              ],
#   timestamps: [<someTimestamp>] -> Must be same size as arrays contained in positions (num_keypoints == num_timestamps)
# }


class PoseMappingEnum(Enum):
    MOCAP = 1


class PoseMapper:
    OPENPOSE_MPI_BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                               "Background": 15}
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
            A PoseMapping Enumeration member defining input/output format of the mapper.

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
            The Pose input string to convert into the constructor specified output.
        Returns
        ----------
        dict
           The converted output Pose dictionary for the given input in the constructor specified output pose format.
        """
        if (self.mapping == PoseMappingEnum.MOCAP):
            return self.map_sequence_mocap(input)

    def map_sequence_mocap(self, input: str) -> Sequence:
        mocap_sequence = json.loads(input)
        body_parts = []
        positions = []
        timestamps = []

        # For each keypoint in the keypoints Array
        for keypoint in mocap_sequence["keypoints"]:
            # Accessing the actual keypoint through the timestamp object key
            for timestamp in keypoint:
                # Store timestamp in timestamp List
                timestamps.append(float(timestamp))
                # Append 1st level list to positions
                positions.append([])
                part_positions = keypoint[timestamp]
                for body_part in part_positions:
                    # Add all body parts if not added already
                    if (len(body_parts) < len(part_positions.keys())):
                        body_parts.append(body_part)
                    # Append position in [x_pos, y_pos, z_pos] format to the 1st level last list (the last timestamp index)
                    position = part_positions[body_part].split(",")
                    positions[len(positions)-1].append([float(position[0]), float(
                        position[1]), float(position[2])])

        return Sequence(body_parts, positions, timestamps)
