from enum import Enum
import json

# TODO: Optimize returned values
# TODO: Implement Sequence class
# {
#   body_parts: ["Head", "Neck", "RShoulder", "RElbow", ...],
#   body_pairs: [[["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"], ["RElbow", "RWrist"], ...],
#   positions: [
#                 [[part-i.x, part-i.y, part-i.z], [part-i.x, part-i.y, part-i.z], [part-i.x, part-i.y, part-i.z]],
#                 [[part-i+1.x, part-i+1.y, part-i+1.z], [part-i+1.x, part-i+1.y, part-i+1.z], [part-i+1.x, part-i+1.y, part-i+1.z]],
#                 ...
#              ],
#   timestamps: [<someTimestamp>] -> Must be same size as arrays contained in positions (num_keypoints == num_timestamps)
# }


class PoseMappingEnum(Enum):
    MOCAP_TO_OPENPOSE_MPI = 1


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

    def map(self, input: str) -> dict:
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
        return self.map_sequence_mocap_to_openpose_mpi(input)

    def map_sequence_mocap_to_openpose_mpi(self, input: str) -> dict:
        """
        Parameters
        ----------
        input : str
            The sequence mocap input string to convert into a openpose MPI output dictionary.
        Returns
        ----------
        dict
           The converted output Pose dictionary for the given input in the constructor specified output pose format.
        """
        mocap_sequence = json.loads(input)
        mpi_sequence = []
        # For each pose in the keypoints Array
        for pose in mocap_sequence["keypoints"]:
            # Accessing the actual keypoint through the timestamp object key
            for timestamp in pose:
                mocap_parts_positions = pose[timestamp]
                mpi_parts_positions = self.map_mocap_to_openpose_mpi(
                    mocap_parts_positions)
                # TODO: Check if we really need the timestamp here
                mpi_sequence.append({f'{timestamp}': mpi_parts_positions})
        return mpi_sequence

    def map_mocap_to_openpose_mpi(self, input: dict) -> dict:
        """
        Parameters
        ----------
        input : dict
            The MOCAP Pose input string to convert into the MPII representation.
        Returns
        ----------
        dict
           A MPII pose representation dictionary of the MOCAP input pose string.
        """
        mocap_parts_positions = input
        # For each Body Part
        for key in mocap_parts_positions:
            part_position = mocap_parts_positions[key].split(',')
            part_position_dict = {}
            # Convert part position Array<str> to Dictionary containing float values
            part_position_dict['x'] = float(part_position[0])
            part_position_dict['y'] = float(part_position[1])
            part_position_dict['z'] = float(part_position[2])
            mocap_parts_positions[key] = part_position_dict
        # The Output Dictionary
        op_mpi_dict = {
            "RAnkle": mocap_parts_positions["RightAnkle"],
            "RKnee": mocap_parts_positions["RightKnee"],
            "RHip": mocap_parts_positions["RightHip"],
            "LHip": mocap_parts_positions["LeftHip"],
            "LKnee": mocap_parts_positions["LeftKnee"],
            "LAnkle": mocap_parts_positions["LeftAnkle"],
            "Chest": mocap_parts_positions["Torso"],
            "Neck": mocap_parts_positions["Neck"],
            "Head": mocap_parts_positions["Head"],
            "RWrist": mocap_parts_positions["RightWrist"],
            "RElbow": mocap_parts_positions["RightElbow"],
            "RShoulder": mocap_parts_positions["RightShoulder"],
            "LShoulder": mocap_parts_positions["LeftShoulder"],
            "LElbow": mocap_parts_positions["LeftElbow"],
            "LWrist": mocap_parts_positions["LeftWrist"],
        }
        return op_mpi_dict
