from enum import Enum
import json


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
        print(self.mapping)

    def map(self, input: str) -> str:
        """
        Parameters
        ----------
        input : str
            The Pose input string to convert into the constructor specified output.
        Returns
        ----------
        str   
           The converted output Pose string for the given input in the constructor specified output pose format.   
        """
        if (self.mapping == PoseMappingEnum.MOCAP_TO_OPENPOSE_MPI):
            return self.map_mocap_to_openpose_mpi(input)

    def map_mocap_to_openpose_mpi(self, input: str) -> dict:
        """
        Parameters
        ----------
        input : str
            The MOCAP Pose input string to convert into the MPII representation.
        Returns
        ----------
        dict
           A MPII pose representation dictionary of the MOCAP input pose string.
        """
        mocap_dict = json.loads(input)
        # For each Body Part
        for key in mocap_dict:
            part_position = mocap_dict[key].split(',')
            part_position_dict = {}
            # Convert part position Array<str> to Dictionary containing float values
            part_position_dict['x'] = float(part_position[0])
            part_position_dict['y'] = float(part_position[1])
            part_position_dict['z'] = float(part_position[2])
            mocap_dict[key] = part_position_dict
        # The Output Dictionary
        op_mpi_dict = {
            "RAnkle": mocap_dict["RightAnkle"],
            "RKnee": mocap_dict["RightKnee"],
            "RHip": mocap_dict["RightHip"],
            "LHip": mocap_dict["LeftHip"],
            "LKnee": mocap_dict["LeftKnee"],
            "LAnkle": mocap_dict["LeftAnkle"],
            "Chest": mocap_dict["Torso"],
            "Neck": mocap_dict["Neck"],
            "Head": mocap_dict["Head"],
            "RWrist": mocap_dict["RightWrist"],
            "RElbow": mocap_dict["RightElbow"],
            "RShoulder": mocap_dict["RightShoulder"],
            "LShoulder": mocap_dict["LeftShoulder"],
            "LElbow": mocap_dict["LeftElbow"],
            "LWrist": mocap_dict["LeftWrist"],
        }
        return op_mpi_dict
