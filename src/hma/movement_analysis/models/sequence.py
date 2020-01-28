"""Contains the code for the sequence model including the scenegraph and angle computation."""
import json

import networkx as nx
import numpy as np
from hma.movement_analysis import angle_calculations as acm


class Sequence:
    """Represents a motion sequence.

    Attributes:
        body_parts (dict): A dictionary mapping body part names to position indices in the "positions" attribute array.
        positions (list): The tracked body part positions for each frame.
        timestamps (list): The timestamps for each tracked frame.
        name (str): The name of this sequence.
        joint_angles (list): The calculated angles derived from the tracked positions of this sequence
    """
    def __init__(self,
                 body_parts: dict,
                 positions: np.ndarray,
                 timestamps: np.ndarray,
                 name: str = 'sequence',
                 joint_angles: list = None,
                 scene_graph: nx.DiGraph = None):
        self.name = name
        # Number, order and label of tracked body parts
        # Example: { "head": 0, "neck": 1, ... }
        self.body_parts = body_parts

        # A Boolean mask list to exclude all frames, where all positions are 0.0
        zero_frames_filter_list = self._filter_zero_frames(positions)
        # Defines positions of each bodypart
        # 1. Dimension = Time
        # 2. Dimension = Bodypart
        # 3. Dimension = x, y, z
        # Example: [
        #           [[f1_bp1_x, f1_bp1_x, f1_bp1_x], [f1_bp2_x, f1_bp2_x, f1_bp2_x], ...],
        #           [[f2_bp1_x, f2_bp1_x, f2_bp1_x], [f2_bp2_x, f2_bp2_x, f2_bp2_x], ...],
        #           ...
        #          ]
        # shape: (num_body_parts, num_keypoints, xyz)
        self.positions = np.array(positions)[zero_frames_filter_list]

        # Timestamps for when the positions have been tracked
        # Example: [<someTimestamp1>, <someTimestamp2>, <someTimestamp3>, ...]
        self.timestamps = np.array(timestamps)[zero_frames_filter_list]
        # Stores joint angles for each frame
        # NOTE: Body part indices are the indices stored in self.body_parts.
        # NOTE: If angles have been computed, the stored value is a dictionary with at least one key "flexion_extension"
        #       and a "abduction_adduction" key for ball joints.
        # NOTE: If no angles have been computed for a particular joint, the stored value is None.
        self.joint_angles = self._calc_joint_angles() if joint_angles is None else np.array(joint_angles)
        # A graph that defines the hierarchy between human body parts
        self.scene_graph = nx.DiGraph([
            ("pelvis", "torso"),
            ("torso", "neck"),
            ("neck", "l_shoulder"),
            ("l_shoulder", "l_elbow"),
            ("l_elbow", "l_wrist"),
            ("neck", "r_shoulder"),
            ("r_shoulder", "r_elbow"),
            ("r_elbow", "r_wrist"),
            ("pelvis", "l_hip"),
            ("l_hip", "l_knee"),
            ("l_knee", "l_ankle"),
            ("pelvis", "r_hip"),
            ("r_hip", "r_knee"),
            ("r_knee", "r_ankle"),
        ]) if scene_graph is None else scene_graph

    def __len__(self) -> int:
        return len(self.joint_angles)

    def __getitem__(self, item) -> 'Sequence':
        """Returns the sub-sequence item. You can either specifiy one element by index or use numpy-like slicing.

        Args:
            item (int/slice): Defines a particular frame or slice from all frames of this sequence.

        Raises NotImplementedError if index is given as tuple.
        Raises TypeError if item is not of type int or slice.
        """
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            # return Seq([self[i] for i in range(start, stop, step)])
        elif isinstance(item, int):
            start, stop, step = item, item + 1, 1
        elif isinstance(item, tuple):
            raise NotImplementedError("Tuple as index")
        else:
            raise TypeError(f"Invalid argument type: {type(item)}")

        return Sequence(self.body_parts, self.positions[start:stop:step], self.timestamps[start:stop:step], self.name,
                        self.joint_angles[start:stop:step])

    #? Do we need a to_mocap_json method?
    def to_json(self) -> str:
        """Returns the sequence instance as a json-formatted string."""
        json_dict = {
            'name': self.name,
            'body_parts': self.body_parts,
            'positions': self.positions.tolist(),
            'timestamps': self.timestamps.tolist(),
            'joint_angles': self.joint_angles.tolist(),
            # TODO: return scene_graph
            # 'scene_graph': self.scene_graph,
        }
        return json.dumps(json_dict)

    @classmethod
    def from_json(cls, json_str: str) -> 'Sequence':
        """Creates a new Sequence instance from a json-formatted string.

        Args:
            json_str (str): The json-formatted string.

        Returns:
            Exercise: a new Sequence instance from the given input.
        """
        json_dict = json.loads(json_str)
        return cls(**json_dict)

    @classmethod
    def from_file(cls, path: str) -> 'Sequence':
        """Loads an Sequence .json file and returns an Sequence object.

        Args:
            path (str): Path to the json file

        Returns:
            Sequence: a new Sequence instance from the given input.
        """
        with open(path, 'r') as sequence_file:
            # load, parse file from json and return class
            return cls.from_json(sequence_file.read())

    @classmethod
    def from_mocap_file(cls, path: str, name: str = 'sequence') -> 'Sequence':
        """Loads an sequence .json file and returns an Sequence object.

        Args:
            path (str): Path to the json file

        Returns:
            Sequence: a new Sequence instance from the given input.
        """
        with open(path, 'r') as sequence_file:
            # load, parse file from json and return class

            mocap_sequence = json.loads(sequence_file.read())
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

            return cls(body_parts, positions, timestamps, name=name)

    def _calc_joint_angles(self) -> np.ndarray:
        """Returns a 3-D list of joint angles for all frames, body parts and angle types."""
        # TODO: Update Angle Calculation to Euler Sequences
        n_frames = len(self.timestamps)
        n_body_parts = len(self.body_parts)
        n_angle_types = 3
        bp = self.body_parts

        ls = acm.calc_angles_shoulder_left(self.positions, bp["shoulder_l"], bp["shoulder_r"], bp["torso"], bp["elbow_l"])
        rs = acm.calc_angles_shoulder_right(self.positions, bp["shoulder_r"], bp["shoulder_l"], bp["torso"], bp["elbow_r"])
        lh = acm.calc_angles_hip_left(self.positions, bp["hip_l"], bp["hip_r"], bp["torso"], bp["knee_l"])
        rh = acm.calc_angles_hip_right(self.positions, bp["hip_r"], bp["hip_l"], bp["torso"], bp["knee_r"])
        le = acm.calc_angles_elbow(self.positions, bp["elbow_l"], bp["shoulder_l"], bp["wrist_l"])
        re = acm.calc_angles_elbow(self.positions, bp["elbow_r"], bp["shoulder_r"], bp["wrist_r"])
        lk = acm.calc_angles_knee(self.positions, bp["knee_l"], bp["hip_l"], bp["ankle_l"])
        rk = acm.calc_angles_knee(self.positions, bp["knee_r"], bp["hip_r"], bp["ankle_r"])

        joint_angles = np.zeros((n_frames, n_body_parts, n_angle_types))
        for frame in range(0, n_frames):
            joint_angles[frame][bp["shoulder_l"]] = ls[frame]
            joint_angles[frame][bp["shoulder_r"]] = rs[frame]
            joint_angles[frame][bp["hip_l"]] = lh[frame]
            joint_angles[frame][bp["hip_r"]] = rh[frame]
            joint_angles[frame][bp["elbow_l"]] = le[frame]
            joint_angles[frame][bp["elbow_r"]] = re[frame]
            joint_angles[frame][bp["knee_l"]] = lk[frame]
            joint_angles[frame][bp["knee_r"]] = rk[frame]

        return joint_angles

    def get_positions_2d(self) -> np.ndarray:
        """Returns the positions for all keypoints in shape: (num_frames, num_bodyparts * xyz)."""
        return np.reshape(self.positions, (self.positions.shape[0], -1))

    def merge(self, sequence: 'Sequence') -> 'Sequence':
        """Returns the merged two sequences.

        Raises ValueError if either the body_parts, the poseformat or the body_parts and keys within the joint_angles do not match!
        """
        if self.body_parts != sequence.body_parts:
            raise ValueError('body_parts of both sequences do not match!')

        # concatenate positions, timestamps and angles
        self.positions = np.concatenate((self.positions, sequence.positions), axis=0)
        self.timestamps = np.concatenate((self.timestamps, sequence.timestamps), axis=0)
        self.joint_angles = np.concatenate((self.joint_angles, sequence.joint_angles), axis=0)

        return self

    def _filter_zero_frames(self, positions: np.ndarray) -> list:
        """Returns a filter mask list to filter frames where all positions equal 0.0.

        Checks whether the sum of all coordinates for a frame is 0.0
            True -> keep this frame
            False -> remove this frame

        Args:
            positions (np.ndarray): The positions to filter "Zero-Position-Frames" from

        Returns:
            (list<boolean>): The filter list.
        """
        bool_list = []
        for pos in positions:
            bool_list.append(np.sum(pos) != 0)

        return bool_list
