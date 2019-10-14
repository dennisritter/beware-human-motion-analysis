import numpy as np
from sklearn.decomposition import PCA
from hma.movement_analysis.enums.pose_format_enum import PoseFormatEnum
from hma.movement_analysis import angle_calculations as acm
from matplotlib import pyplot as plt


class Sequence:

    def __init__(self, body_parts: dict, positions: list, timestamps: list, poseformat: PoseFormatEnum, name: str = 'sequence', joint_angles: list = None):
        self.name = name
        self.poseformat = poseformat
        # Number, order and label of tracked body parts
        # Example: { "Head": 0, "RightShoulder": 1, ... }
        self.body_parts = body_parts

        # A Boolean ndarray mask to exclude all frames, where all positions are 0.0
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

        self.positions_2d = None  # self.get_positions_2d()

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, item):
        """
        Returns the sub-sequence item. You can either specifiy one element by index or use numpy-like slicing.

        Raises NotImplementedError if index is given as tuple.
        Raises TypeError if item is not of type int or slice.
        """
        if isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            # return Seq([self[i] for i in range(start, stop, step)])
        elif isinstance(item, int):
            start, stop, step = item, item+1, 1
        elif isinstance(item, tuple):
            raise NotImplementedError("Tuple as index")
        else:
            raise TypeError(f"Invalid argument type: {type(item)}")

        return Sequence(self.body_parts, self.positions[start:stop:step], self.timestamps[start:stop:step], self.poseformat, self.name, self.joint_angles[start:stop:step])

    """
    Returns a 3-D list of joint angles for all frames, body parts and angle types.
    """

    def _calc_joint_angles(self) -> list:
        n_frames = len(self.timestamps)
        n_body_parts = len(self.body_parts)
        n_angle_types = 3
        bp = self.body_parts

        ls = acm.calc_angles_shoulder_left(self.positions, bp["LeftShoulder"], bp["RightShoulder"], bp["Torso"], bp["LeftElbow"])
        rs = acm.calc_angles_shoulder_right(self.positions, bp["RightShoulder"], bp["LeftShoulder"], bp["Torso"], bp["RightElbow"])
        lh = acm.calc_angles_hip_left(self.positions, bp["LeftHip"], bp["RightHip"], bp["Torso"], bp["LeftKnee"])
        rh = acm.calc_angles_hip_right(self.positions, bp["RightHip"], bp["LeftHip"], bp["Torso"], bp["RightKnee"])
        le = acm.calc_angles_elbow(self.positions, bp["LeftElbow"], bp["LeftShoulder"], bp["LeftWrist"])
        re = acm.calc_angles_elbow(self.positions, bp["RightElbow"], bp["RightShoulder"], bp["RightWrist"])
        lk = acm.calc_angles_knee(self.positions, bp["LeftKnee"], bp["LeftHip"], bp["LeftAnkle"])
        rk = acm.calc_angles_knee(self.positions, bp["RightKnee"], bp["RightHip"], bp["RightAnkle"])

        joint_angles = np.zeros((n_frames, n_body_parts, n_angle_types))
        for frame in range(0, n_frames):
            joint_angles[frame][bp["LeftShoulder"]] = ls[frame]
            joint_angles[frame][bp["RightShoulder"]] = rs[frame]
            joint_angles[frame][bp["LeftHip"]] = lh[frame]
            joint_angles[frame][bp["RightHip"]] = rh[frame]
            joint_angles[frame][bp["LeftElbow"]] = le[frame]
            joint_angles[frame][bp["RightElbow"]] = re[frame]
            joint_angles[frame][bp["LeftKnee"]] = lk[frame]
            joint_angles[frame][bp["RightKnee"]] = rk[frame]

        return joint_angles

    def get_positions_2d(self):
        """
        Returns the positions for all keypoints in 
        shape: (num_frames, num_bodyparts * xyz).
        """

        return np.reshape(self.positions, (self.positions.shape[0], -1))

    def merge(self, sequence: 'Sequence') -> 'Sequence':
        """
        Returns the merged two sequences.

        Raises ValueError if either the body_parts, the poseformat or the body_parts and keys within the joint_angles do not match!
        """
        if self.body_parts != sequence.body_parts:
            raise ValueError('body_parts of both sequences do not match!')
        if self.poseformat != sequence.poseformat:
            raise ValueError('poseformat of both sequences do not match!')

        # concatenate positions, timestamps and angles
        self.positions = np.concatenate((self.positions, sequence.positions), axis=0)
        self.timestamps = np.concatenate((self.timestamps, sequence.timestamps), axis=0)
        self.joint_angles = np.concatenate((self.joint_angles, sequence.joint_angles), axis=0)

        return self

    def get_pcs(self, num_components: int = 3):
        """
        Calculates n principal components for the tracked positions of this sequence
        """
        pca = PCA(n_components=num_components)
        xPCA = pca.fit_transform(self.get_positions_2d())
        return xPCA

    def visualise(self):
        pass

    def _filter_zero_frames(self, positions: list) -> list:
        """
        Returns a list of booleans. 
        Checks whether the sum of all coordinates for a frame is 0.0
            True -> keep this frame; 
            False -> remove this frame
        """
        bool_list = []
        for pos in positions:
            bool_list.append(np.sum(pos) != 0)

        return bool_list
