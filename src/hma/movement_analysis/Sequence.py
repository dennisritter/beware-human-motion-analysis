import numpy as np
from sklearn.decomposition import PCA
from hma.movement_analysis.PoseFormatEnum import PoseFormatEnum
from hma.movement_analysis import angle_calculations as acm


class Sequence:

    def __init__(self, body_parts: dict, positions: list, timestamps: list, poseformat: PoseFormatEnum, name: str = 'sequence', joint_angles: list = None):
        self.name = name
        self.poseformat = poseformat
        # Number, order and label of tracked body parts
        # Example: { "Head": 0, "RightShoulder": 1, ... }
        self.body_parts = body_parts

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
        self.joint_angles = self._calc_joint_angles() if joint_angles is None else joint_angles

        self.positions_2d = None  #self.get_positions_2d()

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

        joint_angles = []
        for idx, bp in enumerate(self.joint_angles):
            if bp is not None:
                bp_dict = {}
                for key in bp:
                    bp_dict[key] = self.joint_angles[idx][key][start:stop:step]
                    # print(self.joint_angles[idx][key][start:stop:step])
                joint_angles.append(bp_dict)
            else:
                joint_angles.append(None)

        return Sequence(self.body_parts, self.positions[start:stop:step], self.timestamps[start:stop:step], self.poseformat, self.name, joint_angles)

    def _calc_joint_angles(self):
        joint_angles = [None] * len(self.body_parts)

        bp = self.body_parts
        joint_angles[bp["LeftShoulder"]] = acm.calc_angles_shoulder_left(self.positions, bp["LeftShoulder"], bp["RightShoulder"], bp["Torso"], bp["LeftElbow"])
        joint_angles[bp["RightShoulder"]] = acm.calc_angles_shoulder_right(self.positions, bp["RightShoulder"], bp["LeftShoulder"], bp["Torso"], bp["RightElbow"])
        joint_angles[bp["LeftHip"]] = acm.calc_angles_hip_left(self.positions, bp["LeftHip"], bp["RightHip"], bp["Torso"], bp["LeftKnee"])
        joint_angles[bp["RightHip"]] = acm.calc_angles_hip_right(self.positions, bp["RightHip"], bp["LeftHip"], bp["Torso"], bp["RightKnee"])
        joint_angles[bp["LeftElbow"]] = acm.calc_angles_elbow(self.positions, bp["LeftElbow"], bp["LeftShoulder"], bp["LeftWrist"])
        joint_angles[bp["RightElbow"]] = acm.calc_angles_elbow(self.positions, bp["RightElbow"], bp["RightShoulder"], bp["RightWrist"])
        joint_angles[bp["LeftKnee"]] = acm.calc_angles_knee(self.positions, bp["LeftKnee"], bp["LeftHip"], bp["LeftAnkle"])
        joint_angles[bp["RightKnee"]] = acm.calc_angles_knee(self.positions, bp["RightKnee"], bp["RightHip"], bp["RightAnkle"])
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

        # iterate through joint angles and merge them
        for idx, bp in enumerate(self.joint_angles):
            if bp is not None:
                for key in bp:
                    if not key in sequence.joint_angles[idx]:
                        raise ValueError(f"Given sequence do not contain key: {key}, but the method called sequence does.")
                    self.joint_angles[idx][key].extend(sequence.joint_angles[idx][key])
            else:
                if sequence.joint_angles[idx] is not None:
                    raise ValueError(f"Given sequence has body_part, with index {idx}, which is not None. The method called sequence do not provide this body_part.")

        # concatenate positions and timestamps
        self.positions = np.concatenate((self.positions, sequence.positions), axis=0)
        self.timestamps = np.concatenate((self.timestamps, sequence.timestamps), axis=0)

        return self

    def get_pcs(self, num_components: int = 3):
        """
        Calculates n principal components for the tracked positions of this sequence
        """
        pca = PCA(n_components=num_components)
        xPCA = pca.fit_transform(self.get_positions_2d())
        return xPCA

    def _filter_zero_frames(self, positions: list) -> list:
        """
        Returns a list of booleans to filter numpy arrays by conditions. 
        Checks whether the sum of all coordinates for a frame is 0.0
            True -> keep this frame; 
            False -> remove this frame
        """
        bool_list = []
        for pos in positions:
            bool_list.append(np.sum(pos) != 0)

        return bool_list
