import numpy as np
from sklearn.decomposition import PCA
from .PoseFormatEnum import PoseFormatEnum
from . import angle_calculations as acm


class Sequence:

    def __init__(self, body_parts: dict, positions: list, timestamps: list, poseformat: PoseFormatEnum, name: str = 'sequence'):
        self.name = name
        # Number, order and label of tracked body parts
        # Example: { "Head": 0, "RightShoulder": 1, ... }
        self.body_parts = body_parts
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
        self.positions = np.array(positions)

        # Stores joint angles for each frame
        # NOTE: Body part indices are the indices stored in self.body_parts.
        # NOTE: If angles have been computed, the stored value is a dictionary with at least one key "flexion_extension"
        #       and a "abduction_adduction" key for ball joints.
        # NOTE: If no angles have been computed for a particular joint, the stored value is None.
        self.joint_angles = self._calc_joint_angles()
        # Timestamps for when the positions have been tracked
        # Example: [<someTimestamp1>, <someTimestamp2>, <someTimestamp3>, ...]
        self.timestamps = np.array(timestamps)
        """ We need this at some point, maybe
        # Skeleton connections between bodyparts
        # Example: [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"], ["RElbow", "RWrist"], ...],
        self.body_pairs = body_pairs
        """
        self.positions_2d = self.get_positions_2d()

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

    def get_pcs(self, num_components: int = 3):
        """
        Calculates n principal components for the tracked positions of this sequence
        """
        pca = PCA(n_components=num_components)
        xPCA = pca.fit_transform(self.get_positions_2d())
        return xPCA
