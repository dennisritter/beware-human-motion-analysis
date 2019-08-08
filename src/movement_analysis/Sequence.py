from .PoseFormatEnum import PoseFormatEnum
import numpy as np
from sklearn.decomposition import PCA


class Sequence:

    def __init__(self, body_parts: dict, positions: list, timestamps: list, poseformat: PoseFormatEnum, name: str = 'sequence'):
        self.name = name
        # Number, order and label of tracked body parts
        # Example: { "Head": 0, "RightShoulder": 1, ... }
        self.body_parts = body_parts
        # Defines positions of each bodypart
        # TODO: Change to: Time, Bodypart, xyz
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
        # Timestamps for when the positions have been tracked
        # Example: [<someTimestamp1>, <someTimestamp2>, <someTimestamp3>, ...]
        self.timestamps = np.array(timestamps)
        """ We need this at some point, maybe
        # Skeleton connections between bodyparts
        # Example: [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"], ["RElbow", "RWrist"], ...],
        self.body_pairs = body_pairs
        """
        self.positions_2d = self.get_positions_2d()

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
