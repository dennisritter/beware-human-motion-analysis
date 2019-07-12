import numpy
from sklearn.decomposition import PCA


class Sequence:

    def __init__(self, body_parts: list, positions: list, timestamps: list, name: str = 'sequence'):
        self.name = name
        # Number, order and label of tracked body parts
        # Example: ["Head", "Neck", "RShoulder", "RElbow", ...]
        self.body_parts = numpy.array(body_parts)
        # Defines positions of each bodypart
        # 1. Dimension = Bodypart
        # 2. Dimension = Time
        # 3. Dimension = x, y, z
        # Example: [
        #             [[part-i.x, part-i.y, part-i.z], [part-i.x, part-i.y, part-i.z], [part-i.x, part-i.y, part-i.z]],
        #             [[part-i+1.x, part-i+1.y, part-i+1.z], [part-i+1.x, part-i+1.y, part-i+1.z], [part-i+1.x, part-i+1.y, part-i+1.z]],
        #             ...
        #          ]
        # shape: (num_body_parts, num_keypoints, xyz)
        self.positions = numpy.array(positions)
        # Timestamps for when the positions have been tracked
        # Example: [<someTimestamp1>, <someTimestamp2>, <someTimestamp3>, ...]
        self.timestamps = numpy.array(timestamps)
        """ We need this at some point, maybe
        # Skeleton connections between bodyparts
        # Example: [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"], ["RElbow", "RWrist"], ...],
        self.body_pairs = body_pairs
        """
        self.positions_2d = self.get_positions_2d()

    def get_positions_2d(self):
        """
        Returns the positions for all keypoints in 
        shape: (num_keypoints, num_bodyparts * xyz).
        """
        return numpy.ndarray.flatten(self.positions).reshape(self.positions.shape[1], -1)

    def get_pcs(self, num_components: int = 3):
        """
        Calculates n principal components for the tracked positions of this sequence
        """
        pca = PCA(n_components=num_components)
        xPCA = pca.fit_transform(self.get_positions_2d())
        return xPCA
