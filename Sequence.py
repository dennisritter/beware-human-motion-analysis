import numpy


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
        self.positions = numpy.array(positions)
        # Timestamps for when the positions have been tracked
        # Example: [<someTimestamp1>, <someTimestamp2>, <someTimestamp3>, ...]
        self.timestamps = numpy.array(timestamps)
        """ We need this at some point, maybe
        # Skeleton connections between bodyparts
        # Example: [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"], ["RElbow", "RWrist"], ...],
        self.body_pairs = body_pairs
        """
