"""Contains the code for the sequence model including the scenegraph and angle computation."""
import json
import networkx as nx
import numpy as np
from scipy.spatial.transform import Rotation
from numpy.linalg import inv
from hma.movement_analysis import angle_calculations as acm
import hma.movement_analysis.transformations as transformations
import hma.movement_analysis.angle_representations as ar

# TODO: Handle expensive calculations with batches instead of loops to increase performance
# TODO: Implement Lazy Loading for props that are expensive to calculate (e.g. joint angles, Scene_graph data)
# TODO: Test from_json and to_json methods for whether they (de)serialize the scene_graph properly
# TODO: Consider outsourcing medical joint_angle calculations and attribute to another module/script/class
#       Maybe a place to handle medical related stuff would be nice to get a clean seperation.  
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
        self.positions = self._get_pelvis_cs_positions(np.array(positions)[zero_frames_filter_list])

        # Timestamps for when the positions have been tracked
        # Example: [<someTimestamp1>, <someTimestamp2>, <someTimestamp3>, ...]
        self.timestamps = np.array(timestamps)[zero_frames_filter_list]
        
        # A directed graph that defines the hierarchy between human body parts
        self.scene_graph = nx.DiGraph([
            ("pelvis", "torso"),
            ("torso", "neck"),
            ("neck", "head"),
            ("neck", "shoulder_l"),
            ("shoulder_l", "elbow_l"),
            ("elbow_l", "wrist_l"),
            ("neck", "shoulder_r"),
            ("shoulder_r", "elbow_r"),
            ("elbow_r", "wrist_r"),
            ("pelvis", "hip_l"),
            ("hip_l", "knee_l"),
            ("knee_l", "ankle_l"),
            ("pelvis", "hip_r"),
            ("hip_r", "knee_r"),
            ("knee_r", "ankle_r"),
        ]) if scene_graph is None else scene_graph.copy()
        self._fill_scenegraph(self.scene_graph, self.positions)

        # Stores joint angles for each frame
        # NOTE: Body part indices are the indices stored in self.body_parts.
        # NOTE: If angles have been computed, the stored value is a dictionary with at least one key "flexion_extension"
        #       and a "abduction_adduction" key for ball joints.
        # NOTE: If no angles have been computed for a particular joint, the stored value is None.
        self.joint_angles = self._calc_joint_angles() if joint_angles is None else np.array(joint_angles)

    def __len__(self) -> int:
        return len(self.positions)

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

        # Slice All data lists stored in the scene_graphs nodes and edges
        scene_graph = self.scene_graph.copy()
        for node in scene_graph.nodes:
            for data_list in scene_graph.nodes[node].keys():
                if data_list:
                    scene_graph.nodes[node][data_list] = scene_graph.nodes[node][data_list][start:stop:step]
        for e1, e2 in scene_graph.edges:
            for data_list in scene_graph[e1][e2].keys():
                if data_list:
                    scene_graph[e1][e2][data_list] = scene_graph[e1][e2][data_list][start:stop:step]
        self.scene_graph = scene_graph
        
        return Sequence(self.body_parts, self.positions[start:stop:step], self.timestamps[start:stop:step], self.name, self.joint_angles[start:stop:step],
                        self.scene_graph)

    def _get_pelvis_cs_positions(self, positions):
        """Transforms all points in positions parameter so they are relative to the pelvis. X-Axis = right, Y-Axis = front, Z-Axis = up. """
        transformed_positions = []
        for i, frame in enumerate(positions):
            transformed_positions.append([])
            pelvis_cs = transformations.get_pelvis_coordinate_system(positions[i][self.body_parts["pelvis"]], positions[i][self.body_parts["torso"]],
                                                                     positions[i][self.body_parts["hip_l"]], positions[i][self.body_parts["hip_r"]])
            M = transformations.get_cs_projection_transformation(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                                                                 np.array([pelvis_cs[0][0], pelvis_cs[0][1][0], pelvis_cs[0][1][1], pelvis_cs[0][1][2]]))
            for _, pos in enumerate(frame):
                transformed_positions[i].append((M @ np.append(pos, 1))[:3])

        return np.array(transformed_positions)

    def _fill_scenegraph(self, scene_graph, positions):
        # Find Scene Graph Root Node
        root_node = None
        nodes = list(scene_graph.nodes)
        # Find root_node
        for node in nodes:
            predecessors = list(scene_graph.predecessors(node))
            if not predecessors:
                root_node = node
                break

        # Predefine node data attributes to store data for each frame of the sequence
        for node in scene_graph.nodes:
            scene_graph.nodes[node]['coordinate_system'] = []
            scene_graph.nodes[node]['angles'] = []

        # Predefine edge data lists to store data for each frame of the sequence
        for n1, n2 in scene_graph.edges:
            scene_graph[n1][n2]['transformation'] = []

        for frame, _ in enumerate(positions):
            # Start recursive function with root node in our directed scene_graph
            self._calc_scene_graph_transformations(scene_graph, root_node, root_node, positions, frame)

    def _calc_scene_graph_transformations(self, scene_graph, node, root_node, positions, frame):
        successors = list(scene_graph.successors(node))

        # Root Node handling
        if node == root_node:
            # The node with no predecessors is the root node, so add the initial coordinate system to its node data
            node_data = scene_graph.nodes[node]
            node_data['coordinate_system'].append({
                "origin": np.array([0, 0, 0]),
                "x_axis": np.array([1, 0, 0]),
                "y_axis": np.array([0, 1, 0]),
                "z_axis": np.array([0, 0, 1])
            })
            # Repeat function recursive for each child node of the root node
            for child_node in successors:
                self._calc_scene_graph_transformations(scene_graph, child_node, root_node, positions, frame)
            return

        node_data = scene_graph.nodes[node]
        node_pos = positions[frame, self.body_parts[node]]
        predecessors = list(scene_graph.predecessors(node))

        parent_node = predecessors[0]
        parent_pos = positions[frame, self.body_parts[parent_node]]
        parent_cs = scene_graph.nodes[parent_node]['coordinate_system'][frame]

        T = transformations.translation_matrix_4x4(node_pos - parent_pos)
        # If No successors or more than one successor present, add translation only to (parent_node, node) edge.
        # TODO: How can we circumvent to check whether node == "torso" ?
        if len(successors) != 1 or node == "torso":
            scene_graph[parent_node][node]['transformation'].append(T)
            node_data['coordinate_system'].append({
                "origin": (T @ np.append(parent_cs['origin'], 1))[:3],
                "x_axis": transformations.norm((np.append(parent_cs['x_axis'], 1))[:3]),
                "y_axis": transformations.norm((np.append(parent_cs['y_axis'], 1))[:3]),
                "z_axis": transformations.norm((np.append(parent_cs['z_axis'], 1))[:3])
            })
        elif len(successors) == 1:
            child_node = successors[0]
            child_pos = positions[frame, self.body_parts[child_node]]

            # --Determine Joint Rotation--
            scene_graph.nodes[node]['angles'].append({})
            # Get parent coordinate system Z-axis as reference for nodes' joint rotation
            parent_cs_z = parent_cs['z_axis']
            # Get direction vector from node to child to determine rotation of nodes' joint
            node_to_child_node = transformations.norm(child_pos - node_pos)
            # Joint angles as 4x4 homogenious rotation matrix
            R_parent_to_node = transformations.get_rotation(-parent_cs_z, node_to_child_node)

            # Get Euler Sequences to be able to determine medical joint angles
            # NOTE: Scipy from_dcm() function has been renamed to 'as_matrix()' in scipy=1.4.*
            #       lates version for win64 is still 1.3.* ; Consider updating scipy dependency when 1.4.* is available for win64.
            euler_angles_xyz = Rotation.from_dcm(R_parent_to_node[:3, :3]).as_euler('XYZ', degrees=True)
            euler_angles_yxz = Rotation.from_dcm(R_parent_to_node[:3, :3]).as_euler('YXZ', degrees=True)
            euler_angles_zxz = Rotation.from_dcm(R_parent_to_node[:3, :3]).as_euler('ZXZ', degrees=True)

            # Store transformation from parent to current node in corresponding edge
            scene_graph[parent_node][node]['transformation'].append(T @ R_parent_to_node)
            # Store Rotation Matrix
            scene_graph.nodes[node]['angles'][frame]['rotation_matrix'] = np.array(R_parent_to_node)
            # Store Euler Sequences
            scene_graph.nodes[node]['angles'][frame]['euler_xyz'] = euler_angles_xyz
            scene_graph.nodes[node]['angles'][frame]['euler_yxz'] = euler_angles_yxz
            scene_graph.nodes[node]['angles'][frame]['euler_zxz'] = euler_angles_zxz
            # Store the nodes coordinate system
            scene_graph.nodes[node]['coordinate_system'].append({
                "origin": (T @ np.append(parent_cs['origin'], 1))[:3],
                "x_axis": transformations.norm((R_parent_to_node @ np.append(parent_cs['x_axis'], 1))[:3]),
                "y_axis": transformations.norm((R_parent_to_node @ np.append(parent_cs['y_axis'], 1))[:3]),
                "z_axis": transformations.norm((R_parent_to_node @ np.append(parent_cs['z_axis'], 1))[:3])
            })

        # Repeat procedure if successors present
        for child_node in successors:
            self._calc_scene_graph_transformations(scene_graph, child_node, root_node, positions, frame)

        return

    def to_json(self) -> str:
        """Returns the sequence instance as a json-formatted string."""
        json_dict = {
            'name': self.name,
            'body_parts': self.body_parts,
            'positions': self.positions.tolist(),
            'timestamps': self.timestamps.tolist(),
            'joint_angles': self.joint_angles.tolist(),
            'scene_graph': nx.readwrite.json_graph.adjacency_data(self.scene_graph)
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
        # TODO: Test if this works properly
        if 'scene_graph' in json_dict.keys():
            json_dict['scene_graph'] = nx.readwrite.json_graph.adjacency_data(json_dict['scene_graph'])
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
            positions = np.array(mocap_sequence["frames"])
            timestamps = np.array(mocap_sequence["timestamps"])

            # reshape positions to 3d array
            positions = np.reshape(positions, (np.shape(positions)[0], int(np.shape(positions)[1] / 3), 3))

            # Center Positions by subtracting the mean of each coordinate
            positions[:, :, 0] -= np.mean(positions[:, :, 0])
            positions[:, :, 1] -= np.mean(positions[:, :, 1])
            positions[:, :, 2] -= np.mean(positions[:, :, 2])

            # Adjust MoCap data to our target Coordinate System
            # X_mocap = Left    ->  X_hma = Right   -->     Flip X-Axis
            # Y_mocap = Up      ->  Y_hma = Front   -->     Switch Y and Z; Flip (new) Y-Axis
            # Z_mocap = Back    ->  Z_hma = Up      -->     Switch Y and Z

            # Switch Y and Z axis.
            # In Mocap Y points up and Z to the back -> We want Z to point up and Y to the front,
            y_positions_mocap = positions[:, :, 1].copy()
            z_positions_mocap = positions[:, :, 2].copy()
            positions[:, :, 1] = z_positions_mocap
            positions[:, :, 2] = y_positions_mocap
            # MoCap coordinate system is left handed -> flip x-axis to adjust data for right handed coordinate system
            positions[:, :, 0] *= -1
            # Flip Y-Axis
            # MoCap Z-Axis (our Y-Axis now) points "behind" the trainee, but we want it to point "forward"
            positions[:, :, 1] *= -1

            # The target Body Part format
            body_parts = {
                "head": 0,
                "neck": 1,
                "shoulder_l": 2,
                "shoulder_r": 3,
                "elbow_l": 4,
                "elbow_r": 5,
                "wrist_l": 6,
                "wrist_r": 7,
                "torso": 8,
                "pelvis": 9,
                "hip_l": 10,
                "hip_r": 11,
                "knee_l": 12,
                "knee_r": 13,
                "ankle_l": 14,
                "ankle_r": 15,
            }

            # Change body part indices according to the target body part format
            positions_mocap = positions.copy()
            positions[:, 0, :] = positions_mocap[:, 15, :]  # "head": 0
            positions[:, 1, :] = positions_mocap[:, 3, :]  # "neck": 1
            positions[:, 2, :] = positions_mocap[:, 2, :]  # "shoulder_l": 2
            positions[:, 3, :] = positions_mocap[:, 14, :]  # "shoulder_r": 3
            positions[:, 4, :] = positions_mocap[:, 1, :]  # "elbow_l": 4
            positions[:, 5, :] = positions_mocap[:, 13, :]  # "elbow_r": 5
            positions[:, 6, :] = positions_mocap[:, 0, :]  # "wrist_l": 6
            positions[:, 7, :] = positions_mocap[:, 12, :]  # "wrist_r": 7
            positions[:, 8, :] = positions_mocap[:, 4, :]  # "torso": 8
            positions[:, 9, :] = positions_mocap[:, 5, :]  # "pelvis": 9
            positions[:, 10, :] = positions_mocap[:, 8, :]  # "hip_l": 10
            positions[:, 11, :] = positions_mocap[:, 11, :]  # "hip_r": 11
            positions[:, 12, :] = positions_mocap[:, 7, :]  # "knee_l": 12
            positions[:, 13, :] = positions_mocap[:, 10, :]  # "knee_r": 13
            positions[:, 14, :] = positions_mocap[:, 6, :]  # "ankle_l": 14
            positions[:, 15, :] = positions_mocap[:, 9, :]  # "ankle_r": 15

            return cls(body_parts, positions, timestamps, name=name)

    def _calc_joint_angles(self) -> np.ndarray:
        """Returns a 3-D list of joint angles for all frames, body parts and angle types."""
        n_frames = len(self.positions)
        n_body_parts = len(self.scene_graph.nodes)
        n_angle_types = 3
        bp = self.body_parts

        joint_angles = np.full((n_frames, n_body_parts, n_angle_types), None)
        ball_joints = ['shoulder_l', 'shoulder_r', 'hip_l', 'hip_r']
        non_ball_joints = ['elbow_l', 'elbow_r', 'knee_l', 'knee_r']


        for node in self.scene_graph.nodes:
            if 'angles' in self.scene_graph.nodes[node].keys():
                for frame, angle_types in enumerate(self.scene_graph.nodes[node]['angles']):
                    if node in ball_joints:
                        joint_angles[frame][bp[node]] = ar.medical_from_euler('xyz', angle_types['euler_xyz'], node)
                    elif node in non_ball_joints:
                        joint_angles[frame][bp[node]] = ar.medical_from_euler('zxz', angle_types['euler_zxz'], node)
                    else:
                        joint_angles[frame][bp[node]] = np.array([None, None, None])
                        
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

        # Copy the given sequence to not change it implicitly
        sequence = sequence[:]

        # concatenate positions, timestamps and angles
        self.positions = np.concatenate((self.positions, sequence.positions), axis=0)
        self.timestamps = np.concatenate((self.timestamps, sequence.timestamps), axis=0)
        
        # Concatenate scene_graph data lists
        for node in self.scene_graph.nodes:
            for data_list in self.scene_graph.nodes[node].keys():
                if data_list and data_list in sequence.scene_graph.nodes[node]:
                    self.scene_graph.nodes[node][data_list] += sequence.scene_graph.nodes[node][data_list]
                # If appending sequence has no data in scene_graph, add it beforehand
                elif data_list and data_list not in sequence.scene_graph.nodes[node]:
                    sequence._fill_scenegraph(sequence.scene_graph, sequence.positions)
                    self.scene_graph.nodes[node][data_list] += sequence.scene_graph.nodes[node][data_list]

        for e1, e2 in self.scene_graph.edges:
            for data_list in self.scene_graph[e1][e2].keys():
                if data_list and data_list in sequence.scene_graph[e1][e2]:
                    self.scene_graph[e1][e2][data_list] += sequence.scene_graph[e1][e2][data_list]
                # If appending sequence has no data in scene_graph, add it beforehand
                elif data_list and data_list not in sequence.scene_graph[e1][e2]:
                    sequence._fill_scenegraph(sequence.scene_graph, sequence.positions)
                    self.scene_graph[e1][e2][data_list] += sequence.scene_graph[e1][e2][data_list]

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
