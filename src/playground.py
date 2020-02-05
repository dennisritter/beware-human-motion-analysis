import json
import tslearn.metrics as ts
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from hma.movement_analysis.models.sequence import Sequence
from hma.movement_analysis.helpers import reformat_angles_dtw
from hma.movement_analysis.exercise_evaluator import ExerciseEvaluator
from hma.movement_analysis.skeleton_visualiser import SkeletonVisualiser
from hma.movement_analysis.helpers import hierarchy_pos
from hma.movement_analysis.helpers import draw_scenegraph
from hma.movement_analysis.transformations import get_pelvis_coordinate_system
from hma.movement_analysis.transformations import get_cs_projection_transformation
import hma.movement_analysis.angle_representations as ar

# filename = "data/sequences/191024_tracking/single/squat/user-2/191024__single__squat__user-2__1.json"
# filename = "data/sequences/191024_tracking/single/biceps_curl_left/user-2/191024__single__biceps_curl_left__user-2__1.json"
# filename = "data/sequences/191024_tracking/single/biceps_curl_right/user-2/191024__single__biceps_curl_right__user-2__1.json"
# filename = "data/sequences/191024_tracking/single/triceps_extension_left/user-2/191024__single__triceps_extension_left__user-2__1.json"
# filename = "data/sequences/191024_tracking/single/triceps_extension_right/user-2/191024__single__triceps_extension_right__user-2__1.json"
# filename = "data/sequences/191024_tracking/single/overhead_press/user-2/191024__single__overhead_press__user-2__1.json"
filename = "data/sequences/191024_tracking/single/lunge_left/user-2/191024__single__lunge_left__user-2__1.json"
frame = 50
sequence = Sequence.from_mocap_file(filename)
sequence._fill_scenegraph(sequence.scene_graph, sequence.positions)

non_ball_joints = ['elbow_l', 'elbow_r', 'knee_l', 'knee_r']
ball_joints = ['shoulder_l', 'shoulder_r', 'hip_l', 'hip_r']

for node in sequence.scene_graph.nodes:
    if sequence.scene_graph.nodes[node]['angles']:
        spherical_angles = np.round(sequence.joint_angles[frame][sequence.body_parts[node]], 2)
        euler_angles_xyz = np.round(sequence.scene_graph.nodes[node]['angles'][frame]['euler_xyz'], 2)
        euler_angles_yxz = np.round(sequence.scene_graph.nodes[node]['angles'][frame]['euler_yxz'], 2)
        euler_angles_zxz = np.round(sequence.scene_graph.nodes[node]['angles'][frame]['euler_zxz'], 2)

        print(node)
        print(f"SPHER: F:{spherical_angles[0]} A:{spherical_angles[1]} I: None")
        print(f"XYZ  : F:{euler_angles_xyz[0]} A:{euler_angles_xyz[1]} I:{euler_angles_xyz[2]}")
        print(f"YXZ  : F:{euler_angles_yxz[0]} A:{euler_angles_yxz[1]} I:{euler_angles_yxz[2]}")
        print(f"ZXZ  : F:{euler_angles_zxz[0]} A:{euler_angles_zxz[1]} I:{euler_angles_zxz[2]}")
        print('-----')

for node in sequence.scene_graph.nodes:
    if node in non_ball_joints:
        print("NON BALL")
        print(f"{node}: {ar.medical_from_euler('zxz', sequence.scene_graph.nodes[node]['angles'][frame]['euler_zxz'], node)}")
    if node in ball_joints:
        print("BALL")
        print(f"{node} XYZ: {ar.medical_from_euler('xyz',sequence.scene_graph.nodes[node]['angles'][frame]['euler_xyz'] , node)}")
        print(f"{node} YXZ: {ar.medical_from_euler('yxz',sequence.scene_graph.nodes[node]['angles'][frame]['euler_yxz'] , node)}")

# sv = SkeletonVisualiser(sequence)
# sv.show()
