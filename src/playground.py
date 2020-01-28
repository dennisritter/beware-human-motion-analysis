import json
import tslearn.metrics as ts
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from hma.movement_analysis.pose_processor import PoseProcessor
from hma.movement_analysis.enums.pose_format_enum import PoseFormatEnum
from hma.movement_analysis import exercise_loader
from hma.movement_analysis.helpers import reformat_angles_dtw
from hma.movement_analysis.exercise_evaluator import ExerciseEvaluator
from hma.movement_analysis.skeleton_visualiser import SkeletonVisualiser
from hma.movement_analysis.helpers import hierarchy_pos
from hma.movement_analysis.helpers import draw_scenegraph
from hma.movement_analysis.transformations import get_pelvis_coordinate_system
from hma.movement_analysis.transformations import get_cs_projection_tranformation

mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)
filename = "data/sequences/191024_tracking/single/squat/user-2/191024__single__squat__user-2__1.json"
sequence = mocap_poseprocessor.load(filename)
pelvis_cs = get_pelvis_coordinate_system(sequence.positions[0][9], sequence.positions[0][8], sequence.positions[0][10], sequence.positions[0][11])
print(pelvis_cs)
# print(f"{np.dot(pcs[0][1][0], pcs[0][1][1])}, {np.dot(pcs[0][1][1], pcs[0][1][2])}, {np.dot(pcs[0][1][0], pcs[0][1][2])}")

M = get_cs_projection_tranformation(
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    np.array([pelvis_cs[0][0], pelvis_cs[0][1][0], pelvis_cs[0][1][1], pelvis_cs[0][1][2]])
)
print(f"M: {M}")
# Transform all current positions from camera coords into pelvis coords
# TODO: Rounding issue. Only use numpy arrays?
pelvis_positions = []
for i, frame in enumerate(sequence.positions):
    pelvis_positions.append([])
    for j, pos in enumerate(frame):
        pelvis_positions[i].append(np.matmul(M, np.append(pos, 1))[:3])
pelvis_positions = np.array(pelvis_positions)

print(sequence.positions[0, 9])
print(pelvis_positions[0, 9])
