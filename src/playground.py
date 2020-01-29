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

filename = "data/sequences/191024_tracking/single/squat/user-2/191024__single__squat__user-2__1.json"
sequence = Sequence.from_mocap_file(filename)
# pelvis_cs = get_pelvis_coordinate_system(sequence.positions[0][9], sequence.positions[0][8], sequence.positions[0][10], sequence.positions[0][11])
# print(pelvis_cs)
# # print(f"{np.dot(pcs[0][1][0], pcs[0][1][1])}, {np.dot(pcs[0][1][1], pcs[0][1][2])}, {np.dot(pcs[0][1][0], pcs[0][1][2])}")

# M = get_cs_projection_transformation(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
#                                      np.array([pelvis_cs[0][0], pelvis_cs[0][1][0], pelvis_cs[0][1][1], pelvis_cs[0][1][2]]))

# # Transform all current positions from camera coords into pelvis coords
# pelvis_positions = []
# for i, frame in enumerate(sequence.positions):
#     pelvis_positions.append([])
#     for j, pos in enumerate(frame):
#         pelvis_positions[i].append((M @ np.append(pos, 1))[:3])
# pelvis_positions = np.array(pelvis_positions)

for frame in sequence.positions:
    print(frame[9])
# print(pelvis_positions[0, 9])
