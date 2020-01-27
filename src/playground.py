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

mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)
filename = "data/sequences/191024_tracking/single/squat/user-2/191024__single__squat__user-2__1.json"
sequence = mocap_poseprocessor.load(filename)
pcs = get_pelvis_coordinate_system(sequence.positions[0][5], self.sequence.positions[frame][4], sequence.positions[0][11], sequence.positions[0][8])
print(pcs)
# pcs = get_pelvis_coordinate_system(np.array([[-1, 2, 2], [-1, 2, 2], [-1, 2, 2]]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
#                                    np.array([[3, 4, -5], [3, 4, -5], [3, 4, -5]]))
# get_pelvis_coordinate_system(pelvis=np.array([0, 0, 0]), hip_l=np.array([-1, -1, 0]), hip_r=np.array([1, 1, 0]))
print(f"{np.dot(pcs[0][1][0], pcs[0][1][1])}, {np.dot(pcs[0][1][1], pcs[0][1][2])}, {np.dot(pcs[0][1][0], pcs[0][1][2])}")
# for i in range(len(pcs)):
#     print(f"{np.dot(pcs[i][1][0], pcs[i][1][1])}, {np.dot(pcs[i][1][1], pcs[i][1][2])}, {np.dot(pcs[i][1][0], pcs[i][1][2])}")
