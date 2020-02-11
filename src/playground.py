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
from hma.movement_analysis.models.exercise import Exercise
import time
import hma.movement_analysis.transformations as transformations
import sklearn.preprocessing as preprocessing

filename = "data/sequences/191024_tracking/single/squat/user-2/191024__single__squat__user-2__1.json"
# filename = "data/sequences/191024_tracking/single/biceps_curl_left/user-2/191024__single__biceps_curl_left__user-2__1.json"
# filename = "data/sequences/191024_tracking/single/biceps_curl_right/user-2/191024__single__biceps_curl_right__user-2__1.json"
# filename = "data/sequences/191024_tracking/single/triceps_extension_left/user-2/191024__single__triceps_extension_left__user-2__1.json"
# filename = "data/sequences/191024_tracking/single/triceps_extension_right/user-2/191024__single__triceps_extension_right__user-2__1.json"
# filename = "data/sequences/191024_tracking/single/overhead_press/user-2/191024__single__overhead_press__user-2__1.json"
# filename = "data/sequences/191024_tracking/single/lunge_left/user-2/191024__single__lunge_left__user-2__1.json"
v1 = [1, 2, 3]
v2 = [3, 2, 1]
v3 = [5, 5, 5]
v4 = [0, 0, 0]
vs = np.array([v1, v2, v3, v4])

print(transformations.norm(v1))
print(transformations.norm(v2))
print(transformations.norm(v3))
print("--------------------------------")
nvs = transformations.norm_batch(v1)
print(nvs)
