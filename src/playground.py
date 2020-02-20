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
import plotly.graph_objects as go

# exercise = "data/exercises/squat.json"
exercise = "data/exercises/overhead-press.json"
# filename = "data/sequences/191024_tracking/single/squat/user-2/191024__single__squat__user-2__1.json"
# filename = "data/sequences/191024_tracking/single/squat/user-4/191024__single__squat__user-4__1.json"
# filename = "data/sequences/191024_tracking/single/biceps_curl_left/user-2/191024__single__biceps_curl_left__user-2__1.json"
# filename = "data/sequences/191024_tracking/single/biceps_curl_right/user-2/191024__single__biceps_curl_right__user-2__1.json"
# filename = "data/sequences/191024_tracking/single/triceps_extension_left/user-2/191024__single__triceps_extension_left__user-2__1.json"
# filename = "data/sequences/191024_tracking/single/triceps_extension_right/user-2/191024__single__triceps_extension_right__user-2__1.json"
filename = "data/sequences/191024_tracking/single/overhead_press/user-3/191024__single__overhead_press__user-3__6.json"
# filename = "data/sequences/191024_tracking/single/overhead_press/user-5/191024__single__overhead_press__user-5__10.json"
# filename = "data/sequences/191024_tracking/single/lunge_left/user-2/191024__single__lunge_left__user-2__1.json"
# filename = "data/sequences/test/100_frames.json"
# f1 = "data/sequences/test/a.json"
# f2 = "data/sequences/test/b.json"

s = Sequence.from_mocap_file(filename)
bps = s.body_parts
# print(s.joint_angles[:, bps['shoulder_l']])

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(s)), y=s.joint_angles[:, bps['shoulder_l'], 0], name='Flex'))
fig.add_trace(go.Scatter(x=np.arange(len(s)), y=s.joint_angles[:, bps['shoulder_l'], 1], name='Abd'))
# print(s.positions[0, bps['elbow_l'], 0])
# d_x = np.sum(np.absolute(np.absolute(s.positions[:, bps['elbow_l'], 0]) - abs(s.positions[0, bps['elbow_l'], 0])))
# d_y = np.sum(np.absolute(np.absolute(s.positions[:, bps['elbow_l'], 1]) - abs(s.positions[0, bps['elbow_l'], 1])))
# fig.add_trace(go.Scatter(x=np.arange(len(s)), y=d_x, name='x_abs_sum (high -> abd?)'))
# fig.add_trace(go.Scatter(x=np.arange(len(s)), y=d_y, name='y_abs_sum (high -> flex?)'))
# print(f"D_X: {np.sum(d_x)}")
# print(f"D_Y: {np.sum(d_y)}")

fig.show()

# s_sv = s[::10]
# sv = SkeletonVisualiser(s_sv)
# sv.show()

exe = Exercise.from_file(exercise)
EE = ExerciseEvaluator(exe, s)
iterations = EE.find_iteration_keypoints(plot=True)
print(iterations)