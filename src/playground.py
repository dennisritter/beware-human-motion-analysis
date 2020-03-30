import json
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
import hma.movement_analysis.medical_joint_angles as ma
from hma.movement_analysis.models.exercise import Exercise
from hma.movement_analysis.enums.angle_types import AngleTypes
import time
import hma.movement_analysis.transformations as transformations
import sklearn.preprocessing as preprocessing
import plotly.graph_objects as go
"""
f0 = "data/sequences/mka_sequences/hannah_squat_0.json"
f1 = "data/sequences/mka_sequences/hannah_squat_1.json"
f2 = "data/sequences/mka_sequences/hannah_squat_2.json"
f3 = "data/sequences/mka_sequences/hannah_squat_3.json"
f4 = "data/sequences/mka_sequences/hannah_squat_4.json"
f5 = "data/sequences/mka_sequences/hannah_squat_5.json"
f6 = "data/sequences/mka_sequences/hannah_squat_6.json"
f7 = "data/sequences/mka_sequences/hannah_squat_7.json"
f8 = "data/sequences/mka_sequences/hannah_squat_8.json"
f9 = "data/sequences/mka_sequences/hannah_squat_9.json"
f10 = "data/sequences/mka_sequences/hannah_squat_10.json"
f11 = "data/sequences/mka_sequences/hannah_squat_11.json"
sequences = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]

json_result = {}
with open(f0, 'r') as sequence_file:
    json_str = sequence_file.read()
    json_data = json.loads(json_str)
    json_result = json_data

for seq in sequences:
    with open(seq, 'r') as sequence_file:
        json_str = sequence_file.read()
        json_data = json.loads(json_str)
        positions = json_data["positions"]
        timestamps = json_data["timestamps"]
        json_result['positions'].extend(positions)
        json_result['timestamps'].extend(timestamps)

with open('data/sequences/mka_sequences/hannah_squat_multi_11', 'w') as outfile:
    json.dump(json_result, outfile)
"""

# exercise = "data/exercises/squat.json"
exercise = "data/exercises/overhead-press.json"
# mir_filename = "data/sequences/191024_tracking/single/squat/user-4/191024__single__squat__user-4__1.json"
# # filename = "data/sequences/191024_tracking/single/overhead_press/user-6/191024__single__overhead_press__user-6__4.json"
mka_filename = "data/sequences/mka_sequences/hannah_overheadpress_sitting_multi_11.json"
# mka_filename = "data/sequences/mka_sequences/hannah_overheadpress_sitting_5.json"

### MKA
seq = Sequence.from_mka_file(mka_filename)
# ### MIR
# # seq = Sequence.from_mir_file(mir_filename)
# ### General
# sv = SkeletonVisualiser(seq)
# sv.show()

exe = Exercise.from_file(exercise)
EE = ExerciseEvaluator(exe, seq)
iterations = EE.find_iteration_keypoints(plot=True)
# print(iterations)
# result = EE.evaluate(iterations[0, 1])
# print(result[30])