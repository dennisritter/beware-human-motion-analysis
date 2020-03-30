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

f0 = "data/sequences/mka_sequences/hannah_overheadpress_sitting_0.json"
f1 = "data/sequences/mka_sequences/hannah_overheadpress_sitting_1.json"
f2 = "data/sequences/mka_sequences/hannah_overheadpress_sitting_2.json"
f3 = "data/sequences/mka_sequences/hannah_overheadpress_sitting_3.json"
f4 = "data/sequences/mka_sequences/hannah_overheadpress_sitting_4.json"
f5 = "data/sequences/mka_sequences/hannah_overheadpress_sitting_5.json"
f6 = "data/sequences/mka_sequences/hannah_overheadpress_sitting_6.json"
f7 = "data/sequences/mka_sequences/hannah_overheadpress_sitting_7.json"
f8 = "data/sequences/mka_sequences/hannah_overheadpress_sitting_8.json"
f9 = "data/sequences/mka_sequences/hannah_overheadpress_sitting_9.json"
f10 = "data/sequences/mka_sequences/hannah_overheadpress_sitting_10.json"
sequences = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]

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

with open('data/sequences/mka_sequences/hannah_overheadpress_sitting_multi_11.json', 'w') as outfile:
    json.dump(json_result, outfile)