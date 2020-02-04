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

# filename = "data/sequences/191024_tracking/single/squat/user-2/191024__single__squat__user-2__1.json"
filename = "data/sequences/191024_tracking/single/overhead_press/user-2/191024__single__overhead_press__user-2__1.json"
sequence = Sequence.from_mocap_file(filename)[70]

sequence._fill_scenegraph(sequence.scene_graph, sequence.positions)

sv = SkeletonVisualiser(sequence)
sv.show()

# print(sequence.scene_graph.nodes(data=True))
# print(sequence.scene_graph.edges(data=True))

# ROT VECTOR: [ 0.81863615 -0.21747889 -0.53154284]
# array([ 0.98696784, -0.08897527, -0.13408166])