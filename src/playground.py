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

exercise = "data/exercises/squat.json"
# exercise = "data/exercises/overhead-press.json"
# filename = "data/sequences/191024_tracking/single/squat/user-2/191024__single__squat__user-2__1.json"
# filename = "data/sequences/191024_tracking/single/overhead_press/user-6/191024__single__overhead_press__user-6__4.json"
filename = "data/sequences/mka_sequences/hannah_squat_1.json"

### Check Sequence angles ang iterations
seq = Sequence.from_mka_file(filename)
sv = SkeletonVisualiser(seq)
sv.show()
# exe = Exercise.from_file(exercise)
# EE = ExerciseEvaluator(exe, seq)
# iterations = EE.find_iteration_keypoints(plot=True)
