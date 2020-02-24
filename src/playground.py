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

# exercise = "data/exercises/squat.json"
exercise = "data/exercises/overhead-press.json"
# filename = "data/sequences/191024_tracking/single/squat/user-2/191024__single__squat__user-2__1.json"
# filename = "data/sequences/191024_tracking/single/squat/user-4/191024__single__squat__user-4__1.json"
# filename = "data/sequences/191024_tracking/single/biceps_curl_left/user-2/191024__single__biceps_curl_left__user-2__1.json"
# filename = "data/sequences/191024_tracking/single/biceps_curl_right/user-2/191024__single__biceps_curl_right__user-2__1.json"
# filename = "data/sequences/191024_tracking/single/triceps_extension_left/user-2/191024__single__triceps_extension_left__user-2__1.json"
# filename = "data/sequences/191024_tracking/single/triceps_extension_right/user-2/191024__single__triceps_extension_right__user-2__1.json"
filename = "data/sequences/191024_tracking/single/overhead_press/user-6/191024__single__overhead_press__user-6__4.json"
# filename = "data/sequences/191024_tracking/single/overhead_press/user-5/191024__single__overhead_press__user-5__10.json"
# filename = "data/sequences/191024_tracking/single/lunge_left/user-2/191024__single__lunge_left__user-2__1.json"
# filename = "data/sequences/test/100_frames.json"
# f1 = "data/sequences/test/a.json"
# f2 = "data/sequences/test/b.json"

### Check Sequence angles ang iterations
seq = Sequence.from_mocap_file(filename)
exe = Exercise.from_file(exercise)
EE = ExerciseEvaluator(exe, seq)
iterations = EE.find_iteration_keypoints(plot=True)