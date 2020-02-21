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
filename = "data/sequences/191024_tracking/single/overhead_press/user-2/191024__single__overhead_press__user-2__15.json"
# filename = "data/sequences/191024_tracking/single/overhead_press/user-5/191024__single__overhead_press__user-5__10.json"
# filename = "data/sequences/191024_tracking/single/lunge_left/user-2/191024__single__lunge_left__user-2__1.json"
# filename = "data/sequences/test/100_frames.json"
# f1 = "data/sequences/test/a.json"
# f2 = "data/sequences/test/b.json"

seq = Sequence.from_mocap_file(filename)
bps = seq.body_parts

#! -180° angles are kind of impossible for the human body.
#! Maybe we should check for a negative threshold only and fix those angles instead of generalizing the issue, resulting in some complicated edge cases.

### Fix ball joint angles that exceed the range of [-180, 180] degrees
ball_joints = ['shoulder_l', 'shoulder_r', 'hip_l', 'hip_r']
# Absolute angle change threshold in between one frame
delta = 180
for ball_joint in ball_joints:
    for angle_type in range(len([AngleTypes.FLEX_EX, AngleTypes.AB_AD])):
        # Get Flexion or Abduction angles for current Ball Joint
        angles = seq.joint_angles[:, bps[ball_joint], angle_type]
        # Get distances between angles
        distances = np.diff(angles)
        # Get indices where the absolute distance exceeds defined threshold
        jump = np.argwhere((distances > delta) | (distances < -delta))

        # Iterate over all jump indices, with a step size of 2 as we want to change a whole slice
        # E.g.: The angle jumps from 180 to -180 and back to 180 later on. We want to fix all angles in between.
        for i in range(0, len(jump), 2):
            # The index in 'angles' after the jump happened
            # As len(distances) == len(angles) - 1, there is always a angles[jump_idx+1] value.
            jump_post_idx = jump[i][0] + 1
            if distances[jump[i][0]] < -180:
                # If there is another jump_idx
                if jump[-1][0] != jump[i][0]:
                    jump_back_idx = jump[i + 1][0] + 1
                    angles[jump_post_idx:jump_back_idx + 1] += 360
                # Else fix all angles until end
                else:
                    angles[jump_post_idx:] += 360

            elif distances[jump[i][0]] > 180:
                # If there is another jump_idx
                if jump[-1][0] != jump[i][0]:
                    jump_back_idx = jump[i + 1][0]
                    angles[jump_post_idx:jump_back_idx + 1] -= 360
                # Else fix all angles until end
                else:
                    angles[jump_post_idx:] -= 360

# flex_angles_compare = flex_angles[:, 2]
# sv = SkeletonVisualiser(bug_seq)
# sv.show()
a = 'moep'
### Check Sequence angles ang iterations
# seq = Sequence.from_mocap_file(filename)
# exe = Exercise.from_file(exercise)
# EE = ExerciseEvaluator(exe, seq)
# iterations = EE.find_iteration_keypoints(plot=True)