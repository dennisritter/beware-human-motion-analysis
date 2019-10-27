from hma.movement_analysis.exercise import Exercise
from hma.movement_analysis.enums.pose_format_enum import PoseFormatEnum
from hma.movement_analysis.enums.angle_target_states import AngleTargetStates
from hma.movement_analysis.sequence import Sequence
from hma.movement_analysis.pose_processor import PoseProcessor
from hma.movement_analysis import exercise_loader
from hma.movement_analysis import angle_calculations as acm
from hma.movement_analysis import transformations
from hma.movement_analysis.exercise_evaluator import ExerciseEvaluator
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema, savgol_filter

mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)
g_seq = mocap_poseprocessor.load(
    'data/sequences/thesis_plots/multi/squat/user-1/191024__multi__squat__user-1__0.json',
    'squat')
# g_seq.visualise()
squat = exercise_loader.load('data/exercises/squat.json')
overheadpress = exercise_loader.load('data/exercises/overhead-press.json')
lungleleft = exercise_loader.load('data/exercises/lunge-left.json')
EE = ExerciseEvaluator(squat, g_seq)
g_iterations = EE.find_iteration_keypoints(plot=True)
print(f"Iterations for complete sequence [{len(g_seq)} Frames]{g_iterations}")

# prio_angles = EE.prio_angles
# for i, iteration_seq in enumerate(iteration_seqs):
#     result = EE.evaluate(iteration_seq, iterations[i][1])
#     print(f"########## Evaluation results for iteration [{i}] ##########")
#     print(f"START FRAME [{iterations[i][0]}]")
#     for angle in prio_angles:
#         print(f"{result[iterations[i][0]][angle[0]][angle[1].value]}")
#     print(f"TURNING FRAME [{iterations[i][1]}]")
#     for angle in prio_angles:
#         print(f"{result[iterations[i][1]][angle[0]][angle[1].value]}")
#     print(f"END FRAME [{iterations[i][1]}]")
#     for angle in prio_angles:
#         print(f"{result[iterations[i][2]][angle[0]][angle[1].value]}")
