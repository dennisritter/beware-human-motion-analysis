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
import glob
from pathlib import Path
import json

mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)
squat = exercise_loader.load('data/exercises/squat.json')
biceps_curl_left = exercise_loader.load('data/exercises/biceps-curl-left.json')
biceps_curl_right = exercise_loader.load('data/exercises/biceps-curl-right.json')
knee_lift_left = exercise_loader.load('data/exercises/knee-lift-left.json')
knee_lift_right = exercise_loader.load('data/exercises/knee-lift-right.json')
EE = None
seqs = []

filenames = list(Path("data/sequences/191024_tracking/single/").rglob("191024__single__knee_lift_right__user-3__18.json"))
for filename in filenames:
    print(f"Loading Sequence file: {filename}")
    sequence = mocap_poseprocessor.load(filename, str(filename).split('\\')[-1])
    seqs.append(sequence)
    sequence.visualise()
# 191024__single__squat__user

# for seq in seqs:
#     if EE is None:
#         EE = ExerciseEvaluator(knee_lift_right, seq)
#     else:
#         EE.set_sequence(seq)
#     iterations = EE.find_iteration_keypoints(plot=True)
#     print(iterations)
#     # print(EE.evaluate(iterations))
