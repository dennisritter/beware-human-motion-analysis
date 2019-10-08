from hma.movement_analysis.Exercise import Exercise
from hma.movement_analysis.PoseFormatEnum import PoseFormatEnum
from hma.movement_analysis.AngleTargetStates import AngleTargetStates
from hma.movement_analysis.Sequence import Sequence
from hma.movement_analysis.PoseProcessor import PoseProcessor
from hma.movement_analysis import exercise_loader
from hma.movement_analysis import angle_calculations as acm
from hma.movement_analysis import transformations
from hma.movement_analysis.ExerciseEvaluator import ExerciseEvaluator
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema, savgol_filter

# Get PoseProcessor instance for MOCAP sequences
mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)
# Convert mocap json string Positions to Sequence Object
# seq1 = mocap_poseprocessor.load('data/sequences/squat-dennis-1/complete-session.json', 'squat-dennis-multi-1')
# seq2 = mocap_poseprocessor.load('data/sequences/squat_2/complete-session.json', 'squat-dennis-multi-2')
# seq3 = mocap_poseprocessor.load('data/sequences/squat_3/complete-session.json', 'squat-dennis-multi-3')
# seq1 = mocap_poseprocessor.load('data/sequences/squat-dennis-multi-1/complete-session.json', 'squat-dennis-multi-1')
# seq2 = mocap_poseprocessor.load('data/sequences/squat-dennis-multi-2/complete-session.json', 'squat-dennis-multi-2')
# seq3 = mocap_poseprocessor.load('data/sequences/squat-dennis-multi-3/complete-session.json', 'squat-dennis-multi-3')
# seq1 = mocap_poseprocessor.load('data/sequences/overheadpress-dennis-multi-1/complete-session.json', 'overheadpress-dennis-multi-1')
# seq2 = mocap_poseprocessor.load('data/sequences/overheadpress-dennis-multi-2/complete-session.json', 'overheadpress-dennis-multi-2')
# seq3 = mocap_poseprocessor.load('data/sequences/overheadpress-dennis-multi-3/complete-session.json', 'overheadpress-dennis-multi-3')
# seqs = [seq1, seq2, seq3]

seq = mocap_poseprocessor.load(
    'data/sequences/squat-dennis-multi-1/complete-session.json', 'squat-dennis-multi-1')
ex = exercise_loader.load('data/exercises/kniebeuge.json')
exval_squat = ExerciseEvaluator(ex)

# seqs = []
# for i in range(0, math.floor(len(seq.positions)/100)):
#     partial_seq = seq[i*100:i*100+100]
#     seqs.append(partial_seq)

# for i in range(0, len(seqs)):
#     exval_squat.find_iteration_keypoints(seqs[i])
exval_squat.find_iteration_keypoints(seq)
# (10, 35, 60)
