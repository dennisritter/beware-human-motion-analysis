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

# g_seq = mocap_poseprocessor.load('data/sequences/unique_iterations/complete-session.json', 'squat-dennis-multi-1')
g_seq = mocap_poseprocessor.load('data/sequences/levente/overhead-press/multi/2019-09-25_15-09-03_records/complete-session.json', 'overheadpress')
squat = exercise_loader.load('data/exercises/kniebeuge.json')
overheadpress = exercise_loader.load('data/exercises/overhead-press.json')
lungleleft = exercise_loader.load('data/exercises/lunge-left.json')
EE = None

seqs = []
# Split long sequence for testing
for i in range(0, math.floor(len(g_seq.positions)/30)):
    partial_seq = g_seq[i*30:i*30+30]
    seqs.append(partial_seq)

# find iterations and merge sequence if it was to short to identify an iteration
iterations = []
g_iterations = []
iteration_seqs = []
idx_bias = 0
merged_seq = None
for seq in seqs:
    if merged_seq == None:
        merged_seq = seq
    else:
        merged_seq.merge(seq)
    if EE == None:
        # EE = ExerciseEvaluator(squat, merged_seq)
        EE = ExerciseEvaluator(overheadpress, merged_seq)
    else:
        EE.set_sequence(merged_seq)
    part_iterations_biased = EE.find_iteration_keypoints(20, 20)
    part_iterations = EE.find_iteration_keypoints(20, 20)
    # If at least one iteration found
    if len(part_iterations) >= 1:
        # Store the sequence in a list
        iteration_seqs.append(merged_seq)
        for it in part_iterations:
            iterations.append(it)
        for it_idx, biased_iteration in enumerate(part_iterations_biased):
            for i in range(len(biased_iteration)):
                part_iterations_biased[it_idx][i] += idx_bias
            g_iterations.append(biased_iteration)
        idx_bias += (len(merged_seq) - 30)
        merged_seq = merged_seq[-30:]

    print(f"Iterations from Script: {g_iterations}")

print(f"Iterations found result (GLOBAL): {g_iterations}")
print(f"Iterations found result (LOCAL): {iterations}")
EE.set_sequence(g_seq)
g_iterations = EE.find_iteration_keypoints(20, 20, True)
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
