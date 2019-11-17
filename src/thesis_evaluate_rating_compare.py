import matplotlib.pyplot as plt
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
import seaborn as sns


mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)
squat = exercise_loader.load('data/exercises/squat.json')

seqs = []
filenames = list(Path("data/evaluation/rating/compare/squat/selection").rglob("*.json"))
for filename in filenames:
    print(f"Loading Sequence file: {filename}")
    sequence = mocap_poseprocessor.load(filename, str(filename).split('\\')[-1])
    seqs.append(sequence)
    # sequence.visualise()

EE = None
for seq in seqs:
    if EE is None:
        EE = ExerciseEvaluator(squat, seq)
    else:
        EE.set_sequence(seq)
    iterations = EE.find_iteration_keypoints()
    rating = EE.evaluate(iterations[0][1])
    prio_angles = EE.prio_angles
    print("---------------------------")
    print(seq.name)
    for angle in prio_angles:
        tf_angle = seq.joint_angles[iterations[0][1], angle[0], angle[1].value]
        print(f"{tf_angle} [{iterations[0][1]}] {rating[iterations[0][1]][angle[0]][angle[1].value]['result_state']}")
        # print(rating[iterations[0][1]][angle[0]][angle[1].value])
    # print(iterations)
