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
seqs = []

filename = Path("data/sequences/191024_tracking/single/squat/user-2/191024__single__squat__user-2__1.json")
sequence = mocap_poseprocessor.load(filename, str(filename).split('\\')[-1])
seqs.append(sequence)
sequence.visualise()
