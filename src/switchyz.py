import os
import glob
import json
import tslearn.metrics as ts
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from hma.movement_analysis.pose_processor import PoseProcessor
from hma.movement_analysis.enums.pose_format_enum import PoseFormatEnum
from hma.movement_analysis import exercise_loader
from hma.movement_analysis.helpers import reformat_angles_dtw
from hma.movement_analysis.exercise_evaluator import ExerciseEvaluator
from hma.movement_analysis.skeleton_visualiser import SkeletonVisualiser
from hma.movement_analysis.helpers import hierarchy_pos
from hma.movement_analysis.helpers import draw_scenegraph
from hma.movement_analysis.transformations import get_pelvis_coordinate_system

np.set_printoptions(suppress=True)

mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)

root = 'data/sequences/191024_tracking/single/squat/user-2'
sequences = []
for filename in Path(root).rglob('191024__single__squat__user-2__1.json'):
    print(f"Loading {filename}")
    seq = mocap_poseprocessor.load(filename)
    sequences.append(seq)

print(np.around(sequences[0][70].positions, 2))
print(np.around(sequences[0][70].joint_angles, 2))

sv = SkeletonVisualiser(sequences[0][70])
sv.show()
