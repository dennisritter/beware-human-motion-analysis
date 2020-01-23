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

mocap_poseprocessor = PoseProcessor(PoseFormatEnum.MOCAP)

root = 'data/sequences/switchyz'
for filename in Path(root).rglob('*.json'):
    print(filename)
