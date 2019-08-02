from Exercise import Exercise
from PoseFormatEnum import PoseFormatEnum
import exercise_loader
from Sequence import Sequence
from PoseMapper import PoseMapper
import angle_calculations_medical as acm
import numpy as np
import transformations
import math

# Get Exercise Object from json file
ex = exercise_loader.load('data/exercises/squat.json')
# Get PoseMapper instance for MOCAP sequences
mocap_posemapper = PoseMapper(PoseFormatEnum.MOCAP)
# Convert mocap json string Positions to Sequence Object
seq = mocap_posemapper.load('data/sequences/squat_3/complete-session.json', 'Squat')

acm.calc_angles_shoulder_left(seq, 2, 14, 3, 1, 0, log=True)
# acm.calc_angles_shoulder_right(seq, 14, 2, 3, 13, log=True)
