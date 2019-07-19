from Exercise import Exercise
from JointsAngleMapper import JointsAngleMapper
from PoseFormatEnum import PoseFormatEnum
from exercise_loader import load
from Sequence import Sequence
from PoseMapper import PoseMapper
import angle_calculations_medical as acm

# Get Exercise Object from json file
ex = load('data/exercises/squat.json')

# Get PoseMapper instance for MOCAP sequences
mocap_posemapper = PoseMapper(PoseFormatEnum.MOCAP)
# Convert mocap json string Positions to Sequence Object
seq = mocap_posemapper.load('data/sequences/squat_3/complete-session.json', 'Squat')

# Add joints to angles property of exercise
jam = JointsAngleMapper(PoseFormatEnum.MOCAP)
jam.addJointsToAngles(ex)

joints = jam.jointsMap
# Calculate angles for Sequence
acm.calc_angles_lefthip_flexion_extension(seq, joints["hip_left"]["flexion_extension"])

# TODO: Analyse motion sequence angles
# -> Get angle for all(?) frames of the sequence
# -> Check if angle breaks defined rules for angles in exercise
