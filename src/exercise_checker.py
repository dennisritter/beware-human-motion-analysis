from Exercise import Exercise
from JointsAngleMapper import JointsAngleMapper
from PoseFormatEnum import PoseFormatEnum
import exercise_loader
from Sequence import Sequence
from PoseMapper import PoseMapper
import visualize
import angle_calculations_medical as acm

# Get Exercise Object from json file
ex = exercise_loader.load('data/exercises/squat.json')

# Get PoseMapper instance for MOCAP sequences
mocap_posemapper = PoseMapper(PoseFormatEnum.MOCAP)
# Convert mocap json string Positions to Sequence Object
seq = mocap_posemapper.load('data/sequences/squat_3/complete-session.json', 'Squat')

# Add joints to angles property of exercise
jam = JointsAngleMapper(PoseFormatEnum.MOCAP)
jam.addJointsToAngles(ex)

joints = jam.jointsMap

# Calculate angles for Sequence
# Left Hip Flexion/Extension
acm.calc_angle_hip_flexion_extension(seq, joints["hip_left"]["flexion_extension"])
# Right Hip Flexion/Extension
acm.calc_angle_hip_flexion_extension(seq, joints["hip_right"]["flexion_extension"])
# Left Knee Flexion/Extension
acm.calc_angle_knee_flexion_extension(seq, joints["knee_left"]["flexion_extension"])
# Right Knee Flexion/Extension
acm.calc_angle_knee_flexion_extension(seq, joints["knee_right"]["flexion_extension"])
# Visualize angle
visualize.vis_angle(seq, joints["knee_left"]["flexion_extension"], 50)
