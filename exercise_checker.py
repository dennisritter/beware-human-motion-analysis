from Exercise import Exercise
from ExercisePoseSegmentor import ExercisePoseSegmentor
from PoseFormatEnum import PoseFormatEnum
from exercise_loader import load

# Get Exercise Object from json file
ex = load('data/exercises/squat.json')

jointAngleMapper = ExercisePoseSegmentor(PoseFormatEnum.MOCAP)
jointAngleMapper.addJointsToAngles(ex)

print(ex.angles)
# TODO: Analyse motion sequence angles
# -> Define Angles:Human_Joints mapping to validate correct angles
# -> Get motion sequence
# -> Get groups of necessary joints for angle validation
# -> Get angle for all(?) frames of the sequence
# -> Check if angle breaks defined rules for angles in exercise
