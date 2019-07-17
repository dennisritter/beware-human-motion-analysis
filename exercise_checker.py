from Exercise import Exercise
from exercise_loader import load

ex = load('data/exercises/squat.json')
print(ex.__dict__)

# TODO: Analyse motion sequence angles
# -> Define Angles:Human_Joints mapping to validate correct angles
# -> Get motion sequence
# -> Get groups of necessary joints for angle validation
# -> Get angle for all(?) frames of the sequence
# -> Check if angle breaks defined rules for angles in exercise
