from Exercise import Exercise
from exercise_loader import load

ex = load('data/exercises/squat.json')
print(ex.__dict__)
