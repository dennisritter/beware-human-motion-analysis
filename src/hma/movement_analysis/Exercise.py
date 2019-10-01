import warnings
import math


class Exercise:

    def __init__(self, name: str, angles: dict, userId: int = 0, description: str = "Sorry, there is no description"):
        # str - The name of this exercise
        self.name = name
        # dict - The angle restrictions for start/end state, for relevant bodyparts
        self.angles = angles
        # int - The user id this exercise is specialized for
        self.userId = userId
        # str - A description for this exercise
        self.description = description
