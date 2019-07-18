class Exercise:

    def __init__(self, name: str, angles: dict, userId: int, description: str = "Sorry, there is no description"):
        # str - The name of this exercise
        self.name = name
        # dict - The angle restrictions for start/end state, for relevant bodyparts
        self.angles = angles
        # int - The number of sets for this exercise
        self.userId = userId
        self.description = description