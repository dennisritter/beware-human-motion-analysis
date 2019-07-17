class Exercise:

    def __init__(self, name: str, angles: dict, sets: int = 3, duration: int = 30, iterations: list = [20, 30], pause: int = 15):
        # str - The name of this exercise
        self.name = name
        # dict - The angle restrictions for start/end state, for relevant bodyparts
        self.angles = angles
        # int - The number of sets for this exercise
        self.sets = sets
        # int - The duration of this exercise in seconds
        self.duration = duration
        # int - The number of iterations for this exercise
        self.iterations = iterations
        # int - The pause between each set in seconds
        self.pause = pause
