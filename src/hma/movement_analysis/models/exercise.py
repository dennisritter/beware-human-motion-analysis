class Exercise:
    """Represents a sport exercise.

    Attributes:
        name (str):             The name of this exercise.
        angles (dict):          The angle restrictions for start/end state, for relevant bodyparts.
        userId (int):           The user id this exercise is personalised for.
        description(str):       A description for this exercise.
    """

    def __init__(self, name: str, angles: dict, userId: int = 0, description: str = "no description"):
        self.name = name
        self.angles = angles
        self.userId = userId
        self.description = description
