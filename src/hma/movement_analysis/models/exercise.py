"""Contains the code for the Exercise model."""
import json
import logging
from typing import Union


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

        #? logging in whole hma project?
        self.logger = logging.getLogger(__name__)

    def to_json(self) -> str:
        """Returns the exercise instance as a json-formatted string."""
        # return json.dumps(self.__dict__)  # does not work because of logger
        json_dict = {'name': self.name, 'angles': self.angles, 'userId': self.userId, 'description': self.description}
        return json.dumps(json_dict)

    @classmethod
    def from_json(cls, json_data: Union[str, dict]) -> 'Exercise':
        """Creates a new Exercise instance from a json-formatted string.

        Args:
            json_data Union[str, dict]: The json-formatted string or dict.

        Returns:
            Exercise: a new Exercise instance from the given input.
        """
        if isinstance(json_data, str):
            json_dict = json.loads(json_data)
        else:
            json_dict = json_data
        return cls(**json_dict)

    @classmethod
    def from_file(cls, path: str) -> 'Exercise':
        """Loads an exercise .json file and returns an Exercise object.

        Args:
            path (str): Path to the json file

        Returns:
            Exercise: a new Exercise instance from the given input.
        """
        with open(path, 'r') as exercise_file:
            # load, parse file from json and return class
            return cls.from_json(exercise_file.read())
