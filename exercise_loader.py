import warnings
import json
from Exercise import Exercise

REQUIRED_KEYS = ["name", "angles"]
OPTIONAL_KEYS = ["sets", "duration", "iterations", "pause"]


def load(path: str):
    """ Loads a exercise represantation .json file and returns an Exercise object if the file is valid.
    """
    with open(path, 'r') as myfile:
        ex_str = myfile.read()
    ex = json.loads(ex_str)
    # Validate loaded file
    is_valid = validate_exercise(ex)
    if is_valid:
        return Exercise(ex["name"], ex["angles"])
    else:
        warnings.warn(f"File '{path}' is not a valid exercise file.")
        return


def validate_exercise(ex: dict):
    """ Validates if the given dictionary is a correct Exercise representation.
    """
    # TODO: Check if Angles structure and values are also valid
    is_valid = True
    # Check if Required Keys present
    for req_key in REQUIRED_KEYS:
        if req_key not in ex:
            # raise RuntimeError(f"The Exercise file does not contain the required key '{req_key}'.")
            warnings.warn(f"The Exercise file does not contain the required key '{req_key}'.")
            is_valid = False
    # Check if Optional Keys present and warn if not
    for opt_key in OPTIONAL_KEYS:
        if opt_key not in ex:
            warnings.warn(f"The Exercise file does not contain the key '{opt_key}'. Using default value.")
    return is_valid
