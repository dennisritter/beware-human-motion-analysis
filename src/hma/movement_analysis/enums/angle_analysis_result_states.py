from enum import Enum

"""
Defines result states after comparing motion sequence angles to predifined angle targets´.
"""


class AngleAnalysisResultStates(Enum):
    TARGET_EXCEEDED = "TARGET_EXCEEDED"
    TARGET_UNDERCUT = "TARGET_UNDERCUT"
    IN_TARGET_RANGE = "IN_TARGET_RANGE"
    NONE = "NONE"
    ERROR = "ERROR"
