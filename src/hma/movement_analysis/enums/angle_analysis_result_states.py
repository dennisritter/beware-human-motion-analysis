from enum import Enum

class AngleAnalysisResultStates(Enum):
    TARGET_EXCEEDED = "TARGET_EXCEEDED"
    TARGET_UNDERCUT = "TARGET_UNDERCUT"
    IN_TARGET_RANGE = "IN_TARGET_RANGE"
    NONE = "NONE"
    ERROR = "ERROR"