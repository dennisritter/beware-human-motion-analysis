import warnings
import math
from movement_analysis.AngleTargetStates import AngleTargetStates
from movement_analysis.AngleAnalysisResultStates import AngleAnalysisResultStates


class Exercise:

    HIGH_PRIO = 1.0
    MEDIUM_PRIO = 0.5
    LOW_PRIO = 0.0

    def __init__(self, name: str, angles: dict, userId: int = 0, description: str = "Sorry, there is no description"):
        # str - The name of this exercise
        self.name = name
        # dict - The angle restrictions for start/end state, for relevant bodyparts
        self.angles = angles
        # int - The user id this exercise is specialized for
        self.userId = userId
        # str - A description for this exercise
        self.description = description

    def _check_angle_shoulder_left_flexion_extension(self, angle: list, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        if target_state.value not in self.angles.keys():
            warnings.warn("The target_state parameter value is not present in the Exercises' angles attribute. Cancelng Analysis.")
            return []
        
        # Determine whether movement from START to END is Extension, Flexion, or None
        target_end = self.angles[AngleTargetStates.END.value]["shoulder_left"]["flexion_extension"]["angle"][1]
        target_start = self.angles[AngleTargetStates.START.value]["shoulder_left"]["flexion_extension"]["angle"][1]
        target_end_is_flexion = target_end > target_start
        target_end_is_extension = target_end < target_start
        target_end_is_zero = target_end == target_start

        target_min = min(self.angles[target_state.value]["shoulder_left"]["flexion_extension"]["angle"])
        target_max = max(self.angles[target_state.value]["shoulder_left"]["flexion_extension"]["angle"])

        result_state = AngleAnalysisResultStates.NONE.value
        if target_state.value == AngleTargetStates.START.value:
            if angle < target_min - tolerance:
                result_state = AngleAnalysisResultStates.TARGET_EXCEEDED.value if target_end_is_flexion else result_state
                result_state = AngleAnalysisResultStates.TARGET_UNDERCUT.value if target_end_is_extension else result_state
                result_state = AngleAnalysisResultStates.TARGET_EXCEEDED.value if target_end_is_zero else result_state
            if angle > target_max + tolerance:
                result_state = AngleAnalysisResultStates.TARGET_UNDERCUT.value if target_end_is_flexion else result_state
                result_state = AngleAnalysisResultStates.TARGET_EXCEEDED.value if target_end_is_extension else result_state
                result_state = AngleAnalysisResultStates.TARGET_EXCEEDED.value if target_end_is_zero else result_state
            if angle <= target_max + tolerance and angle >= target_min - tolerance:
                result_state = AngleAnalysisResultStates.IN_TARGET_RANGE.value

        if target_state.value == AngleTargetStates.END.value:
            if angle < target_min - tolerance:
                result_state = AngleAnalysisResultStates.TARGET_UNDERCUT.value if target_end_is_flexion else result_state
                result_state = AngleAnalysisResultStates.TARGET_EXCEEDED.value if target_end_is_extension else result_state
                result_state = AngleAnalysisResultStates.TARGET_EXCEEDED.value if target_end_is_zero else result_state
            if angle > target_max + tolerance:
                result_state = AngleAnalysisResultStates.TARGET_EXCEEDED.value if target_end_is_flexion else result_state
                result_state = AngleAnalysisResultStates.TARGET_UNDERCUT.value if target_end_is_extension else result_state
                result_state = AngleAnalysisResultStates.TARGET_EXCEEDED.value if target_end_is_zero else result_state
            if angle <= target_max + tolerance and angle >= target_min - tolerance:
                result_state = AngleAnalysisResultStates.IN_TARGET_RANGE.value

        result = {
            "angle": angle,
            "target_min": target_min,
            "target_max": target_max,
            "target_state": target_state.value,
            "result_state": result_state,
        }
        return result
    
    def _check_angles_shoulder_left_abduction_adduction(self, angles: list, target_state: AngleTargetStates, tolerance: int = 10):
        return 0