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

    def check_angles_shoulder_left(self, angle_flex_ex: float, angle_abd_add: float, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        result = {
            "flexion_extension": self._check_angle_shoulder_left_flexion_extension(angle_flex_ex, target_state, tolerance),
            "abduction_adduction": self._check_angle_shoulder_left_abduction_adduction(angle_abd_add, target_state, tolerance)
        }
        return result

    def check_angles_shoulder_right(self, angle_flex_ex: float, angle_abd_add: float, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        result = {
            "flexion_extension": self._check_angle_shoulder_right_flexion_extension(angle_flex_ex, target_state, tolerance),
            "abduction_adduction": self._check_angle_shoulder_right_abduction_adduction(angle_abd_add, target_state, tolerance)
        }
        return result

    def check_angles_hip_left(self, angle_flex_ex: float, angle_abd_add: float, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        result = {
            "flexion_extension": self._check_angle_hip_left_flexion_extension(angle_flex_ex, target_state, tolerance),
            "abduction_adduction": self._check_angle_hip_left_abduction_adduction(angle_abd_add, target_state, tolerance)
        }
        return result

    def check_angles_hip_right(self, angle_flex_ex: float, angle_abd_add: float, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        result = {
            "flexion_extension": self._check_angle_hip_right_flexion_extension(angle_flex_ex, target_state, tolerance),
            "abduction_adduction": self._check_angle_hip_right_abduction_adduction(angle_abd_add, target_state, tolerance)
        }
        return result

    def check_angles_elbow_left(self, angle_flex_ex: float, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        result = {
            "flexion_extension": self._check_angle_elbow_left_flexion_extension(angle_flex_ex, target_state, tolerance),
        }
        return result

    def check_angles_elbow_right(self, angle_flex_ex: float, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        result = {
            "flexion_extension": self._check_angle_elbow_right_flexion_extension(angle_flex_ex, target_state, tolerance),
        }
        return result

    def check_angles_knee_left(self, angle_flex_ex: float, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        result = {
            "flexion_extension": self._check_angle_knee_left_flexion_extension(angle_flex_ex, target_state, tolerance),
        }
        return result

    def check_angles_knee_right(self, angle_flex_ex: float, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        result = {
            "flexion_extension": self._check_angle_knee_right_flexion_extension(angle_flex_ex, target_state, tolerance),
        }
        return result

    def _get_angle_analysis_result_state(self,
                                         angle: float,
                                         target_state: AngleTargetStates,
                                         target_start: float,
                                         target_end: float,
                                         target_min: float,
                                         target_max: float,
                                         tolerance: int) -> AngleAnalysisResultStates:

        target_end_is_flexion = target_end > target_start
        target_end_is_extension = target_end < target_start
        target_end_is_zero = target_end == target_start

        result_state = AngleAnalysisResultStates.NONE.value
        if target_state.value == AngleTargetStates.START.value:
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

        return result_state

    def _check_angle_shoulder_left_flexion_extension(self, angle: float, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        if target_state.value not in self.angles.keys():
            warnings.warn("The target_state parameter value is not present in the Exercises' angles attribute. Cancelng Analysis.")
            return []

        target_end = self.angles[AngleTargetStates.END.value]["shoulder_left"]["flexion_extension"]["angle"][1]
        target_start = self.angles[AngleTargetStates.START.value]["shoulder_left"]["flexion_extension"]["angle"][1]
        # Determine whether movement from START to END is Extension, Flexion, or None
        target_min = min(self.angles[target_state.value]["shoulder_left"]["flexion_extension"]["angle"])
        target_max = max(self.angles[target_state.value]["shoulder_left"]["flexion_extension"]["angle"])

        result = {
            "angle": angle,
            "target_min": target_min,
            "target_max": target_max,
            "target_state": target_state.value,
            "result_state": self._get_angle_analysis_result_state(angle, target_state, target_start, target_end, target_min, target_max, tolerance),
        }
        return result

    def _check_angle_shoulder_right_flexion_extension(self, angle: float, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        if target_state.value not in self.angles.keys():
            warnings.warn("The target_state parameter value is not present in the Exercises' angles attribute. Cancelng Analysis.")
            return []

        target_end = self.angles[AngleTargetStates.END.value]["shoulder_right"]["flexion_extension"]["angle"][1]
        target_start = self.angles[AngleTargetStates.START.value]["shoulder_right"]["flexion_extension"]["angle"][1]
        # Determine whether movement from START to END is Extension, Flexion, or None
        target_min = min(self.angles[target_state.value]["shoulder_right"]["flexion_extension"]["angle"])
        target_max = max(self.angles[target_state.value]["shoulder_right"]["flexion_extension"]["angle"])

        result = {
            "angle": angle,
            "target_min": target_min,
            "target_max": target_max,
            "target_state": target_state.value,
            "result_state": self._get_angle_analysis_result_state(angle, target_state, target_start, target_end, target_min, target_max, tolerance),
        }
        return result

    def _check_angle_shoulder_left_abduction_adduction(self, angle: float, target_state: AngleTargetStates, tolerance: int = 10):
        if target_state.value not in self.angles.keys():
            warnings.warn("The target_state parameter value is not present in the Exercises' angles attribute. Cancelng Analysis.")
            return []

        # Determine whether movement from START to END is Extension, Flexion, or None
        target_end = self.angles[AngleTargetStates.END.value]["shoulder_left"]["abduction_adduction"]["angle"][1]
        target_start = self.angles[AngleTargetStates.START.value]["shoulder_left"]["abduction_adduction"]["angle"][1]
        target_min = min(self.angles[target_state.value]["shoulder_left"]["abduction_adduction"]["angle"])
        target_max = max(self.angles[target_state.value]["shoulder_left"]["abduction_adduction"]["angle"])

        result = {
            "angle": angle,
            "target_min": target_min,
            "target_max": target_max,
            "target_state": target_state.value,
            "result_state": self._get_angle_analysis_result_state(angle, target_state, target_start, target_end, target_min, target_max, tolerance),
        }
        return result

    def _check_angle_shoulder_right_abduction_adduction(self, angle: float, target_state: AngleTargetStates, tolerance: int = 10):
        if target_state.value not in self.angles.keys():
            warnings.warn("The target_state parameter value is not present in the Exercises' angles attribute. Cancelng Analysis.")
            return []

        # Determine whether movement from START to END is Extension, Flexion, or None
        target_end = self.angles[AngleTargetStates.END.value]["shoulder_right"]["abduction_adduction"]["angle"][1]
        target_start = self.angles[AngleTargetStates.START.value]["shoulder_right"]["abduction_adduction"]["angle"][1]
        target_min = min(self.angles[target_state.value]["shoulder_right"]["abduction_adduction"]["angle"])
        target_max = max(self.angles[target_state.value]["shoulder_right"]["abduction_adduction"]["angle"])

        result = {
            "angle": angle,
            "target_min": target_min,
            "target_max": target_max,
            "target_state": target_state.value,
            "result_state": self._get_angle_analysis_result_state(angle, target_state, target_start, target_end, target_min, target_max, tolerance),
        }
        return result

    def _check_angle_hip_left_flexion_extension(self, angle: float, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        if target_state.value not in self.angles.keys():
            warnings.warn("The target_state parameter value is not present in the Exercises' angles attribute. Cancelng Analysis.")
            return []

        target_end = self.angles[AngleTargetStates.END.value]["hip_left"]["flexion_extension"]["angle"][1]
        target_start = self.angles[AngleTargetStates.START.value]["hip_left"]["flexion_extension"]["angle"][1]
        # Determine whether movement from START to END is Extension, Flexion, or None
        target_min = min(self.angles[target_state.value]["hip_left"]["flexion_extension"]["angle"])
        target_max = max(self.angles[target_state.value]["hip_left"]["flexion_extension"]["angle"])

        result = {
            "angle": angle,
            "target_min": target_min,
            "target_max": target_max,
            "target_state": target_state.value,
            "result_state": self._get_angle_analysis_result_state(angle, target_state, target_start, target_end, target_min, target_max, tolerance),
        }
        return result

    def _check_angle_hip_right_flexion_extension(self, angle: float, target_state: AngleTargetStates, tolerance: int = 10) -> dict:
        if target_state.value not in self.angles.keys():
            warnings.warn("The target_state parameter value is not present in the Exercises' angles attribute. Cancelng Analysis.")
            return []

        target_end = self.angles[AngleTargetStates.END.value]["hip_right"]["flexion_extension"]["angle"][1]
        target_start = self.angles[AngleTargetStates.START.value]["hip_right"]["flexion_extension"]["angle"][1]
        # Determine whether movement from START to END is Extension, Flexion, or None
        target_min = min(self.angles[target_state.value]["hip_right"]["flexion_extension"]["angle"])
        target_max = max(self.angles[target_state.value]["hip_right"]["flexion_extension"]["angle"])

        result = {
            "angle": angle,
            "target_min": target_min,
            "target_max": target_max,
            "target_state": target_state.value,
            "result_state": self._get_angle_analysis_result_state(angle, target_state, target_start, target_end, target_min, target_max, tolerance),
        }
        return result

    def _check_angle_hip_left_abduction_adduction(self, angle: float, target_state: AngleTargetStates, tolerance: int = 10):
        if target_state.value not in self.angles.keys():
            warnings.warn("The target_state parameter value is not present in the Exercises' angles attribute. Cancelng Analysis.")
            return []

        # Determine whether movement from START to END is Extension, Flexion, or None
        target_end = self.angles[AngleTargetStates.END.value]["hip_left"]["abduction_adduction"]["angle"][1]
        target_start = self.angles[AngleTargetStates.START.value]["hip_left"]["abduction_adduction"]["angle"][1]
        target_min = min(self.angles[target_state.value]["hip_left"]["abduction_adduction"]["angle"])
        target_max = max(self.angles[target_state.value]["hip_left"]["abduction_adduction"]["angle"])

        result = {
            "angle": angle,
            "target_min": target_min,
            "target_max": target_max,
            "target_state": target_state.value,
            "result_state": self._get_angle_analysis_result_state(angle, target_state, target_start, target_end, target_min, target_max, tolerance),
        }
        return result

    def _check_angle_hip_right_abduction_adduction(self, angle: float, target_state: AngleTargetStates, tolerance: int = 10):
        if target_state.value not in self.angles.keys():
            warnings.warn("The target_state parameter value is not present in the Exercises' angles attribute. Cancelng Analysis.")
            return []

        # Determine whether movement from START to END is Extension, Flexion, or None
        target_end = self.angles[AngleTargetStates.END.value]["hip_right"]["abduction_adduction"]["angle"][1]
        target_start = self.angles[AngleTargetStates.START.value]["hip_right"]["abduction_adduction"]["angle"][1]
        target_min = min(self.angles[target_state.value]["hip_right"]["abduction_adduction"]["angle"])
        target_max = max(self.angles[target_state.value]["hip_right"]["abduction_adduction"]["angle"])

        result = {
            "angle": angle,
            "target_min": target_min,
            "target_max": target_max,
            "target_state": target_state.value,
            "result_state": self._get_angle_analysis_result_state(angle, target_state, target_start, target_end, target_min, target_max, tolerance),
        }
        return result

    def _check_angle_elbow_left_flexion_extension(self, angle: float, target_state: AngleTargetStates, tolerance: int = 10):
        if target_state.value not in self.angles.keys():
            warnings.warn("The target_state parameter value is not present in the Exercises' angles attribute. Cancelng Analysis.")
            return []

        # Determine whether movement from START to END is Extension, Flexion, or None
        target_end = self.angles[AngleTargetStates.END.value]["elbow_left"]["flexion_extension"]["angle"][1]
        target_start = self.angles[AngleTargetStates.START.value]["elbow_left"]["flexion_extension"]["angle"][1]
        target_min = min(self.angles[target_state.value]["elbow_left"]["flexion_extension"]["angle"])
        target_max = max(self.angles[target_state.value]["elbow_left"]["flexion_extension"]["angle"])

        result = {
            "angle": angle,
            "target_min": target_min,
            "target_max": target_max,
            "target_state": target_state.value,
            "result_state": self._get_angle_analysis_result_state(angle, target_state, target_start, target_end, target_min, target_max, tolerance),
        }
        return result

    def _check_angle_elbow_right_flexion_extension(self, angle: float, target_state: AngleTargetStates, tolerance: int = 10):
        if target_state.value not in self.angles.keys():
            warnings.warn("The target_state parameter value is not present in the Exercises' angles attribute. Cancelng Analysis.")
            return []

        # Determine whether movement from START to END is Extension, Flexion, or None
        target_end = self.angles[AngleTargetStates.END.value]["elbow_right"]["flexion_extension"]["angle"][1]
        target_start = self.angles[AngleTargetStates.START.value]["elbow_right"]["flexion_extension"]["angle"][1]
        target_min = min(self.angles[target_state.value]["elbow_right"]["flexion_extension"]["angle"])
        target_max = max(self.angles[target_state.value]["elbow_right"]["flexion_extension"]["angle"])

        result = {
            "angle": angle,
            "target_min": target_min,
            "target_max": target_max,
            "target_state": target_state.value,
            "result_state": self._get_angle_analysis_result_state(angle, target_state, target_start, target_end, target_min, target_max, tolerance),
        }
        return result

    def _check_angle_knee_left_flexion_extension(self, angle: float, target_state: AngleTargetStates, tolerance: int = 10):
        if target_state.value not in self.angles.keys():
            warnings.warn("The target_state parameter value is not present in the Exercises' angles attribute. Cancelng Analysis.")
            return []

        # Determine whether movement from START to END is Extension, Flexion, or None
        target_end = self.angles[AngleTargetStates.END.value]["knee_left"]["flexion_extension"]["angle"][1]
        target_start = self.angles[AngleTargetStates.START.value]["knee_left"]["flexion_extension"]["angle"][1]
        target_min = min(self.angles[target_state.value]["knee_left"]["flexion_extension"]["angle"])
        target_max = max(self.angles[target_state.value]["knee_left"]["flexion_extension"]["angle"])

        result = {
            "angle": angle,
            "target_min": target_min,
            "target_max": target_max,
            "target_state": target_state.value,
            "result_state": self._get_angle_analysis_result_state(angle, target_state, target_start, target_end, target_min, target_max, tolerance),
        }
        return result

    def _check_angle_knee_right_flexion_extension(self, angle: float, target_state: AngleTargetStates, tolerance: int = 10):
        if target_state.value not in self.angles.keys():
            warnings.warn("The target_state parameter value is not present in the Exercises' angles attribute. Cancelng Analysis.")
            return []

        # Determine whether movement from START to END is Extension, Flexion, or None
        target_end = self.angles[AngleTargetStates.END.value]["knee_right"]["flexion_extension"]["angle"][1]
        target_start = self.angles[AngleTargetStates.START.value]["knee_right"]["flexion_extension"]["angle"][1]
        target_min = min(self.angles[target_state.value]["knee_right"]["flexion_extension"]["angle"])
        target_max = max(self.angles[target_state.value]["knee_right"]["flexion_extension"]["angle"])

        result = {
            "angle": angle,
            "target_min": target_min,
            "target_max": target_max,
            "target_state": target_state.value,
            "result_state": self._get_angle_analysis_result_state(angle, target_state, target_start, target_end, target_min, target_max, tolerance),
        }
        return result
