from enum import Enum

"""
Defines types of joint angles.
"""


class AngleTypes(Enum):
    FLEX_EX = 0  # Flexion / Extension
    AB_AD = 1  # Abduction / Adduction
    IN_EX_ROT = 2  # Internal / External rotation
