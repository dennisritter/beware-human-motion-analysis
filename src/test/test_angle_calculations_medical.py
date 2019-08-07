import unittest
import numpy as np
from movement_analysis import angle_calculations_medical as acm
from movement_analysis.Sequence import Sequence
from movement_analysis.PoseFormatEnum import PoseFormatEnum


class TestAngleCalculationsMedical(unittest.TestCase):

    def setUp(self):
        self.bp = {
            "LeftWrist": 0,
            "LeftElbow": 1,
            "LeftShoulder": 2,
            "Neck": 3,
            "Torso": 4,
            "Waist": 5,
            "LeftAnkle": 6,
            "LeftKnee": 7,
            "LeftHip": 8,
            "RightAnkle": 9,
            "RightKnee": 10,
            "RightHip": 11,
            "RightWrist": 12,
            "RightElbow": 13,
            "RightShoulder": 14,
            "Head": 15
        }

    def test_calc_angle_0(self):
        self.assertEqual(acm.calc_angle([0, 0, 0], [1, 0, 0], [1, 0, 0]), 0)

    def test_calc_angle_90(self):
        self.assertEqual(acm.calc_angle([0, 0, 0], [1, 0, 0], [0, 1, 0]), 90.0)

    def test_calc_angle_180(self):
        self.assertEqual(acm.calc_angle([0, 0, 0], [1, 0, 0], [-1, 0, 0]), 180.0)

    def test_calc_angle_45(self):
        self.assertAlmostEqual(acm.calc_angle([0, 0, 0], [1, 0, 0], [1, 1, 0]), 45.0)

    def test_calc_angles_hip_left_flex0_abd0(self):
        positions = [
            [
                [0, 0, 0],        # "LeftWrist": 0,
                [0, 0, 0],        # "LeftElbow": 1,
                [0, 0, 0],        # "LeftShoulder": 2,
                [0, 0, 0],        # "Neck": 3,
                [1.5, 2, 1],      # "Torso": 4,
                [0, 0, 0],        # "Waist": 5,
                [0, 0, 0],        # "LeftAnkle": 6,
                [1, 0.5, 1],      # "LeftKnee": 7,
                [1, 1, 1],        # "LeftHip": 8,
                [0, 0, 0],        # "RightAnkle": 9,
                [0, 0, 0],        # "RightKnee": 10,
                [2, 1, 1],        # "RightHip": 11,
                [0, 0, 0],        # "RightWrist": 12,
                [0, 0, 0],        # "RightElbow": 13,
                [0, 0, 0],        # "RightShoulder": 14,
                [0, 0, 0],        # "Head": 15
            ]
        ]
        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        expected_result = {
            "flexion_extension": [0.0],
            "abduction_adduction": [0.0],
        }
        self.assertDictEqual(acm.calc_angles_hip_left(seq, self.bp["LeftHip"], self.bp["RightHip"], self.bp["Torso"], self.bp["LeftKnee"]), expected_result)


if __name__ == '__main__':
    unittest.main()
