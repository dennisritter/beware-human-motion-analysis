import unittest
import numpy as np

from src.hma.movement_analysis import angle_calculations_medical as acm
from src.hma.movement_analysis.Sequence import Sequence
from src.hma.movement_analysis.PoseFormatEnum import PoseFormatEnum


class TestAngleCalculationsMedicalcElbow(unittest.TestCase):

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
        self.positions = [
            [
                [1, 0.5, 1],      # "LeftWrist": 0,
                [1, 1, 1],        # "LeftElbow": 1,
                [1, 1.5, 1],      # "LeftShoulder": 2,
                [0, 0, 0],        # "Neck": 3,
                [0, 0, 0],        # "Torso": 4,
                [0, 0, 0],        # "Waist": 5,
                [0, 0, 0],        # "LeftAnkle": 6,
                [0, 0, 0],        # "LeftKnee": 7,
                [0, 0, 0],        # "LeftHip": 8,
                [0, 0, 0],        # "RightAnkle": 9,
                [0, 0, 0],        # "RightKnee": 10,
                [0, 0, 0],        # "RightHip": 11,
                [0, 0, 0],        # "RightWrist": 12,
                [0, 0, 0],        # "RightElbow": 13,
                [0, 0, 0],        # "RightShoulder": 14,
                [0, 0, 0],        # "Head": 15
            ]
        ]

    def test_calc_angle_elbow0(self):
        positions = self.positions
        positions[0][0] = [1, 0.5, 1]
        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        self.assertAlmostEqual(acm.calc_angles_elbow(seq, self.bp["LeftElbow"], self.bp["LeftShoulder"], self.bp["LeftWrist"])["flexion_extension"][0], 0)

    def test_calc_angle_elbow90(self):
        positions = self.positions
        positions[0][0] = [1, 1, 0.5]
        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        self.assertAlmostEqual(acm.calc_angles_elbow(seq, self.bp["LeftElbow"], self.bp["LeftShoulder"], self.bp["LeftWrist"])["flexion_extension"][0], 90.0)

    def test_calc_angle_elbow180(self):
        positions = self.positions
        positions[0][0] = [1, 1.5, 1]
        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        self.assertAlmostEqual(acm.calc_angles_elbow(seq, self.bp["LeftElbow"], self.bp["LeftShoulder"], self.bp["LeftWrist"])["flexion_extension"][0], 180.0)

    def test_calc_angle_elbow45(self):
        positions = self.positions
        positions[0][0] = [1, 0.5, 0.5]
        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        self.assertAlmostEqual(acm.calc_angles_elbow(seq, self.bp["LeftElbow"], self.bp["LeftShoulder"], self.bp["LeftWrist"])["flexion_extension"][0], 45.0)
