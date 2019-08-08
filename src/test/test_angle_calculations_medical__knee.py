import unittest
import numpy as np
from movement_analysis import angle_calculations_medical as acm
from movement_analysis.Sequence import Sequence
from movement_analysis.PoseFormatEnum import PoseFormatEnum


class TestAngleCalculationsMedicalKnee(unittest.TestCase):

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
                [0, 0, 0],        # "LeftWrist": 0,
                [0, 0, 0],        # "LeftElbow": 1,
                [0, 0, 0],        # "LeftShoulder": 2,
                [0, 0, 0],        # "Neck": 3,
                [0, 0, 0],        # "Torso": 4,
                [0, 0, 0],        # "Waist": 5,
                [1, 0.5, 1],      # "LeftAnkle": 6,
                [1, 1, 1],        # "LeftKnee": 7,
                [1, 1.5, 1],      # "LeftHip": 8,
                [0, 0, 0],        # "RightAnkle": 9,
                [0, 0, 0],        # "RightKnee": 10,
                [0, 0, 0],        # "RightHip": 11,
                [0, 0, 0],        # "RightWrist": 12,
                [0, 0, 0],        # "RightElbow": 13,
                [0, 0, 0],        # "RightShoulder": 14,
                [0, 0, 0],        # "Head": 15
            ]
        ]

    def test_calc_angle_knee0(self):
        positions = self.positions
        positions[0][6] = [1, 0.5, 1]
        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        self.assertAlmostEqual(acm.calc_angles_knee(seq, self.bp["LeftKnee"], self.bp["LeftHip"], self.bp["LeftAnkle"])["flexion_extension"][0], 0)

    def test_calc_angle_knee90(self):
        positions = self.positions
        positions[0][6] = [1, 1, 1.5]
        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        self.assertAlmostEqual(acm.calc_angles_knee(seq, self.bp["LeftKnee"], self.bp["LeftHip"], self.bp["LeftAnkle"])["flexion_extension"][0], 90)

    def test_calc_angle_knee180(self):
        positions = self.positions
        positions[0][6] = [1, 1.5, 1]
        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        self.assertAlmostEqual(acm.calc_angles_knee(seq, self.bp["LeftKnee"], self.bp["LeftHip"], self.bp["LeftAnkle"])["flexion_extension"][0], 180)

    def test_calc_angle_knee45(self):
        positions = self.positions
        positions[0][6] = [1, 0.5, 1.5]
        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        self.assertAlmostEqual(acm.calc_angles_knee(seq, self.bp["LeftKnee"], self.bp["LeftHip"], self.bp["LeftAnkle"])["flexion_extension"][0], 45)
