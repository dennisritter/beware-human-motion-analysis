import unittest
import numpy as np

from src.hma.movement_analysis import angle_calculations as acm
from src.hma.movement_analysis.Sequence import Sequence
from src.hma.movement_analysis.PoseFormatEnum import PoseFormatEnum


class TestAngleCalculationsMedicalHip(unittest.TestCase):

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
        self.positions_left = [
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
        self.positions_right = [
            [
                [0, 0, 0],        # "LeftWrist": 0,
                [0, 0, 0],        # "LeftElbow": 1,
                [0, 0, 0],        # "LeftShoulder": 2,
                [0, 0, 0],        # "Neck": 3,
                [1.5, 2, 1],      # "Torso": 4,
                [0, 0, 0],        # "Waist": 5,
                [0, 0, 0],        # "LeftAnkle": 6,
                [0, 0, 0],        # "LeftKnee": 7,
                [1, 1, 1],        # "LeftHip": 8,
                [0, 0, 0],        # "RightAnkle": 9,
                [2, 0.5, 1],      # "RightKnee": 10,
                [2, 1, 1],        # "RightHip": 11,
                [0, 0, 0],        # "RightWrist": 12,
                [0, 0, 0],        # "RightElbow": 13,
                [0, 0, 0],        # "RightShoulder": 14,
                [0, 0, 0],        # "Head": 15
            ]
        ]
    ### calc_angles_hip_left ###

    def test_calc_angles_hip_left_flex0_abd0(self):
        positions = self.positions_left
        positions[0][7] = [1, 0.5, 1]

        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        expected_result = {
            "flexion_extension": np.array([0.0]),
            "abduction_adduction": np.array([0.0]),
        }
        self.assertAlmostEqual(acm.calc_angles_hip_left(seq,
                                                        self.bp["LeftHip"],
                                                        self.bp["RightHip"],
                                                        self.bp["Torso"],
                                                        self.bp["LeftKnee"])["flexion_extension"][0],
                               expected_result["flexion_extension"][0])
        self.assertAlmostEqual(acm.calc_angles_hip_left(seq,
                                                        self.bp["LeftHip"],
                                                        self.bp["RightHip"],
                                                        self.bp["Torso"],
                                                        self.bp["LeftKnee"])["abduction_adduction"][0],
                               expected_result["abduction_adduction"][0])

    def test_calc_angles_hip_left_flex90_abd0(self):
        positions = self.positions_left
        positions[0][7] = [1, 1, 0.5]

        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        expected_result = {
            "flexion_extension": np.array([90.0]),
            "abduction_adduction": np.array([0.0]),
        }
        self.assertAlmostEqual(acm.calc_angles_hip_left(seq,
                                                        self.bp["LeftHip"],
                                                        self.bp["RightHip"],
                                                        self.bp["Torso"],
                                                        self.bp["LeftKnee"])["flexion_extension"][0],
                               expected_result["flexion_extension"][0])
        self.assertAlmostEqual(acm.calc_angles_hip_left(seq,
                                                        self.bp["LeftHip"],
                                                        self.bp["RightHip"],
                                                        self.bp["Torso"],
                                                        self.bp["LeftKnee"])["abduction_adduction"][0],
                               expected_result["abduction_adduction"][0])

    def test_calc_angles_hip_left_flex90n_abd0(self):
        positions = self.positions_left
        positions[0][7] = [1, 1, 1.5]

        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        expected_result = {
            "flexion_extension": np.array([-90.0]),
            "abduction_adduction": np.array([0.0]),
        }
        self.assertAlmostEqual(acm.calc_angles_hip_left(seq,
                                                        self.bp["LeftHip"],
                                                        self.bp["RightHip"],
                                                        self.bp["Torso"],
                                                        self.bp["LeftKnee"])["flexion_extension"][0],
                               expected_result["flexion_extension"][0])
        self.assertAlmostEqual(acm.calc_angles_hip_left(seq,
                                                        self.bp["LeftHip"],
                                                        self.bp["RightHip"],
                                                        self.bp["Torso"],
                                                        self.bp["LeftKnee"])["abduction_adduction"][0],
                               expected_result["abduction_adduction"][0])

    def test_calc_angles_hip_left_flex45_abd0(self):
        positions = self.positions_left
        positions[0][7] = [1, 0.5, 0.5]

        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        expected_result = {
            "flexion_extension": np.array([45.0]),
            "abduction_adduction": np.array([0.0]),
        }
        self.assertAlmostEqual(acm.calc_angles_hip_left(seq,
                                                        self.bp["LeftHip"],
                                                        self.bp["RightHip"],
                                                        self.bp["Torso"],
                                                        self.bp["LeftKnee"])["flexion_extension"][0],
                               expected_result["flexion_extension"][0])
        self.assertAlmostEqual(acm.calc_angles_hip_left(seq,
                                                        self.bp["LeftHip"],
                                                        self.bp["RightHip"],
                                                        self.bp["Torso"],
                                                        self.bp["LeftKnee"])["abduction_adduction"][0],
                               expected_result["abduction_adduction"][0])

    def test_calc_angles_hip_left_flex45n_abd0(self):
        positions = self.positions_left
        positions[0][7] = [1, 0.5, 1.5]

        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        expected_result = {
            "flexion_extension": np.array([-45.0]),
            "abduction_adduction": np.array([0.0]),
        }
        self.assertAlmostEqual(acm.calc_angles_hip_left(seq,
                                                        self.bp["LeftHip"],
                                                        self.bp["RightHip"],
                                                        self.bp["Torso"],
                                                        self.bp["LeftKnee"])["flexion_extension"][0],
                               expected_result["flexion_extension"][0])
        self.assertAlmostEqual(acm.calc_angles_hip_left(seq,
                                                        self.bp["LeftHip"],
                                                        self.bp["RightHip"],
                                                        self.bp["Torso"],
                                                        self.bp["LeftKnee"])["abduction_adduction"][0],
                               expected_result["abduction_adduction"][0])

    def test_calc_angles_hip_left_flex0_abd90(self):
        positions = self.positions_left
        positions[0][7] = [0.5, 1, 1]

        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        expected_result = {
            "flexion_extension": np.array([0.0]),
            "abduction_adduction": np.array([90.0]),
        }
        self.assertAlmostEqual(acm.calc_angles_hip_left(seq,
                                                        self.bp["LeftHip"],
                                                        self.bp["RightHip"],
                                                        self.bp["Torso"],
                                                        self.bp["LeftKnee"])["flexion_extension"][0],
                               expected_result["flexion_extension"][0])
        self.assertAlmostEqual(acm.calc_angles_hip_left(seq,
                                                        self.bp["LeftHip"],
                                                        self.bp["RightHip"],
                                                        self.bp["Torso"],
                                                        self.bp["LeftKnee"])["abduction_adduction"][0],
                               expected_result["abduction_adduction"][0])

    def test_calc_angles_hip_left_flex0_abd90n(self):
        positions = self.positions_left
        positions[0][7] = [1.5, 1, 1]

        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        expected_result = {
            "flexion_extension": np.array([0.0]),
            "abduction_adduction": np.array([-90.0]),
        }
        self.assertAlmostEqual(acm.calc_angles_hip_left(seq,
                                                        self.bp["LeftHip"],
                                                        self.bp["RightHip"],
                                                        self.bp["Torso"],
                                                        self.bp["LeftKnee"])["flexion_extension"][0],
                               expected_result["flexion_extension"][0])
        self.assertAlmostEqual(acm.calc_angles_hip_left(seq,
                                                        self.bp["LeftHip"],
                                                        self.bp["RightHip"],
                                                        self.bp["Torso"],
                                                        self.bp["LeftKnee"])["abduction_adduction"][0],
                               expected_result["abduction_adduction"][0])

    def test_calc_angles_hip_left_flex0_abd45(self):
        positions = self.positions_left
        positions[0][7] = [0.5, 0.5, 1]

        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        expected_result = {
            "flexion_extension": np.array([0.0]),
            "abduction_adduction": np.array([45.0]),
        }
        self.assertAlmostEqual(acm.calc_angles_hip_left(seq,
                                                        self.bp["LeftHip"],
                                                        self.bp["RightHip"],
                                                        self.bp["Torso"],
                                                        self.bp["LeftKnee"])["flexion_extension"][0],
                               expected_result["flexion_extension"][0])
        self.assertAlmostEqual(acm.calc_angles_hip_left(seq,
                                                        self.bp["LeftHip"],
                                                        self.bp["RightHip"],
                                                        self.bp["Torso"],
                                                        self.bp["LeftKnee"])["abduction_adduction"][0],
                               expected_result["abduction_adduction"][0])

    def test_calc_angles_hip_left_flex0_abd45n(self):
        positions = self.positions_left
        positions[0][7] = [1.5, 0.5, 1]

        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        expected_result = {
            "flexion_extension": np.array([0.0]),
            "abduction_adduction": np.array([-45.0]),
        }
        self.assertAlmostEqual(acm.calc_angles_hip_left(seq,
                                                        self.bp["LeftHip"],
                                                        self.bp["RightHip"],
                                                        self.bp["Torso"],
                                                        self.bp["LeftKnee"])["flexion_extension"][0],
                               expected_result["flexion_extension"][0])
        self.assertAlmostEqual(acm.calc_angles_hip_left(seq,
                                                        self.bp["LeftHip"],
                                                        self.bp["RightHip"],
                                                        self.bp["Torso"],
                                                        self.bp["LeftKnee"])["abduction_adduction"][0],
                               expected_result["abduction_adduction"][0])

    def test_calc_angles_hip_left_flex90_abd45(self):
        positions = self.positions_left
        positions[0][7] = [0.5, 1, 0.5]

        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        expected_result = {
            "flexion_extension": np.array([90.0]),
            "abduction_adduction": np.array([45.0]),
        }
        self.assertAlmostEqual(acm.calc_angles_hip_left(seq,
                                                        self.bp["LeftHip"],
                                                        self.bp["RightHip"],
                                                        self.bp["Torso"],
                                                        self.bp["LeftKnee"])["flexion_extension"][0],
                               expected_result["flexion_extension"][0])
        self.assertAlmostEqual(acm.calc_angles_hip_left(seq,
                                                        self.bp["LeftHip"],
                                                        self.bp["RightHip"],
                                                        self.bp["Torso"],
                                                        self.bp["LeftKnee"])["abduction_adduction"][0],
                               expected_result["abduction_adduction"][0])

    def test_calc_angles_hip_left_flex90n_abd45n(self):
        positions = self.positions_left
        positions[0][7] = [1.5, 1, 1.5]

        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        expected_result = {
            "flexion_extension": np.array([-90.0]),
            "abduction_adduction": np.array([-45.0]),
        }
        self.assertAlmostEqual(acm.calc_angles_hip_left(seq,
                                                        self.bp["LeftHip"],
                                                        self.bp["RightHip"],
                                                        self.bp["Torso"],
                                                        self.bp["LeftKnee"])["flexion_extension"][0],
                               expected_result["flexion_extension"][0])
        self.assertAlmostEqual(acm.calc_angles_hip_left(seq,
                                                        self.bp["LeftHip"],
                                                        self.bp["RightHip"],
                                                        self.bp["Torso"],
                                                        self.bp["LeftKnee"])["abduction_adduction"][0],
                               expected_result["abduction_adduction"][0])

    ### calc_angles_hip_right ###

    def test_calc_angles_hip_right_flex0_abd0(self):
        positions = self.positions_right
        positions[0][10] = [2, 0.5, 1]

        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        expected_result = {
            "flexion_extension": np.array([0.0]),
            "abduction_adduction": np.array([0.0]),
        }
        self.assertAlmostEqual(acm.calc_angles_hip_right(seq,
                                                         self.bp["RightHip"],
                                                         self.bp["LeftHip"],
                                                         self.bp["Torso"],
                                                         self.bp["RightKnee"])["flexion_extension"][0],
                               expected_result["flexion_extension"][0])
        self.assertAlmostEqual(acm.calc_angles_hip_right(seq,
                                                         self.bp["RightHip"],
                                                         self.bp["LeftHip"],
                                                         self.bp["Torso"],
                                                         self.bp["RightKnee"])["abduction_adduction"][0],
                               expected_result["abduction_adduction"][0])

    def test_calc_angles_hip_right_flex90_abd0(self):
        positions = self.positions_right
        positions[0][10] = [2, 1, 0.5]

        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        expected_result = {
            "flexion_extension": np.array([90.0]),
            "abduction_adduction": np.array([0.0]),
        }
        self.assertAlmostEqual(acm.calc_angles_hip_right(seq,
                                                         self.bp["RightHip"],
                                                         self.bp["LeftHip"],
                                                         self.bp["Torso"],
                                                         self.bp["RightKnee"])["flexion_extension"][0],
                               expected_result["flexion_extension"][0])
        self.assertAlmostEqual(acm.calc_angles_hip_right(seq,
                                                         self.bp["RightHip"],
                                                         self.bp["LeftHip"],
                                                         self.bp["Torso"],
                                                         self.bp["RightKnee"])["abduction_adduction"][0],
                               expected_result["abduction_adduction"][0])

    def test_calc_angles_hip_right_flex90n_abd0(self):
        positions = self.positions_right
        positions[0][10] = [2, 1, 1.5]

        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        expected_result = {
            "flexion_extension": np.array([-90.0]),
            "abduction_adduction": np.array([0.0]),
        }
        self.assertAlmostEqual(acm.calc_angles_hip_right(seq,
                                                         self.bp["RightHip"],
                                                         self.bp["LeftHip"],
                                                         self.bp["Torso"],
                                                         self.bp["RightKnee"])["flexion_extension"][0],
                               expected_result["flexion_extension"][0])
        self.assertAlmostEqual(acm.calc_angles_hip_right(seq,
                                                         self.bp["RightHip"],
                                                         self.bp["LeftHip"],
                                                         self.bp["Torso"],
                                                         self.bp["RightKnee"])["abduction_adduction"][0],
                               expected_result["abduction_adduction"][0])

    def test_calc_angles_hip_right_flex45_abd0(self):
        positions = self.positions_right
        positions[0][10] = [2, 0.5, 0.5]

        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        expected_result = {
            "flexion_extension": np.array([45.0]),
            "abduction_adduction": np.array([0.0]),
        }
        self.assertAlmostEqual(acm.calc_angles_hip_right(seq,
                                                         self.bp["RightHip"],
                                                         self.bp["LeftHip"],
                                                         self.bp["Torso"],
                                                         self.bp["RightKnee"])["flexion_extension"][0],
                               expected_result["flexion_extension"][0])
        self.assertAlmostEqual(acm.calc_angles_hip_right(seq,
                                                         self.bp["RightHip"],
                                                         self.bp["LeftHip"],
                                                         self.bp["Torso"],
                                                         self.bp["RightKnee"])["abduction_adduction"][0],
                               expected_result["abduction_adduction"][0])

    def test_calc_angles_hip_right_flex45n_abd0(self):
        positions = self.positions_right
        positions[0][10] = [2, 0.5, 1.5]

        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        expected_result = {
            "flexion_extension": np.array([-45.0]),
            "abduction_adduction": np.array([0.0]),
        }
        self.assertAlmostEqual(acm.calc_angles_hip_right(seq,
                                                         self.bp["RightHip"],
                                                         self.bp["LeftHip"],
                                                         self.bp["Torso"],
                                                         self.bp["RightKnee"])["flexion_extension"][0],
                               expected_result["flexion_extension"][0])
        self.assertAlmostEqual(acm.calc_angles_hip_right(seq,
                                                         self.bp["RightHip"],
                                                         self.bp["LeftHip"],
                                                         self.bp["Torso"],
                                                         self.bp["RightKnee"])["abduction_adduction"][0],
                               expected_result["abduction_adduction"][0])

    def test_calc_angles_hip_right_flex0_abd90(self):
        positions = self.positions_right
        positions[0][10] = [2.5, 1, 1]

        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        expected_result = {
            "flexion_extension": np.array([0.0]),
            "abduction_adduction": np.array([90.0]),
        }
        self.assertAlmostEqual(acm.calc_angles_hip_right(seq,
                                                         self.bp["RightHip"],
                                                         self.bp["LeftHip"],
                                                         self.bp["Torso"],
                                                         self.bp["RightKnee"])["flexion_extension"][0],
                               expected_result["flexion_extension"][0])
        self.assertAlmostEqual(acm.calc_angles_hip_right(seq,
                                                         self.bp["RightHip"],
                                                         self.bp["LeftHip"],
                                                         self.bp["Torso"],
                                                         self.bp["RightKnee"])["abduction_adduction"][0],
                               expected_result["abduction_adduction"][0])

    def test_calc_angles_hip_right_flex0_abd90n(self):
        positions = self.positions_right
        positions[0][10] = [1.5, 1, 1]

        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        expected_result = {
            "flexion_extension": np.array([0.0]),
            "abduction_adduction": np.array([-90.0]),
        }
        self.assertAlmostEqual(acm.calc_angles_hip_right(seq,
                                                         self.bp["RightHip"],
                                                         self.bp["LeftHip"],
                                                         self.bp["Torso"],
                                                         self.bp["RightKnee"])["flexion_extension"][0],
                               expected_result["flexion_extension"][0])
        self.assertAlmostEqual(acm.calc_angles_hip_right(seq,
                                                         self.bp["RightHip"],
                                                         self.bp["LeftHip"],
                                                         self.bp["Torso"],
                                                         self.bp["RightKnee"])["abduction_adduction"][0],
                               expected_result["abduction_adduction"][0])

    def test_calc_angles_hip_right_flex0_abd45(self):
        positions = self.positions_right
        positions[0][10] = [2.5, 0.5, 1]

        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        expected_result = {
            "flexion_extension": np.array([0.0]),
            "abduction_adduction": np.array([45.0]),
        }
        self.assertAlmostEqual(acm.calc_angles_hip_right(seq,
                                                         self.bp["RightHip"],
                                                         self.bp["LeftHip"],
                                                         self.bp["Torso"],
                                                         self.bp["RightKnee"])["flexion_extension"][0],
                               expected_result["flexion_extension"][0])
        self.assertAlmostEqual(acm.calc_angles_hip_right(seq,
                                                         self.bp["RightHip"],
                                                         self.bp["LeftHip"],
                                                         self.bp["Torso"],
                                                         self.bp["RightKnee"])["abduction_adduction"][0],
                               expected_result["abduction_adduction"][0])

    def test_calc_angles_hip_right_flex0_abd45n(self):
        positions = self.positions_right
        positions[0][10] = [1.5, 0.5, 1]

        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        expected_result = {
            "flexion_extension": np.array([0.0]),
            "abduction_adduction": np.array([-45.0]),
        }
        self.assertAlmostEqual(acm.calc_angles_hip_right(seq,
                                                         self.bp["RightHip"],
                                                         self.bp["LeftHip"],
                                                         self.bp["Torso"],
                                                         self.bp["RightKnee"])["flexion_extension"][0],
                               expected_result["flexion_extension"][0])
        self.assertAlmostEqual(acm.calc_angles_hip_right(seq,
                                                         self.bp["RightHip"],
                                                         self.bp["LeftHip"],
                                                         self.bp["Torso"],
                                                         self.bp["RightKnee"])["abduction_adduction"][0],
                               expected_result["abduction_adduction"][0])

    def test_calc_angles_hip_right_flex90_abd45(self):
        positions = self.positions_right
        positions[0][10] = [2.5, 1, 0.5]

        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        expected_result = {
            "flexion_extension": np.array([90.0]),
            "abduction_adduction": np.array([45.0]),
        }
        self.assertAlmostEqual(acm.calc_angles_hip_right(seq,
                                                         self.bp["RightHip"],
                                                         self.bp["LeftHip"],
                                                         self.bp["Torso"],
                                                         self.bp["RightKnee"])["flexion_extension"][0],
                               expected_result["flexion_extension"][0])
        self.assertAlmostEqual(acm.calc_angles_hip_right(seq,
                                                         self.bp["RightHip"],
                                                         self.bp["LeftHip"],
                                                         self.bp["Torso"],
                                                         self.bp["RightKnee"])["abduction_adduction"][0],
                               expected_result["abduction_adduction"][0])

    def test_calc_angles_hip_right_flex90n_abd45n(self):
        positions = self.positions_right
        positions[0][10] = [1.5, 1, 1.5]

        seq = Sequence(self.bp, positions, [0.0], PoseFormatEnum.MOCAP)
        expected_result = {
            "flexion_extension": np.array([-90.0]),
            "abduction_adduction": np.array([-45.0]),
        }
        self.assertAlmostEqual(acm.calc_angles_hip_right(seq,
                                                         self.bp["RightHip"],
                                                         self.bp["LeftHip"],
                                                         self.bp["Torso"],
                                                         self.bp["RightKnee"])["flexion_extension"][0],
                               expected_result["flexion_extension"][0])
        self.assertAlmostEqual(acm.calc_angles_hip_right(seq,
                                                         self.bp["RightHip"],
                                                         self.bp["LeftHip"],
                                                         self.bp["Torso"],
                                                         self.bp["RightKnee"])["abduction_adduction"][0],
                               expected_result["abduction_adduction"][0])
