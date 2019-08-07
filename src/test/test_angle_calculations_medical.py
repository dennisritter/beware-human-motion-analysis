import unittest
import numpy as np
from movement_analysis import angle_calculations_medical as acm


class TestAngleCalculationsMedical(unittest.TestCase):

    def test_calc_angle_0(self):
        self.assertEqual(acm.calc_angle([0, 0, 0], [1, 0, 0], [1, 0, 0]), 0)

    def test_calc_angle_90(self):
        self.assertEqual(acm.calc_angle([0, 0, 0], [1, 0, 0], [0, 1, 0]), 90.0)

    def test_calc_angle_180(self):
        self.assertEqual(acm.calc_angle([0, 0, 0], [1, 0, 0], [-1, 0, 0]), 180.0)

    def test_calc_angle_45(self):
        self.assertAlmostEqual(acm.calc_angle([0, 0, 0], [1, 0, 0], [1, 1, 0]), 45.0)


if __name__ == '__main__':
    unittest.main()
