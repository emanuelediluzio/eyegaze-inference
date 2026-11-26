import unittest

import numpy as np

import r1stats as s


class R1Stats(unittest.TestCase):
    def test_draws_fixed_per_person(self):
        a, b = s.calibrationDraws("mpii", "p10", 300), s.calibrationDraws("mpii", "p10", 300)
        self.assertEqual(len(a), 20)
        self.assertTrue(all((x == y).all() and len(set(x)) == 16 for x, y in zip(a, b)))
        self.assertFalse((a[0] == s.calibrationDraws("mpii", "p11", 300)[0]).all())

    def test_linear_bias_calibrated_away(self):
        rng = np.random.default_rng(1)
        true = rng.uniform(-0.4, 0.4, (300, 2))
        pred = 0.8 * true + 0.05                          # a per-axis line recovers it exactly
        error, frames = s.calibratedError(pred, true, s.calibrationDraws("x", "p", 300))
        self.assertLess(error, 1e-6)
        self.assertLessEqual(len(frames), 300)

    def test_calibration_frames_not_scored(self):
        true = np.zeros((40, 2))
        pred = np.zeros((40, 2))
        draws = s.calibrationDraws("x", "p", 40)
        pred[draws[0]] = 1.0                               # wrong only on draw 0's calibration frames
        _, frames = s.calibratedError(pred, true, draws[:1])
        self.assertEqual(len(frames), 24)

    def test_paired_difference(self):
        mean, lo, hi = s.pairedDifference({"a": 5.0, "b": 6.0, "c": 7.0}, {"a": 6.0, "b": 7.0, "c": 8.0})
        self.assertAlmostEqual(mean, -1.0)
        self.assertAlmostEqual(lo, -1.0)
        self.assertAlmostEqual(hi, -1.0)
        with self.assertRaises(ValueError):
            s.pairedDifference({"a": 1.0}, {"b": 1.0})

    def test_gap_and_bands(self):
        self.assertAlmostEqual(s.angularError(np.array([[0.0, 0.0]]), np.array([[np.radians(10), 0.0]]))[0], 10, 5)
        errors = np.ones(60)
        people = np.array(["a"] * 30 + ["b"] * 30)
        yaw = np.array([5.0] * 25 + [15.0] * 5 + [5.0] * 30)
        rows = s.perBand(errors, people, yaw, s.YAW_BANDS, absolute=True)
        self.assertEqual(rows[0][3], 2)                    # both have >= 20 frames under 10 deg
        self.assertEqual(rows[1][3], 0)                    # 5 frames are too few

    def test_evenly_spaced(self):
        self.assertEqual(list(s.evenlySpaced(5, 10)), [0, 1, 2, 3, 4])
        self.assertEqual(list(s.evenlySpaced(11, 3)), [0, 5, 10])


if __name__ == "__main__":
    unittest.main()
