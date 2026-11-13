"""Tests for the session calibration of ScreenPointOfGaze (N2uCore/inference/screengaze.py).

  a. ``ScreenCalibration.fit``: the polynomial degree with the lowest leave-one-out error is kept,
     the report carries the cross-validated error, the stored mapping hits the targets;
  b. ``headPose``: eye position and span in image-width units; samples collected on the frame time;
  c. ``ScreenCalibration.drift``: the head pose is compared with the one during calibration, the move
     furthest beyond its threshold gives the hint (move_closer / move_back / move_left / move_right /
     move_up / move_down); a small sideways move is tolerated.

Pure numpy, no event loop, no model.

Run from project root:
    python3 -m unittest tests.dyn.test_screen_calibration -v
"""

import math
import unittest

import numpy as np

from N2uCore.inference.screengaze import ScreenCalibration, headPose, screenPoint

W, H = 1440, 900
GRID = [(W * (i + 0.5) / 4, H * (j + 0.5) / 4) for j in range(4) for i in range(4)]
HEAD = [0.5, 0.4, 0.3]
IRIS = [0.45, 0.05, 0.3]     # iris x, y in the eye and lid opening, steady


def calibrate(mapping, noise=0.0, seed=0, head=HEAD, iris=lambda x, y: IRIS):
    """Run a full 16-target calibration where ``mapping(x, y)`` gives the gaze angles of a screen point and
    ``iris(x, y)`` the iris features."""
    rng = np.random.default_rng(seed)
    cal = ScreenCalibration()
    cal.err = lambda *args: None
    assert cal.begin(W, H)
    for x, y in GRID:
        assert cal.target(x, y, 0)
        for _ in range(12):
            yaw, pitch = mapping(x, y)
            cal.feed([yaw + rng.normal(0, noise), pitch + rng.normal(0, noise), *iris(x, y)], 1.5, head)
        assert cal.finishTarget()
    return cal, cal.fit()


def linear(x, y):
    return (x - W / 2) / 2000, (y - H / 2) / 2000


def cubic(x, y):
    u, v = (x - W / 2) / W, (y - H / 2) / H
    return 0.4 * u + 0.3 * u**3, 0.3 * v + 0.2 * v**3 + 0.1 * u * u * v


class TestFit(unittest.TestCase):
    def test_noisy_linear_mapping_keeps_a_low_degree(self):
        cal, report = calibrate(linear, noise=math.radians(1))
        self.assertLess(report["degree"], 3)
        self.assertEqual(cal.calibration()["cx"][3:], [0.0] * 7)     # the horizontal stays a line

    def test_noiseless_cubic_mapping_keeps_the_cubic(self):
        cal, report = calibrate(cubic)
        self.assertEqual(report["degree"], 3)
        self.assertEqual(len(cal.calibration()["cx"]), 10)
        self.assertLess(report["cv_error_px"], 5.0)   # the inverse of a cubic is close to, not exactly, a cubic

    def test_report_has_cross_validated_error(self):
        _, report = calibrate(linear, noise=math.radians(1))
        self.assertGreaterEqual(report["cv_error_px"], report["fit_residual_px"])
        self.assertGreaterEqual(report["cv_p90_px"], 0)

    def test_stored_mapping_hits_the_targets(self):
        cal, _ = calibrate(linear)
        for x, y in GRID:
            px, py = screenPoint([*linear(x, y), *IRIS], cal.calibration())
            self.assertLessEqual(math.hypot(px - x, py - y), 2)

    def test_steady_iris_has_no_weight(self):
        cal, _ = calibrate(linear, noise=math.radians(1))
        np.testing.assert_allclose(cal.calibration()["cx_iris"] + cal.calibration()["cy_iris"], 0, atol=1e-6)

    def test_iris_corrects_what_the_angles_miss(self):
        # the pitch only carries noise, the iris follows the screen height: the iris terms must take the vertical
        flat = lambda x, y: ((x - W / 2) / 2000, 0.0)
        iris = lambda x, y: [0.0, (y - H / 2) / H * 0.2, 0.3]
        _, without = calibrate(flat, noise=math.radians(0.5))
        cal, report = calibrate(flat, noise=math.radians(0.5), iris=iris)
        self.assertLess(report["cv_error_px"], without["cv_error_px"] / 2)
        self.assertGreater(cal.calibration()["cy_iris"][1], 0)     # iris lower in the eye -> lower on the screen

    def test_stored_iris_coefficients_reproduce_the_fit(self):
        # raw iris coefficients with the centering in the constant term give the fitted points back
        iris = lambda x, y: [0.1 + (x - W / 2) / W * 0.3, (y - H / 2) / H * 0.2, 0.3 - (y - H / 2) / H * 0.05]
        cal, report = calibrate(cubic, iris=iris)
        errors = [math.hypot(*np.subtract(screenPoint([*cubic(x, y), *iris(x, y)], cal.calibration()), (x, y)))
                  for x, y in GRID]
        self.assertAlmostEqual(np.mean(errors), report["fit_residual_px"], delta=1.0)

    def test_reference_is_the_calibration_head_pose(self):
        cal, _ = calibrate(linear)
        self.assertEqual(cal.reference(), HEAD)


class TestOutliers(unittest.TestCase):
    def test_gaussian_samples_are_mostly_kept(self):
        rng = np.random.default_rng(1)
        cal = ScreenCalibration()
        cal.err = lambda *args: None
        cal.begin(W, H)
        cal.target(100, 100, 0)
        for yaw, pitch in rng.normal(0, 0.05, (40, 2)):
            cal.feed([yaw, pitch, *IRIS], 1.5, HEAD)
        kept = cal.finishTarget()["samples"]
        self.assertGreaterEqual(kept, 34)   # the old cut kept ~40%

    def test_a_far_sample_is_dropped(self):
        cal = ScreenCalibration()
        cal.err = lambda *args: None
        cal.begin(W, H)
        cal.target(100, 100, 0)
        for i in range(12):
            cal.feed([0.01 * (i % 3), 0.01 * (i % 2), *IRIS], 1.5, HEAD)
        cal.feed([1.0, 1.0, *IRIS], 1.5, HEAD)
        self.assertEqual(cal.finishTarget()["samples"], 12)


class TestHeadPose(unittest.TestCase):
    def test_pose_from_the_eyes_in_image_width_units(self):
        pose = headPose({"eye_center_px": [320, 240], "eye_span_px": 128}, {"w": 640, "h": 480})
        self.assertEqual(pose, [0.5, 0.375, 0.2])

    def test_collect_window_is_on_the_frame_time(self):
        cal = ScreenCalibration()
        cal.begin(W, H)
        cal.target(100, 100, 1000.0)
        for ts in (1000.5, 1001.5, 1002.5):          # before settling, inside, after the 2 s target
            cal.feed([0.1, 0.1, *IRIS], ts, HEAD)
        self.assertEqual(cal.status()["active_samples"], 1)


class TestDrift(unittest.TestCase):
    """Head pose [eyes x, eyes y, eye span] in image-width units; HEAD has an eye span of 0.3."""

    def setUp(self):
        self.cal, _ = calibrate(linear)

    def hint(self, x, y, span):
        return self.cal.drift([x, y, span])["hint"]

    def test_same_position_is_not_drift(self):
        drift = self.cal.drift([0.51, 0.41, 0.305])
        self.assertEqual((drift["drifted"], drift["hint"]), (False, "ok"))

    def test_farther_asks_to_move_closer(self):
        self.assertEqual(self.hint(0.5, 0.4, 0.26), "move_closer")      # 13% smaller eye span

    def test_closer_asks_to_move_back(self):
        self.assertEqual(self.hint(0.5, 0.4, 0.35), "move_back")

    def test_vertical_shift(self):
        self.assertEqual(self.hint(0.5, 0.5, 0.3), "move_up")           # eyes lower in the image
        self.assertEqual(self.hint(0.5, 0.3, 0.3), "move_down")

    def test_small_sideways_shift_is_tolerated(self):
        self.assertEqual(self.hint(0.7, 0.4, 0.3), "ok")                # 0.67 eye spans: it matters little

    def test_large_sideways_shift(self):
        self.assertEqual(self.hint(0.9, 0.4, 0.3), "move_right")        # eyes right in the image = user moved left
        self.assertEqual(self.hint(0.1, 0.4, 0.3), "move_left")

    def test_the_move_furthest_beyond_its_threshold_wins(self):
        # 10% farther (1.25x its threshold) and 0.5 eye spans down (2x its threshold)
        self.assertEqual(self.hint(0.5, 0.55, 0.27), "move_up")

    def test_drift_carries_both_poses_for_drawing(self):
        drift = self.cal.drift([0.55, 0.4, 0.3])
        self.assertEqual(drift["reference"], HEAD)
        self.assertEqual(drift["current"], [0.55, 0.4, 0.3])

    def test_a_cancelled_new_calibration_keeps_the_old_reference(self):
        self.cal.begin(W, H)
        self.cal.cancel()
        self.assertEqual(self.cal.reference(), HEAD)


if __name__ == "__main__":
    unittest.main()
