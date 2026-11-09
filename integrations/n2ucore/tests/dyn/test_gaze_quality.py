"""Tests for the real-time quality of the gaze nodes.

  a. ``GazeDinoEngine._nearestFace``: the biggest (nearest) face, kept while followed unless another
     is clearly bigger; a face cut by the border is flagged;
  b. ``GazeDinoEngine._quality``: score and hint from face size (move_closer / move_back)
     and from the jitter of the raw estimate (unstable), a steady head turn not counting;
  c. ``GazeDinoEngine._eyeGeometry``: eye midpoint and 3D eye-corner distance (camera distance);
  c2. ``GazeDinoEngine.analyze``: the smoothing weight follows the time between estimates;
  c3. ``GazeDinoEngine.analyze``: with the eyes closed (blink) the last estimate is kept;
  c4. ``GazeDinoEngine._predictCrop``: the crop reaches the model in its precision (fp16 on a GPU), angles in float;
  d. ``ScreenPointOfGaze._process``: the upstream quality travels with the point.

Built without model, camera or event loop (``__new__`` + attributes, SimpleNamespace
landmarks), like tests/dyn/test_eyegaze.py.

Run from project root:
    python3 -m unittest tests.dyn.test_gaze_quality -v
"""

import math
import time
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from N2uCore.dyn.screenpointofgaze import ScreenPointOfGaze
from N2uCore.inference.gazedinoengine import GazeDinoEngine


def face(x1, y1, x2, y2):
    """Landmarks of a face filling the normalized box (two corner points are enough)."""
    return SimpleNamespace(landmark=[SimpleNamespace(x=x1, y=y1), SimpleNamespace(x=x2, y=y2)])


def engine():
    e = GazeDinoEngine.__new__(GazeDinoEngine)
    e._raw = deque(maxlen=GazeDinoEngine.JITTER_WINDOW)
    e._bbox = None
    return e


class TestNearestFace(unittest.TestCase):
    def test_biggest_face_is_chosen(self):
        far, near = face(0.05, 0.1, 0.15, 0.3), face(0.5, 0.2, 0.8, 0.7)
        points, box, edge = engine()._nearestFace([far, near], 640, 480)
        self.assertIs(points, near.landmark)
        self.assertEqual(box, [320, 96, 512, 336])
        self.assertFalse(edge)

    def test_followed_face_is_kept_against_a_similar_one(self):
        e = engine()
        followed, other = face(0.1, 0.2, 0.4, 0.6), face(0.55, 0.2, 0.87, 0.62)   # other ~7% bigger
        e._bbox = [64, 96, 256, 288]
        points, _, _ = e._nearestFace([other, followed], 640, 480)
        self.assertIs(points, followed.landmark)

    def test_clearly_bigger_face_takes_over(self):
        e = engine()
        followed, near = face(0.1, 0.2, 0.3, 0.45), face(0.5, 0.1, 0.9, 0.8)
        e._bbox = [64, 96, 192, 216]
        points, _, _ = e._nearestFace([followed, near], 640, 480)
        self.assertIs(points, near.landmark)

    def test_face_cut_by_the_border_is_flagged(self):
        _, box, edge = engine()._nearestFace([face(-0.1, 0.2, 0.3, 0.7)], 640, 480)
        self.assertTrue(edge)
        self.assertEqual(box[0], 0)

    def test_empty_boxes_are_skipped(self):
        self.assertEqual(engine()._nearestFace([face(0.5, 0.5, 0.5, 0.5)], 640, 480), (None, None, False))


class TestQuality(unittest.TestCase):
    def test_normal_distance_still_head_is_full_quality(self):
        e = engine()
        for _ in range(GazeDinoEngine.JITTER_MIN_SAMPLES):
            q = e._quality(np.array([0.1, 0.0]), 0.25)
        self.assertEqual((q["score"], q["hint"]), (100, "ok"))
        self.assertAlmostEqual(q["jitter_deg"], 0.0)

    def warm(self, ratio):
        e = engine()
        for _ in range(GazeDinoEngine.JITTER_MIN_SAMPLES):
            q = e._quality(np.array([0.0, 0.0]), ratio)
        return q

    def test_small_face_asks_to_move_closer(self):
        q = self.warm(0.12)
        self.assertEqual(q["hint"], "move_closer")
        self.assertLess(q["score"], 100)
        self.assertEqual(self.warm(0.06)["score"], 0)

    def test_large_face_asks_to_move_back(self):
        q = self.warm(0.7)
        self.assertEqual(q["hint"], "move_back")
        self.assertLess(q["score"], 100)

    def test_no_score_while_warming_up(self):
        q = engine()._quality(np.array([0.0, 0.0]), 0.25)
        self.assertIsNone(q["jitter_deg"])
        self.assertIsNone(q["score"])

    def test_face_at_the_border_asks_to_center(self):
        self.assertEqual(engine()._quality(np.array([0.0, 0.0]), 0.25, atEdge=True)["hint"], "center_face")

    def test_jittery_estimate_is_unstable(self):
        e = engine()
        step = math.radians(20)
        for i in range(GazeDinoEngine.JITTER_WINDOW):
            q = e._quality(np.array([step * (i % 2), 0.0]), 0.25)
        self.assertEqual(q["hint"], "unstable")
        self.assertAlmostEqual(q["jitter_deg"], 40.0, places=3)
        self.assertLess(q["score"], 50)

    def test_steady_head_turn_is_not_jitter(self):
        e = engine()
        for i in range(GazeDinoEngine.JITTER_WINDOW):
            q = e._quality(np.array([math.radians(4) * i, math.radians(-2) * i]), 0.25)
        self.assertEqual((q["score"], q["hint"]), (100, "ok"))
        self.assertAlmostEqual(q["jitter_deg"], 0.0, places=6)

    def test_a_single_saccade_does_not_lower_quality(self):
        e = engine()
        for i in range(GazeDinoEngine.JITTER_WINDOW):
            q = e._quality(np.array([0.0 if i < 8 else math.radians(20), 0.0]), 0.25)
        self.assertEqual((q["score"], q["hint"]), (100, "ok"))

    def test_reset_forgets_the_jitter_history(self):
        e = engine()
        e._ema = e._bbox = None
        for _ in range(GazeDinoEngine.JITTER_MIN_SAMPLES):
            e._quality(np.array([0.0, 0.0]), 0.25)
        e.reset()
        self.assertEqual(len(e._raw), 0)


class TestEyeGeometry(unittest.TestCase):
    def landmarks(self, depth=0.0, lid=0.01):
        """478 points with the eye corners 33, 133, 362, 263 and the lids set; ``depth``: z of the outer right
        corner; ``lid``: lid distance from the corner line (0.01: eye aspect ratio 0.3 at 640x480, 0: closed)."""
        points = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
        points[33] = SimpleNamespace(x=0.40, y=0.40, z=0.0)
        points[133] = SimpleNamespace(x=0.45, y=0.40, z=0.0)
        points[362] = SimpleNamespace(x=0.55, y=0.40, z=0.0)
        points[263] = SimpleNamespace(x=0.60, y=0.40, z=depth)
        for upper, lower, x in ((160, 144, 0.42), (158, 153, 0.43), (385, 380, 0.57), (387, 373, 0.58)):
            points[upper] = SimpleNamespace(x=x, y=0.40 - lid, z=0.0)
            points[lower] = SimpleNamespace(x=x, y=0.40 + lid, z=0.0)
        return points

    def test_midpoint_and_span_in_pixels(self):
        x, y, span = engine()._eyeGeometry(self.landmarks(), 640, 480)
        self.assertAlmostEqual(x, 320.0)
        self.assertAlmostEqual(y, 192.0)
        self.assertAlmostEqual(span, 128.0)

    def test_span_keeps_the_depth_of_a_turned_head(self):
        # turned head: the outer corners get closer in the image but one moves back in depth
        flat = engine()._eyeGeometry(self.landmarks(), 640, 480)[2]
        turned = engine()._eyeGeometry(self.landmarks(depth=0.1), 640, 480)[2]
        self.assertAlmostEqual(turned, math.hypot(flat, 64.0))

    def irises(self, dx=0.0, dy=0.0):
        """landmarks with both irises in the middle of their eye, moved by ``dx``, ``dy`` (image fractions)."""
        points = self.landmarks()
        points[468] = SimpleNamespace(x=0.425 + dx, y=0.40 + dy, z=0.0)
        points[473] = SimpleNamespace(x=0.575 + dx, y=0.40 + dy, z=0.0)
        return points

    def test_iris_in_the_middle_of_both_eyes(self):
        np.testing.assert_allclose(engine()._irisPosition(self.irises(), 640, 480), [0.0, 0.0], atol=1e-9)

    def test_iris_position_in_eye_widths(self):
        # eye width 32 px: 6.4 px to the right is 0.2 for both eyes; 2.4 px up is -0.075
        rx, ry = engine()._irisPosition(self.irises(dx=0.01, dy=-0.005), 640, 480)
        self.assertAlmostEqual(rx, 0.2)
        self.assertAlmostEqual(ry, -0.075)


class TestSmoothing(unittest.TestCase):
    def analyze(self, e, yaw, ts, lid=0.01):
        e._predictCrop = MagicMock(return_value=np.array([yaw, 0.0]))
        e._mesh.process.return_value = SimpleNamespace(multi_face_landmarks=[SimpleNamespace(
            landmark=TestEyeGeometry.landmarks(None, lid=lid))])
        return e.analyze(np.zeros((480, 640, 3), np.uint8), ts)["yaw_rad"]

    def engine(self):
        e = engine()
        e._alpha = 0.25
        e._ema = e._emaEyes = e._emaTs = e._last = None
        e._mesh = MagicMock()
        return e

    def test_alpha_is_the_weight_at_15_estimates_per_second(self):
        e = self.engine()
        self.analyze(e, 0.0, 0)
        self.assertAlmostEqual(self.analyze(e, 1.0, 1e6 / 15), 0.25)

    def test_slower_estimates_weigh_more(self):
        # 5 estimates per second: three 15/s steps, the same smoothing time
        e = self.engine()
        self.analyze(e, 0.0, 0)
        self.assertAlmostEqual(self.analyze(e, 1.0, 1e6 / 5), 1 - 0.75 ** 3)

    def test_sample_carries_iris_and_lid_opening(self):
        e = self.engine()
        self.analyze(e, 0.0, 0)
        iris = e._last["iris"]
        self.assertEqual(len(iris), 3)
        self.assertAlmostEqual(iris[2], 0.3)       # eye aspect ratio of the open lids of the landmarks

    def test_open_eyes_are_not_a_blink(self):
        self.assertAlmostEqual(engine()._eyeOpening(TestEyeGeometry.landmarks(None), 640, 480), 0.3)

    def test_blink_keeps_the_last_estimate(self):
        # closed lids read as looking down: the estimate is not used and the model does not run
        e = self.engine()
        self.analyze(e, 0.0, 0)
        self.assertEqual(self.analyze(e, 1.0, 1e6 / 15, lid=0.0), 0.0)
        e._predictCrop.assert_not_called()

    def test_after_a_blink_the_new_estimate_weighs_for_the_time_passed(self):
        e = self.engine()
        self.analyze(e, 0.0, 0)
        self.analyze(e, 1.0, 1e6 / 15, lid=0.0)
        self.assertAlmostEqual(self.analyze(e, 1.0, 2e6 / 15), 1 - 0.75 ** 2)

    def test_blink_on_the_first_estimate_is_still_used(self):
        # nothing to keep yet
        self.assertEqual(self.analyze(self.engine(), 1.0, 0, lid=0.0), 1.0)


class TestPrecision(unittest.TestCase):
    def predict(self, dtype):
        e = engine()
        e._device, e._dtype = torch.device("cpu"), dtype
        e._transform = lambda rgb: torch.zeros(3, 4, 4)
        e._model = MagicMock(return_value=torch.tensor([[0.1, -0.2]], dtype=dtype))
        angles = e._predictCrop(np.zeros((4, 4, 3), np.uint8))
        return e._model.call_args[0][0].dtype, angles

    def test_half_precision_crop_and_float_angles(self):
        sent, angles = self.predict(torch.float16)
        self.assertEqual(sent, torch.float16)
        self.assertEqual(angles.dtype, np.float64)
        np.testing.assert_allclose(angles, [0.1, -0.2], atol=1e-3)

    def test_full_precision_crop(self):
        self.assertEqual(self.predict(torch.float32)[0], torch.float32)

    def test_half_precision_only_on_a_gpu(self):
        self.assertEqual(set(GazeDinoEngine.HALF_DEVICES), set(GazeDinoEngine.SUPPORTED_DEVICES) - {"cpu"})


class TestScreenQuality(unittest.TestCase):
    def node(self):
        n = ScreenPointOfGaze.__new__(ScreenPointOfGaze)
        n._cal = MagicMock()
        n._cal.calibration.return_value = None
        n._imageSize = {"w": 640, "h": 480}
        return n

    def sample(self, **extra):
        return {"schema": "gaze-direction.v1", "status": "ok", "yaw_rad": 0.1, "pitch_rad": 0.0,
                "eye_center_px": [320.0, 200.0], "eye_span_px": 110.0, "iris": [0.0, 0.05, 0.3],
                "bbox_xyxy_px": [220, 100, 420, 340], "quality": None, **extra}

    def test_quality_travels_with_the_point(self):
        quality = {"score": 80, "hint": "ok", "face_size_ratio": 0.2, "jitter_deg": 0.5}
        out = self.node()._process(self.sample(quality=quality), time.time_ns() // 1000)
        self.assertEqual(out["status"], "uncalibrated")
        self.assertEqual(out["quality"], quality)

    def test_quality_travels_even_when_not_measured_yet(self):
        out = self.node()._process(self.sample(), time.time_ns() // 1000)
        self.assertIsNone(out["quality"])


if __name__ == "__main__":
    unittest.main()
