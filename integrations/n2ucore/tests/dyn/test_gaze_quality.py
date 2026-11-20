"""Tests for the real-time quality of the gaze nodes.

  a. ``GazeDinoEngine._faceBox``: the face box in pixels from the tracking node's landmarks; a face cut by
     the border is flagged;
  b. ``GazeDinoEngine._quality``: score and hint from face size (move_closer / move_back)
     and from the jitter of the raw estimate (unstable), a steady head turn not counting;
  c. ``GazeDinoEngine._eyeGeometry``: eye midpoint and 3D eye-corner distance (camera distance);
  c2. ``GazeDinoEngine.analyze``: the smoothing weight follows the time between estimates;
  c3. ``GazeDinoEngine.analyze``: with the eyes closed (blink) the last estimate is kept; the all-zero mesh of
      the tracking node is no face, a mesh of another size is refused;
  c4. ``GazeDinoEngine._predictCrop``: the crop reaches the model in its precision (fp16 on a GPU), angles in float;
  c5. ``GazeDinoEngine._loadCoreML``: a Core ML package runs on all compute units (CPU only with the CPU device),
      takes the crop as fp32 and gives float angles;
  d. ``ScreenPointOfGaze._process``: the upstream quality travels with the point.

Built without model, camera or event loop (``__new__`` + attributes, landmarks as the (478, 3) float32 array
of the Landmarks channel), like tests/dyn/test_eyegaze.py.

Run from project root:
    python3 -m unittest tests.dyn.test_gaze_quality -v
"""

import math
import sys
import time
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from N2uCore.dyn.screenpointofgaze import ScreenPointOfGaze
from N2uCore.inference.gazedinoengine import GazeDinoEngine


def face(x1, y1, x2, y2):
    """Landmarks of a face filling the normalized box (two corner points are enough), as the Landmarks channel."""
    return np.array([[x1, y1, 0.0], [x2, y2, 0.0]], np.float32)


def engine():
    e = GazeDinoEngine.__new__(GazeDinoEngine)
    e._raw = deque(maxlen=GazeDinoEngine.JITTER_WINDOW)
    e._bbox = None
    return e


class TestFaceBox(unittest.TestCase):
    def test_box_in_pixels(self):
        box, edge = engine()._faceBox(face(0.5, 0.2, 0.8, 0.7), 640, 480)
        self.assertEqual(box, [320, 96, 512, 336])
        self.assertFalse(edge)

    def test_face_cut_by_the_border_is_flagged(self):
        box, edge = engine()._faceBox(face(-0.1, 0.2, 0.3, 0.7), 640, 480)
        self.assertTrue(edge)
        self.assertEqual(box[0], 0)

    def test_empty_box_is_no_face(self):
        self.assertEqual(engine()._faceBox(face(0.5, 0.5, 0.5, 0.5), 640, 480), (None, False))


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
        points = np.tile(np.array([0.5, 0.5, 0.0]), (GazeDinoEngine.MESH_POINTS, 1))
        points[33] = [0.40, 0.40, 0.0]
        points[133] = [0.45, 0.40, 0.0]
        points[362] = [0.55, 0.40, 0.0]
        points[263] = [0.60, 0.40, depth]
        for upper, lower, x in ((160, 144, 0.42), (158, 153, 0.43), (385, 380, 0.57), (387, 373, 0.58)):
            points[upper] = [x, 0.40 - lid, 0.0]
            points[lower] = [x, 0.40 + lid, 0.0]
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
        points[468] = [0.425 + dx, 0.40 + dy, 0.0]
        points[473] = [0.575 + dx, 0.40 + dy, 0.0]
        return points

    def test_iris_in_the_middle_of_both_eyes(self):
        np.testing.assert_allclose(engine()._irisPosition(self.irises(), 640, 480), [0.0, 0.0], atol=1e-9)

    def test_iris_position_in_eye_widths(self):
        # eye width 32 px: 6.4 px to the right is 0.2 for both eyes; 2.4 px up is -0.075
        rx, ry = engine()._irisPosition(self.irises(dx=0.01, dy=-0.005), 640, 480)
        self.assertAlmostEqual(rx, 0.2)
        self.assertAlmostEqual(ry, -0.075)


class TestCoreML(unittest.TestCase):
    def load(self, deviceName):
        ct = SimpleNamespace(ComputeUnit=SimpleNamespace(ALL="all", CPU_ONLY="cpu_only"),
                             models=SimpleNamespace(MLModel=MagicMock()))
        ct.models.MLModel.return_value.predict.return_value = {"gaze": np.array([[0.1, -0.2]], np.float16)}
        e = GazeDinoEngine(Path("model.mlpackage"), deviceName=deviceName)
        e._checkpoint = Path("model.mlpackage")
        with patch.dict(sys.modules, {"coremltools": ct}):
            self.assertTrue(e._loadCoreML())
        return e, ct.models.MLModel

    def test_package_runs_on_all_units_and_gives_float_angles(self):
        e, MLModel = self.load("auto")
        MLModel.assert_called_once_with("model.mlpackage", compute_units="all")
        self.assertEqual((e._device.type, e._dtype), ("cpu", torch.float32))
        out = e._model(torch.zeros(1, 3, 224, 224))
        self.assertEqual(out.dtype, torch.float32)
        np.testing.assert_allclose(out.numpy(), [[0.1, -0.2]], atol=1e-3)
        crop = MLModel.return_value.predict.call_args[0][0]["image"]
        self.assertEqual((crop.shape, crop.dtype), ((1, 3, 224, 224), np.float32))

    def test_cpu_device_keeps_core_ml_on_the_cpu(self):
        _, MLModel = self.load("cpu")
        MLModel.assert_called_once_with("model.mlpackage", compute_units="cpu_only")


class TestSmoothing(unittest.TestCase):
    def analyze(self, e, yaw, ts, lid=0.01):
        e._predictCrop = MagicMock(return_value=np.array([yaw, 0.0]))
        mesh = TestEyeGeometry.landmarks(None, lid=lid).astype(np.float32)
        return e.analyze(np.zeros((480, 640, 3), np.uint8), mesh, ts)["yaw_rad"]

    def engine(self):
        e = engine()
        e._alpha = 0.25
        e._ema = e._emaEyes = e._emaTs = e._last = None
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
        self.assertAlmostEqual(iris[2], 0.3, places=5)   # eye aspect ratio of the open lids (float32 landmarks)

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

    def test_zero_mesh_is_no_face(self):
        # the tracking node publishes all zeros when it finds no face
        e = self.engine()
        e._predictCrop = MagicMock()
        mesh = np.zeros((GazeDinoEngine.MESH_POINTS, 3), np.float32)
        self.assertEqual(e.analyze(np.zeros((480, 640, 3), np.uint8), mesh, 0), {"status": "no_face"})
        e._predictCrop.assert_not_called()

    def test_mesh_without_irises_is_refused(self):
        # 468 points: a tracking channel without the irises
        e = self.engine()
        e._predictCrop = MagicMock()
        self.assertIsNone(e.analyze(np.zeros((480, 640, 3), np.uint8), np.ones((468, 3), np.float32), 0))
        e._predictCrop.assert_not_called()


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
