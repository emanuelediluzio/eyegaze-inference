from __future__ import annotations

import gc
import math
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from N2uCore import log
from N2uCore.inference.gazeartifact import loadModelState
from N2uCore.dyn.wrk.memprovider import releaseComputeCache, resolveComputeResources


class CoreMLGaze(object):
    """GazeDINO as a Core ML model, called like the PyTorch one: normalized crop tensor in, (1, 2) angles out."""

    def __init__(self, model):
        self._model = model

    def __call__(self, crop):
        return torch.from_numpy(self._model.predict({"image": crop.numpy()})["gaze"].astype(np.float32))


class GazeDinoEngine(object):
    SUPPORTED_DEVICES = ("cuda", "xpu", "mps", "cpu")
    HALF_DEVICES = ("cuda", "xpu", "mps")  # half precision on a GPU: same angles (0.008 deg apart on a recording),
                                           # 27% faster on an M1; the CPU stays in fp32 (no gain there)

    MESH_POINTS = 478           # FaceLandmarker points from the tracking node's Landmarks channel, irises included
    FACE_RATIO_GOOD = (0.18, 0.55)  # longer face box side / image width at full quality (sitting distance)
    FACE_RATIO_LIMIT = (0.08, 0.85) # quality reaches 0 at these ratios (too far / too close)
    JITTER_GOOD_DEG = 12.0      # median change of the frame-to-frame step at full quality (noise, not motion);
                                # first live measure (19 frames, 2.6 FPS): ~16 deg; to be tuned on live data
    JITTER_LIMIT_DEG = 30.0     # quality reaches 0 at this median change
    JITTER_WINDOW = 15          # raw estimates kept to measure the jitter (about 1 s at 15 FPS)
    JITTER_MIN_SAMPLES = 5      # estimates needed before the jitter counts
    EYE_CORNERS = (33, 133, 362, 263)  # outer and inner corners of both eyes (FaceMesh indices)
    EYE_LIDS = ((33, 160, 158, 133, 153, 144), (362, 385, 387, 263, 373, 380))  # corner, 2 upper, corner, 2 lower
    IRISES = ((468, 133, 33), (473, 362, 263))  # iris centre, inner and outer corner of each eye
    BLINK_EAR = 0.15            # eye aspect ratio under which the eyes are closed (open 0.30-0.38, 0.28 looking at the
                                # bottom of the screen): closed lids read as looking down, so the last estimate is kept
                                # (on a recording, 0.3 s after a blink 311 -> 112 px)
    ALPHA_FPS = 15              # alpha is the weight of a new estimate at this rate: the smoothing lasts the same
                                # time at any rate (at 5 FPS a per-estimate 0.25 made the calibration 30% worse)

    def __init__(self, checkpoint, deviceName="auto", hubDir="", alpha=0.25):
        self._checkpoint = checkpoint
        self._deviceName = deviceName
        self._hubDir = hubDir
        self._alpha = alpha
        self._model = self._transform = self._device = self._dtype = None
        self._ema = self._emaEyes = self._bbox = self._emaTs = self._last = None
        self._raw = deque(maxlen=self.JITTER_WINDOW)

    def load(self):
        try:
            if not isinstance(self._alpha, (int, float)) or not 0 < self._alpha <= 1:
                log.err("alpha must be in (0, 1]")
                return False
            self._checkpoint = Path(self._checkpoint).expanduser()
            if not self._checkpoint.exists():
                log.err("GazeDINO checkpoint not found: {}".format(self._checkpoint))
                return False
            if not (self._loadCoreML() if self._checkpoint.suffix == ".mlpackage" else self._loadTorch()):
                return False
            self._transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            return True
        except Exception as exc:
            log.err("load: {}".format(exc))
            self.close()
            return False

    def _loadTorch(self):
        # PyTorch weights (gazeartifact): half precision on a GPU, fp32 on the CPU
        deviceName = self._deviceName
        if deviceName == "auto":
            resources = resolveComputeResources(supportedDevices=self.SUPPORTED_DEVICES)
            if resources is None:
                return False
            deviceName, _ = resources
        self._device = torch.device(deviceName)
        hubSource = (
            Path(self._hubDir)
            if self._hubDir
            else (Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main")
        )
        if not (hubSource / "hubconf.py").is_file():
            log.err("Install DINOv2 in the local Torch Hub cache before starting GazeDINO")
            return False
        model = self._buildModel(hubSource)
        modelState = loadModelState(self._checkpoint)
        if modelState is None:
            return False
        model.load_state_dict(modelState, strict=True)
        del modelState
        self._dtype = torch.float16 if self._device.type in self.HALF_DEVICES else torch.float32
        self._model = model.eval().to(self._device, self._dtype)
        return True

    def _loadCoreML(self):
        # Core ML export on macOS (eyegaze-inference experiments/model-size/export_coreml.py, int8 weights):
        # same angles as PyTorch on MPIIFaceGaze (6.37 vs 6.35 deg) and on the recordings, 87 MB instead of 324,
        # ~10 ms a crop on an M1 Neural Engine instead of ~105 ms on MPS; Core ML picks its compute units itself
        try:
            import coremltools as ct
        except ImportError:
            log.err("coremltools is needed for the Core ML GazeDINO")
            return False
        units = ct.ComputeUnit.CPU_ONLY if self._deviceName == "cpu" else ct.ComputeUnit.ALL
        self._device, self._dtype = torch.device("cpu"), torch.float32
        self._model = CoreMLGaze(ct.models.MLModel(str(self._checkpoint), compute_units=units))
        return True

    def _buildModel(self, hubSource):
        backbone = torch.hub.load(str(hubSource), "dinov2_vitb14", source="local", pretrained=False)
        head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 2),
        )
        return nn.Sequential(OrderedDict([("backbone", backbone), ("head", head)]))

    def reset(self):
        self._ema = self._emaEyes = self._bbox = self._emaTs = None
        self._raw.clear()

    def _predictCrop(self, rgb):
        try:
            with torch.inference_mode():
                output = self._model(self._transform(rgb).unsqueeze(0).to(self._device, self._dtype))
            angles = output[0].detach().cpu().numpy().astype(float)
            if angles.shape != (2,) or not np.isfinite(angles).all():
                log.err("GazeDINO returned invalid angles")
                return None
            return angles
        except Exception as exc:
            log.err("_predictCrop: {}".format(exc))
            return None

    @staticmethod
    def _ramp(value, good, limit):
        # 1 inside good, falling linearly to 0 at limit
        if value < good[0]:
            return max(0.0, (value - limit[0]) / (good[0] - limit[0]))
        if value > good[1]:
            return max(0.0, (limit[1] - value) / (limit[1] - good[1]))
        return 1.0

    @staticmethod
    def _iou(a, b):
        inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / max(1, union)

    def _faceBox(self, mesh, width, height):
        # single face from the shared mesh (tracking is num_faces=1): bounding box in pixels, and whether it
        # touches the image edge (used by the quality hint)
        xs = mesh[:, 0] * width
        ys = mesh[:, 1] * height
        left, right = float(np.min(xs)), float(np.max(xs))
        top, bottom = float(np.min(ys)), float(np.max(ys))
        box = [max(0, int(left)), max(0, int(top)), min(width, int(right)), min(height, int(bottom))]
        if box[2] <= box[0] or box[3] <= box[1]:
            return None, False
        edge = left < 0 or top < 0 or right > width or bottom > height
        return box, edge

    def _quality(self, angles, faceRatio, atEdge=False):
        # from face size and stability of the raw angles: the model gives no reliable uncertainty
        self._raw.append(angles)
        jitter = None
        if len(self._raw) >= self.JITTER_MIN_SAMPLES:
            # second differences: a steady head turn has equal steps (~0), noise does not
            steps = np.diff(np.asarray(self._raw), n=2, axis=0)
            jitter = math.degrees(float(np.median(np.hypot(steps[:, 0], steps[:, 1]))))
        size = self._ramp(faceRatio, self.FACE_RATIO_GOOD, self.FACE_RATIO_LIMIT)
        stable = 1.0 if jitter is None else self._ramp(jitter, (0, self.JITTER_GOOD_DEG), (0, self.JITTER_LIMIT_DEG))
        if atEdge:
            hint = "center_face"
        elif faceRatio < self.FACE_RATIO_GOOD[0]:
            hint = "move_closer"
        elif faceRatio > self.FACE_RATIO_GOOD[1]:
            hint = "move_back"
        elif stable < 0.5:
            hint = "unstable"
        else:
            hint = "ok"
        return {
            "score": None if jitter is None else int(round(100 * size * stable)),
            "hint": hint,
            "face_size_ratio": faceRatio,
            "jitter_deg": jitter,
        }

    def _eyeGeometry(self, mesh, width, height):
        # [x, y, span] in pixels: the eye corners do not move with the gaze, and the 3D corner distance
        # (FaceMesh z is on the x scale) shrinks little when the head turns
        idx = list(self.EYE_CORNERS)
        x = float(np.mean(mesh[idx, 0]) * width)
        y = float(np.mean(mesh[idx, 1]) * height)
        o, e = idx[0], idx[-1]
        span = math.sqrt(
            ((mesh[e, 0] - mesh[o, 0]) * width) ** 2 + ((mesh[e, 1] - mesh[o, 1]) * height) ** 2 + ((mesh[e, 2] - mesh[o, 2]) * width) ** 2
        )
        return np.asarray([x, y, span])

    def _eyeOpening(self, mesh, width, height):
        # eye aspect ratio (Soukupova and Cech 2016), mean of both eyes: lid distance over corner distance
        ratios = []
        for ids in self.EYE_LIDS:
            p = [np.array([mesh[i, 0] * width, mesh[i, 1] * height]) for i in ids]
            ratios.append((np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * np.linalg.norm(p[0] - p[3])))
        return sum(ratios) / len(ratios)

    def _irisPosition(self, mesh, width, height):
        # iris from the inner eye corner in eye widths, mean of both eyes: the eye width is a distance, so looking
        # sideways moves both irises the same way
        positions = []
        for iris, inner, outer in self.IRISES:
            i, a, b = (np.array([mesh[k, 0] * width, mesh[k, 1] * height]) for k in (iris, inner, outer))
            positions.append((i - a) / np.linalg.norm(b - a))
        return np.mean(positions, axis=0)

    def analyze(self, rgb, mesh, ts):
        try:
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                log.err("GazeDINO requires an RGB UINT8 image with depth 3")
                return None
            if mesh is None or len(mesh) != self.MESH_POINTS:
                log.err("GazeDINO requires the {} FaceLandmarker points".format(self.MESH_POINTS))
                return None
            if not mesh.any():          # no face: the tracking node publishes its FAKE_MESH, all zeros
                self.reset()
                return {"status": "no_face"}
            height, width = rgb.shape[:2]
            box, atEdge = self._faceBox(mesh, width, height)
            if box is None:
                self.reset()
                return {"status": "no_face"}
            x1, y1, x2, y2 = box
            if self._bbox is not None and self._iou(box, self._bbox) < 0.2:
                self.reset()            # another face, or a jump: no smoothing across it
            opening = self._eyeOpening(mesh, width, height)
            if self._ema is not None and opening < self.BLINK_EAR:
                return self._last
            angles = self._predictCrop(rgb[y1:y2, x1:x2])
            if angles is None:
                self.reset()
                return None
            # the longer box side shrinks less than the width when the head turns
            quality = self._quality(angles, max(x2 - x1, y2 - y1) / width, atEdge)
            # eye position smoothed like the angles, so that both lag the same when the head moves
            eyes = self._eyeGeometry(mesh, width, height)
            if self._ema is None:
                self._ema, self._emaEyes = angles, eyes
            else:
                alpha = 1 - (1 - self._alpha) ** ((ts - self._emaTs) / 1e6 * self.ALPHA_FPS)
                self._ema = alpha * angles + (1 - alpha) * self._ema
                self._emaEyes = alpha * eyes + (1 - alpha) * self._emaEyes
            self._emaTs = ts
            self._bbox = box
            self._last = {
                "status": "ok",
                "yaw_rad": float(self._ema[0]),
                "pitch_rad": float(self._ema[1]),
                "bbox_xyxy_px": box,
                "eyes_px": [[float(mesh[i, 0] * width), float(mesh[i, 1] * height)] for i in (468, 473)],
                "eye_center_px": [float(self._emaEyes[0]), float(self._emaEyes[1])],
                "eye_span_px": float(self._emaEyes[2]),
                # [x, y, lid opening] of the iris in the eye, not smoothed: with the angles they keep the screen
                # point where the head moved (ScreenCalibration)
                "iris": [*map(float, self._irisPosition(mesh, width, height)), float(opening)],
                "quality": quality,
            }
            return self._last
        except Exception as exc:
            log.err("analyze: {}".format(exc))
            return None

    def close(self):
        """Release the model resources."""
        self._model = self._transform = None
        self.reset()
        gc.collect()
        releaseComputeCache(self._device)
