from __future__ import annotations

import gc
import math
from collections import OrderedDict, deque
from pathlib import Path

import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from N2uCore import log

from N2uCore.inference.gazeartifact import loadModelState
from N2uCore.dyn.wrk.memprovider import releaseComputeCache, resolveComputeResources


class GazeDinoEngine(object):
    SUPPORTED_DEVICES = ("cuda", "xpu", "mps", "cpu")
    HALF_DEVICES = ("cuda", "xpu", "mps")  # half precision on a GPU: same angles (0.008 deg apart on a recording),
                                           # 27% faster on an M1; the CPU stays in fp32 (no gain there)

    MAX_FACES = 2               # faces tracked per frame; only the nearest one is analysed (with 4 tracked
                                # faces MediaPipe stops detecting, so a nearer newcomer would be missed)
    SWITCH_RATIO = 1.25         # another face must be this much bigger to replace the one being followed
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
        self._model = self._mesh = self._transform = self._device = self._dtype = None
        self._ema = self._emaEyes = self._bbox = self._emaTs = self._last = None
        self._raw = deque(maxlen=self.JITTER_WINDOW)

    def load(self):
        try:
            if not isinstance(self._alpha, (int, float)) or not 0 < self._alpha <= 1:
                log.err("alpha must be in (0, 1]")
                return False
            self._checkpoint = Path(self._checkpoint).expanduser()
            if not self._checkpoint.is_file():
                log.err("GazeDINO checkpoint not found: {}".format(self._checkpoint))
                return False
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
            self._transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=self.MAX_FACES,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            return True
        except Exception as exc:
            log.err("load: {}".format(exc))
            self.close()
            return False

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

    def _nearestFace(self, faces, width, height):
        # nearest = longest box side; the face already followed is kept unless another is SWITCH_RATIO times
        # bigger, so two people of similar size do not make the estimate jump between them
        candidates = []
        for face in faces:
            points = face.landmark
            left = min(point.x for point in points) * width
            top = min(point.y for point in points) * height
            right = max(point.x for point in points) * width
            bottom = max(point.y for point in points) * height
            box = [max(0, int(left)), max(0, int(top)), min(width, int(right)), min(height, int(bottom))]
            if box[2] > box[0] and box[3] > box[1]:
                edge = left < 0 or top < 0 or right > width or bottom > height
                candidates.append((max(box[2] - box[0], box[3] - box[1]), points, box, edge))
        if not candidates:
            return None, None, False
        best = max(candidates, key=lambda c: c[0])
        if self._bbox is not None:
            followed = max(candidates, key=lambda c: self._iou(c[2], self._bbox))
            if self._iou(followed[2], self._bbox) >= 0.2 and best[0] < self.SWITCH_RATIO * followed[0]:
                best = followed
        return best[1], best[2], best[3]

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

    def _eyeGeometry(self, points, width, height):
        # [x, y, span] in pixels: the eye corners do not move with the gaze, and the 3D corner distance
        # (FaceMesh z is on the x scale) shrinks little when the head turns
        corners = [points[i] for i in self.EYE_CORNERS]
        x = sum(p.x for p in corners) / len(corners) * width
        y = sum(p.y for p in corners) / len(corners) * height
        outer, other = corners[0], corners[-1]
        span = math.sqrt(
            ((other.x - outer.x) * width) ** 2 + ((other.y - outer.y) * height) ** 2 + ((other.z - outer.z) * width) ** 2
        )
        return np.asarray([x, y, span])

    def _eyeOpening(self, points, width, height):
        # eye aspect ratio (Soukupova and Cech 2016), mean of both eyes: lid distance over corner distance
        ratios = []
        for ids in self.EYE_LIDS:
            p = [np.array([points[i].x * width, points[i].y * height]) for i in ids]
            ratios.append((np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * np.linalg.norm(p[0] - p[3])))
        return sum(ratios) / len(ratios)

    def _irisPosition(self, points, width, height):
        # iris from the inner eye corner in eye widths, mean of both eyes: the eye width is a distance, so looking
        # sideways moves both irises the same way
        positions = []
        for iris, inner, outer in self.IRISES:
            i, a, b = (np.array([points[k].x * width, points[k].y * height]) for k in (iris, inner, outer))
            positions.append((i - a) / np.linalg.norm(b - a))
        return np.mean(positions, axis=0)

    def analyze(self, rgb, ts):
        try:
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                log.err("GazeDINO requires an RGB UINT8 image with depth 3")
                return None
            detected = self._mesh.process(np.ascontiguousarray(rgb))
            if not detected.multi_face_landmarks:
                self.reset()
                return {"status": "no_face"}
            height, width = rgb.shape[:2]
            points, box, atEdge = self._nearestFace(detected.multi_face_landmarks, width, height)
            if points is None:
                self.reset()
                return {"status": "no_face"}
            x1, y1, x2, y2 = box
            if self._bbox is not None and self._iou(box, self._bbox) < 0.2:
                self.reset()            # another face, or a jump: no smoothing across it
            opening = self._eyeOpening(points, width, height)
            if self._ema is not None and opening < self.BLINK_EAR:
                return self._last
            angles = self._predictCrop(rgb[y1:y2, x1:x2])
            if angles is None:
                self.reset()
                return None
            # the longer box side shrinks less than the width when the head turns
            quality = self._quality(angles, max(x2 - x1, y2 - y1) / width, atEdge)
            # eye position smoothed like the angles, so that both lag the same when the head moves
            eyes = self._eyeGeometry(points, width, height)
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
                "eyes_px": [[points[i].x * width, points[i].y * height] for i in (468, 473)],
                "eye_center_px": [float(self._emaEyes[0]), float(self._emaEyes[1])],
                "eye_span_px": float(self._emaEyes[2]),
                # [x, y, lid opening] of the iris in the eye, not smoothed: with the angles they keep the screen
                # point where the head moved (ScreenCalibration)
                "iris": [*map(float, self._irisPosition(points, width, height)), float(opening)],
                "quality": quality,
            }
            return self._last
        except Exception as exc:
            log.err("analyze: {}".format(exc))
            return None

    def close(self):
        """Release both tracker and model even if the external tracker fails to close."""
        try:
            if self._mesh is not None:
                self._mesh.close()
        except Exception as exc:
            log.err("Cannot close face tracker: {}".format(exc))
        finally:
            self._mesh = self._model = self._transform = None
            self.reset()
            gc.collect()
            releaseComputeCache(self._device)
