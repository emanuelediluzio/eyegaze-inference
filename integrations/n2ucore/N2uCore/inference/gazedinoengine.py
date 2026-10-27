import gc
import math
from collections import OrderedDict
from pathlib import Path

import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from N2uCore.inference.gazeartifact import loadModelState


class GazeDinoEngine(object):
    def __init__(self, checkpoint, deviceName="auto", hubDir="", alpha=0.25):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self._checkpoint = Path(checkpoint).expanduser()
        self._deviceName = deviceName
        self._hubDir = hubDir
        self._alpha = alpha
        self._model = self._mesh = self._transform = self._device = None
        self._ema = self._bbox = None

    def load(self):
        if not self._checkpoint.is_file():
            raise FileNotFoundError(self._checkpoint)
        deviceName = self._deviceName
        if deviceName == "auto":
            deviceName = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        self._device = torch.device(deviceName)
        hubSource = (
            Path(self._hubDir)
            if self._hubDir
            else (Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main")
        )
        if not (hubSource / "hubconf.py").is_file():
            raise FileNotFoundError("Install DINOv2 in the local Torch Hub cache before starting GazeDINO")
        model = self._buildModel(hubSource)
        modelState = loadModelState(self._checkpoint)
        model.load_state_dict(modelState, strict=True)
        del modelState
        self._model = model.eval().to(self._device)
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

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
        self._ema = self._bbox = None

    def _predictCrop(self, rgb):
        with torch.inference_mode():
            output = self._model(self._transform(rgb).unsqueeze(0).to(self._device))
        angles = output[0].detach().cpu().numpy().astype(float)
        if angles.shape != (2,) or not np.isfinite(angles).all():
            raise ValueError("GazeDINO returned invalid angles")
        return angles

    def _headRoll(self, points, width, height):
        if len(points) <= 263:
            return None
        left, right = points[33], points[263]
        dx = (right.x - left.x) * width
        dy = (right.y - left.y) * height
        if not np.isfinite([dx, dy]).all() or math.hypot(dx, dy) < 1e-6:
            return None
        return math.atan2(dy, dx)

    def analyze(self, rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("GazeDINO requires an RGB UINT8 image with depth 3")
        detected = self._mesh.process(np.ascontiguousarray(rgb))
        sample = {
            "schema": "gaze-direction.v1",
            "status": "no_face",
            "yaw_rad": None,
            "pitch_rad": None,
            "roll_rad": None,
            "yaw_deg": None,
            "pitch_deg": None,
            "roll_deg": None,
            "roll_source": "head_landmarks",
            "direction_xyz": None,
            "coordinate_space": "camera",
            "bbox_xyxy_px": None,
            "confidence": None,
            "image_size_px": [rgb.shape[1], rgb.shape[0]],
            "eyes_px": [],
        }
        if not detected.multi_face_landmarks:
            self.reset()
            return sample
        height, width = rgb.shape[:2]
        points = detected.multi_face_landmarks[0].landmark
        if len(points) >= 478:
            sample["eyes_px"] = [[points[i].x * width, points[i].y * height] for i in (468, 473)]
        x1 = max(0, int(min(point.x for point in points) * width))
        y1 = max(0, int(min(point.y for point in points) * height))
        x2 = min(width, int(max(point.x for point in points) * width))
        y2 = min(height, int(max(point.y for point in points) * height))
        if x2 <= x1 or y2 <= y1:
            self.reset()
            return sample
        if self._bbox is not None:
            a, b, c, d = self._bbox
            intersection = max(0, min(c, x2) - max(a, x1)) * max(0, min(d, y2) - max(b, y1))
            union = (c - a) * (d - b) + (x2 - x1) * (y2 - y1) - intersection
            if intersection / max(1, union) < 0.2:
                self.reset()
        angles = self._predictCrop(rgb[y1:y2, x1:x2])
        self._ema = angles if self._ema is None else self._alpha * angles + (1 - self._alpha) * self._ema
        self._bbox = [x1, y1, x2, y2]
        yaw, pitch = map(float, self._ema)
        roll = self._headRoll(points, width, height)
        sample.update(
            status="ok",
            yaw_rad=yaw,
            pitch_rad=pitch,
            roll_rad=roll,
            yaw_deg=math.degrees(yaw),
            pitch_deg=math.degrees(pitch),
            roll_deg=math.degrees(roll) if roll is not None else None,
            bbox_xyxy_px=self._bbox,
        )
        sample["direction_xyz"] = [
            math.sin(yaw) * math.cos(pitch),
            -math.sin(pitch),
            -math.cos(yaw) * math.cos(pitch),
        ]
        return sample

    def close(self):
        if self._mesh is not None:
            self._mesh.close()
        self._mesh = self._model = self._transform = None
        self.reset()
        gc.collect()
        if self._device is not None and self._device.type == "cuda":
            torch.cuda.empty_cache()
        elif self._device is not None and self._device.type == "mps":
            torch.mps.empty_cache()
