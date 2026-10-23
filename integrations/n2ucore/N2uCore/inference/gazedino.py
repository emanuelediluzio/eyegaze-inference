"""RGB gaze regression, independent of Tk and of Tobii pupillometry.

Model architecture matches emanuelediluzio/eye-gaze (GazeDINO).
The supplied checkpoint contains the complete fine-tuned backbone.
"""
import gc
import math
from pathlib import Path

import numpy as np


class GazeDinoEngine:
    """Single face; ImageNet RGB preprocessing and EMA angles in radians."""

    def __init__(self, checkpoint, device="auto", hub_dir="", alpha=0.25):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.checkpoint = Path(checkpoint).expanduser()
        self.device_name = device
        self.hub_dir = hub_dir
        self.alpha = alpha
        self.model = self.mesh = self.transform = self.device = None
        self._ema = self._bbox = None

    def load(self):
        import torch
        import torch.nn as nn
        import torchvision.transforms as T
        import mediapipe as mp

        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        device = self.device_name
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        hub_source = self.hub_dir
        if not hub_source:
            cached = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
            if (cached / "hubconf.py").is_file():
                hub_source = str(cached)
        # Fixed ViT-B/14 identity. Never download a redundant pretrained backbone.
        class GazeDINO(nn.Module):
            def __init__(inner):
                super().__init__()
                inner.backbone = torch.hub.load(
                    hub_source or "facebookresearch/dinov2", "dinov2_vitb14",
                    source="local" if hub_source else "github", pretrained=False)
                inner.head = nn.Sequential(
                    nn.LayerNorm(768), nn.Linear(768, 512), nn.GELU(), nn.Dropout(.3),
                    nn.Linear(512, 128), nn.GELU(), nn.Dropout(.15), nn.Linear(128, 2))

            def forward(inner, x):
                return inner.head(inner.backbone(x))

        model = GazeDINO()
        model.load_state_dict(checkpoint["model"], strict=True)
        self.model = model.eval().to(self.device)
        self.transform = T.Compose([
            T.ToPILImage(), T.Resize((224, 224)), T.ToTensor(),
            T.Normalize([.485, .456, .406], [.229, .224, .225])])
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=.5, min_tracking_confidence=.5)

    def reset(self):
        self._ema = self._bbox = None

    def predict_crop(self, rgb):
        import torch
        with torch.inference_mode():
            output = self.model(self.transform(rgb).unsqueeze(0).to(self.device))
        angles = output[0].detach().cpu().numpy().astype(float)
        if angles.shape != (2,) or not np.isfinite(angles).all():
            raise ValueError("GazeDINO returned invalid angles")
        return angles

    def analyze(self, rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("GazeDINO requires an RGB UINT8 image with depth 3")
        detected = self.mesh.process(np.ascontiguousarray(rgb))
        sample = {"schema": "gaze-direction.v1", "status": "no_face", "yaw_rad": None,
                  "pitch_rad": None, "yaw_deg": None, "pitch_deg": None,
                  "direction_xyz": None, "coordinate_space": "camera",
                  "bbox_xyxy_px": None, "confidence": None}
        if not detected.multi_face_landmarks:
            self.reset()
            return sample
        h, w = rgb.shape[:2]
        points = detected.multi_face_landmarks[0].landmark
        x1 = max(0, int(min(p.x for p in points)*w))
        y1 = max(0, int(min(p.y for p in points)*h))
        x2 = min(w, int(max(p.x for p in points)*w))
        y2 = min(h, int(max(p.y for p in points)*h))
        if x2 <= x1 or y2 <= y1:
            self.reset()
            return sample
        # Do not smooth across loss of tracking / a large face displacement.
        if self._bbox is not None:
            a, b, c, d = self._bbox
            intersection = max(0, min(c,x2)-max(a,x1))*max(0,min(d,y2)-max(b,y1))
            union = (c-a)*(d-b)+(x2-x1)*(y2-y1)-intersection
            if intersection / max(1, union) < .2:
                self.reset()
        angles = self.predict_crop(rgb[y1:y2, x1:x2])
        self._ema = angles if self._ema is None else self.alpha*angles+(1-self.alpha)*self._ema
        self._bbox = [x1, y1, x2, y2]
        yaw, pitch = map(float, self._ema)
        sample.update(status="ok", yaw_rad=yaw, pitch_rad=pitch,
                      yaw_deg=math.degrees(yaw), pitch_deg=math.degrees(pitch),
                      bbox_xyxy_px=self._bbox)
        sample["direction_xyz"] = [math.sin(yaw)*math.cos(pitch),
                                   -math.sin(pitch), -math.cos(yaw)*math.cos(pitch)]
        return sample

    def close(self):
        if self.mesh is not None:
            self.mesh.close()
        self.mesh = self.model = self.transform = None
        self.reset()
        gc.collect()
        import torch
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device is not None and self.device.type == "mps":
            torch.mps.empty_cache()
