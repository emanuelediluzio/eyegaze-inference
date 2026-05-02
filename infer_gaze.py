"""
Gaze estimation core — model, FaceMesh analyzer, EMA, drawing helpers.

Used by gui.py as the single entry point.
"""
from __future__ import annotations
import math
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Image transform — ImageNet normalisation (matches GazeDINO training)
# ---------------------------------------------------------------------------

_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: torch.device):
    """Load GazeDINO checkpoint."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    from models.gaze_model import GazeDINO
    model = GazeDINO().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[infer] Checkpoint  epoch={ckpt.get('epoch','?')}  "
          f"val_err={ckpt.get('val_err', ckpt.get('val_err_deg', float('nan'))):.2f}\u00b0")
    return model


def predict(model, face_bgr: np.ndarray, device: torch.device) -> tuple[float, float]:
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    inp = _TRANSFORM(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(inp)[0].cpu().numpy()
        yaw, pitch = float(out[0]), float(out[1])

    return yaw, pitch


# ---------------------------------------------------------------------------
# Constants — landmarks
# ---------------------------------------------------------------------------

_L_IRIS   = [468, 469, 470, 471, 472]
_R_IRIS   = [473, 474, 475, 476, 477]
_L_INNER, _L_OUTER = 133, 33
_R_INNER, _R_OUTER = 362, 263
_L_CHEEK, _R_CHEEK = 234, 454

# Distance
_REAL_FACE_W_CM  = 14.0
_FOCAL_LENGTH_PX = 554.0
_MIN_DIST_CM     = 40.0
_MAX_DIST_CM     = 90.0

# EAR (Eye Aspect Ratio) — blink detection
_L_EYE_V = [(159, 145), (158, 153), (160, 144)]
_L_EYE_H = (33, 133)
_R_EYE_V = [(386, 374), (385, 380), (387, 373)]
_R_EYE_H = (263, 362)
_EAR_BLINK_THRESH = 0.21

# Head pose — 3D model points (generic face, mm)
_HEAD_3D = np.array([
    (0.0,    0.0,     0.0),      # nose tip        (1)
    (0.0, -330.0,   -65.0),      # chin             (199)
    (-225.0, 170.0, -135.0),     # left eye corner  (33)
    (225.0,  170.0, -135.0),     # right eye corner (263)
    (-150.0, -150.0, -125.0),    # left mouth       (61)
    (150.0,  -150.0, -125.0),    # right mouth      (291)
], dtype=np.float64)
_HEAD_IDX = [1, 199, 33, 263, 61, 291]


# ---------------------------------------------------------------------------
# Blink tracker
# ---------------------------------------------------------------------------

class BlinkTracker:
    def __init__(self, window_sec: float = 60.0):
        self._window = window_sec
        self._timestamps: list[float] = []
        self._was_closed = False

    def update(self, ear: float) -> int:
        """Feed EAR value, returns current blinks/minute."""
        now = time.monotonic()
        # detect blink: EAR drops below threshold then rises
        if ear < _EAR_BLINK_THRESH:
            self._was_closed = True
        elif self._was_closed:
            self._was_closed = False
            self._timestamps.append(now)

        # prune old
        cutoff = now - self._window
        self._timestamps = [t for t in self._timestamps if t > cutoff]
        # scale to per-minute
        elapsed = min(self._window, now - self._timestamps[0]) if self._timestamps else self._window
        if elapsed < 1.0:
            return 0
        return int(len(self._timestamps) * 60.0 / elapsed)

    def reset(self):
        self._timestamps.clear()
        self._was_closed = False


# ---------------------------------------------------------------------------
# FaceMesh analyzer
# ---------------------------------------------------------------------------

class FaceMeshAnalyzer:
    def __init__(self, max_faces: int = 5):
        import mediapipe as mp
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=max_faces,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._fd = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        # store tesselation edges for mesh drawing
        self._tess = list(mp.solutions.face_mesh.FACEMESH_TESSELATION)

        self._max_faces = max_faces
        self._dist_ema: dict[int, float] = {}
        self._l_iris_ema: dict[int, float] = {}
        self._r_iris_ema: dict[int, float] = {}
        self._da = 0.1
        self._pa = 0.2
        print(f"[infer] FaceMesh: ON (max_faces={max_faces})")

    @property
    def tesselation(self):
        return self._tess

    def process(self, frame_rgb: np.ndarray) -> list[dict]:
        """
        Returns list of dicts per face:
            face_bbox, left_eye, right_eye, distance, pupil,
            landmarks (Nx2 float32), ear (float), head_pose (yaw,pitch,roll deg),
            head_rvec, head_tvec
        """
        h, w = frame_rgb.shape[:2]
        res = self._mesh.process(frame_rgb)

        if not res.multi_face_landmarks:
            return []

        # camera matrix (approximate)
        focal = w
        cam_mtx = np.array([
            [focal, 0,     w / 2],
            [0,     focal, h / 2],
            [0,     0,     1    ],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        faces = []
        for fi, face_lm in enumerate(res.multi_face_landmarks):
            lm  = face_lm.landmark
            pts = np.array([[l.x * w, l.y * h] for l in lm], dtype=np.float32)

            out: dict = {
                "face_bbox": None,
                "left_eye":  None,
                "right_eye": None,
                "distance":  {"cm": None, "status": "no_face",
                              "color": (160, 160, 160), "label": ""},
                "pupil":     None,
                "landmarks": pts,
                "ear":       None,
                "head_pose": None,
                "head_rvec": None,
                "head_tvec": None,
            }

            # -- face bbox
            xs, ys = pts[:, 0], pts[:, 1]
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            out["face_bbox"] = (x1, y1, x2 - x1, y2 - y1)

            # -- iris
            if len(lm) >= 478:
                out["left_eye"]  = (int(pts[468, 0]), int(pts[468, 1]))
                out["right_eye"] = (int(pts[473, 0]), int(pts[473, 1]))

                # pupillometry
                def _iris_d(idxs):
                    p = pts[idxs]
                    return float((np.linalg.norm(p[1]-p[3]) + np.linalg.norm(p[2]-p[4])) / 2)

                def _eye_w(i, o):
                    return float(np.linalg.norm(pts[i] - pts[o]))

                ld = _iris_d(_L_IRIS);  lw = _eye_w(_L_INNER, _L_OUTER)
                rd = _iris_d(_R_IRIS);  rw = _eye_w(_R_INNER, _R_OUTER)

                if lw > 1 and rw > 1:
                    lr = ld / lw;  rr = rd / rw
                    prev_l = self._l_iris_ema.get(fi)
                    prev_r = self._r_iris_ema.get(fi)
                    self._l_iris_ema[fi] = (lr if prev_l is None
                                            else self._pa * lr + (1-self._pa) * prev_l)
                    self._r_iris_ema[fi] = (rr if prev_r is None
                                            else self._pa * rr + (1-self._pa) * prev_r)
                    out["pupil"] = {
                        "left_rel":  self._l_iris_ema[fi],
                        "right_rel": self._r_iris_ema[fi],
                        "mean_rel":  (self._l_iris_ema[fi] + self._r_iris_ema[fi]) / 2,
                    }

            # -- EAR (Eye Aspect Ratio)
            def _ear(v_pairs, h_pair):
                vsum = sum(np.linalg.norm(pts[a] - pts[b]) for a, b in v_pairs)
                hw = np.linalg.norm(pts[h_pair[0]] - pts[h_pair[1]])
                return float(vsum / (3.0 * hw)) if hw > 1 else 1.0

            l_ear = _ear(_L_EYE_V, _L_EYE_H)
            r_ear = _ear(_R_EYE_V, _R_EYE_H)
            out["ear"] = (l_ear + r_ear) / 2.0

            # -- distance
            face_w_px = float(np.linalg.norm(pts[_L_CHEEK] - pts[_R_CHEEK]))
            if face_w_px > 1:
                raw = (_REAL_FACE_W_CM * _FOCAL_LENGTH_PX) / face_w_px
                prev_d = self._dist_ema.get(fi)
                self._dist_ema[fi] = (raw if prev_d is None
                                      else self._da * raw + (1-self._da) * prev_d)
                d = self._dist_ema[fi]
                if d < _MIN_DIST_CM:
                    status, color, label = "too_close", (0, 0, 220), "Too close!"
                elif d > _MAX_DIST_CM:
                    status, color, label = "too_far",   (0, 165, 255), "Too far!"
                else:
                    status, color, label = "ok",        (0, 200, 60),  ""
                out["distance"] = {"cm": d, "status": status, "color": color, "label": label}

            # -- head pose (solvePnP)
            pts_2d = pts[_HEAD_IDX].astype(np.float64)
            ok, rvec, tvec = cv2.solvePnP(
                _HEAD_3D, pts_2d, cam_mtx, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                rmat, _ = cv2.Rodrigues(rvec)
                # decompose to Euler (yaw, pitch, roll)
                sy = math.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
                if sy > 1e-6:
                    hp_x = math.atan2(rmat[2, 1], rmat[2, 2])
                    hp_y = math.atan2(-rmat[2, 0], sy)
                    hp_z = math.atan2(rmat[1, 0], rmat[0, 0])
                else:
                    hp_x = math.atan2(-rmat[1, 2], rmat[1, 1])
                    hp_y = math.atan2(-rmat[2, 0], sy)
                    hp_z = 0.0
                out["head_pose"] = (
                    math.degrees(hp_y),   # yaw
                    math.degrees(hp_x),   # pitch
                    math.degrees(hp_z),   # roll
                )
                out["head_rvec"] = rvec
                out["head_tvec"] = tvec

            faces.append(out)

        return faces

    def reset(self):
        self._dist_ema.clear()
        self._l_iris_ema.clear()
        self._r_iris_ema.clear()

    def detect_face_only(self, frame_rgb: np.ndarray):
        h, w = frame_rgb.shape[:2]
        res = self._fd.process(frame_rgb)
        if not res.detections:
            return None
        best = max(res.detections, key=lambda d: d.score[0])
        bb = best.location_data.relative_bounding_box
        return (max(0, int(bb.xmin*w)), max(0, int(bb.ymin*h)),
                int(bb.width*w), int(bb.height*h))


# ---------------------------------------------------------------------------
# EMA smoother
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._val = None

    def update(self, x: np.ndarray) -> np.ndarray:
        self._val = x.copy() if self._val is None \
                    else self.alpha * x + (1 - self.alpha) * self._val
        return self._val.copy()

    def reset(self):
        self._val = None


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_eye_arrows(frame, left_eye, right_eye, yaw, pitch, length=120,
                     alpha: float = 0.45):
    overlay = frame.copy()
    dx = int(math.sin(yaw)    * length)
    dy = int(-math.sin(pitch) * length)
    for (ex, ey) in [p for p in (left_eye, right_eye) if p]:
        cv2.circle(overlay, (ex, ey), 4, (200, 220, 200), -1)
        end = (ex + dx, ey + dy)
        cv2.arrowedLine(overlay, (ex, ey), end, (180, 220, 180), 2, tipLength=0.25)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_gaze_arrow(frame, cx, cy, yaw, pitch, length=150,
                     alpha: float = 0.45):
    overlay = frame.copy()
    dx = int(math.sin(yaw)    * length)
    dy = int(-math.sin(pitch) * length)
    end = (cx + dx, cy + dy)
    cv2.arrowedLine(overlay, (cx, cy), end, (180, 220, 180), 2, tipLength=0.22)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_face_box(frame, x1, y1, x2, y2, alpha: float = 0.4):
    overlay = frame.copy()
    bw, bh = x2 - x1, y2 - y1
    corner_len = min(18, bw // 5, bh // 5)
    c = (200, 200, 200)
    for (cx, cy, dx, dy) in [
        (x1, y1, 1, 1), (x2, y1, -1, 1),
        (x1, y2, 1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(overlay, (cx, cy), (cx + dx * corner_len, cy), c, 1)
        cv2.line(overlay, (cx, cy), (cx, cy + dy * corner_len), c, 1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_iris_dots(frame, left_eye, right_eye, alpha: float = 0.5):
    overlay = frame.copy()
    for pt in [p for p in (left_eye, right_eye) if p]:
        cv2.circle(overlay, pt, 3, (220, 220, 220), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_face_mesh(frame, landmarks, edges, alpha: float = 0.15):
    overlay = frame.copy()
    pts = landmarks.astype(np.int32)
    for (a, b) in edges:
        if a < len(pts) and b < len(pts):
            cv2.line(overlay, tuple(pts[a]), tuple(pts[b]), (180, 180, 180), 1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_head_axes(frame, rvec, tvec, cam_mtx, length=80.0, alpha: float = 0.5):
    """Draw RGB XYZ axes at the nose tip."""
    axes_3d = np.array([
        [0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))
    pts_2d, _ = cv2.projectPoints(axes_3d, rvec, tvec, cam_mtx, dist_coeffs)
    pts_2d = pts_2d.reshape(-1, 2).astype(int)
    origin = tuple(pts_2d[0])

    overlay = frame.copy()
    cv2.arrowedLine(overlay, origin, tuple(pts_2d[1]), (80, 80, 220), 2, tipLength=0.2)   # X red
    cv2.arrowedLine(overlay, origin, tuple(pts_2d[2]), (80, 200, 80), 2, tipLength=0.2)    # Y green
    cv2.arrowedLine(overlay, origin, tuple(pts_2d[3]), (220, 140, 80), 2, tipLength=0.2)   # Z blue
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
