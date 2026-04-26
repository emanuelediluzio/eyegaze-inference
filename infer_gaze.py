"""
Real-time gaze estimation — full pipeline.

Features
--------
  - GazeDINO (PyTorch) gaze estimation
  - Single MediaPipe FaceMesh pass: iris positions, distance, pupillometry
  - 9-point calibration: maps gaze angles -> screen coordinates
  - Distance alarm: too close / too far from screen
  - Pupillometry: relative iris diameter (proxy for dilation)
  - EMA smoothing on gaze output

Usage
-----
    python infer_gaze.py
    python infer_gaze.py --model checkpoints/best.pt
    python infer_gaze.py --calibrate
    python infer_gaze.py --calib_file calibration.pkl

Controls
--------
    Q / ESC  — quit
    C        — run calibration
    R        — reset EMA + monitors
    D        — toggle distance overlay
    P        — toggle pupillometry overlay
"""
from __future__ import annotations
import argparse
import math
import time
from pathlib import Path

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
          f"val_err={ckpt.get('val_err_deg', float('nan')):.2f}\u00b0")
    return model


def predict(model, face_bgr: np.ndarray, device: torch.device) -> tuple[float, float]:
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    inp = _TRANSFORM(rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(inp)[0].cpu().numpy()
        yaw, pitch = float(out[0]), float(out[1])

    return yaw, pitch


# ---------------------------------------------------------------------------
# Unified FaceMesh analyzer — single pass per frame
# ---------------------------------------------------------------------------
#
# MediaPipe landmark indices used:
#   iris centres  : 468 (left), 473 (right)
#   iris ring L   : 468-472  (centre, right, bottom, left, top)
#   iris ring R   : 473-477
#   eye corners L : inner 133, outer 33
#   eye corners R : inner 362, outer 263
#   cheekbones    : 234 (left), 454 (right)  — for distance

_L_IRIS   = [468, 469, 470, 471, 472]
_R_IRIS   = [473, 474, 475, 476, 477]
_L_INNER, _L_OUTER = 133, 33
_R_INNER, _R_OUTER = 362, 263
_L_CHEEK, _R_CHEEK = 234, 454

# Distance constants
_REAL_FACE_W_CM  = 14.0   # average bizygomatic width
_FOCAL_LENGTH_PX = 554.0  # for ~640px wide, ~60 FOV camera
_MIN_DIST_CM     = 40.0
_MAX_DIST_CM     = 90.0


class FaceMeshAnalyzer:
    """
    Wraps MediaPipe FaceMesh and extracts in one pass:
        - face bounding box        (for face crop -> gaze model)
        - iris centre positions    (for gaze arrow origin)
        - iris diameters / eye widths  (for pupillometry)
        - face width in px         (for distance estimate)
    """

    def __init__(self, max_faces: int = 5):
        import mediapipe as mp
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,   # required for iris (468-477)
            max_num_faces=max_faces,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        # face detection fallback
        self._fd = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self._max_faces = max_faces
        # per-face EMA state (keyed by face index)
        self._dist_ema: dict[int, float] = {}
        self._l_iris_ema: dict[int, float] = {}
        self._r_iris_ema: dict[int, float] = {}
        _EMA_ALPHA_DIST  = 0.1
        _EMA_ALPHA_PUPIL = 0.2
        self._da = _EMA_ALPHA_DIST
        self._pa = _EMA_ALPHA_PUPIL
        print(f"[infer] FaceMesh: ON (max_faces={max_faces}, iris + distance + pupillometry)")

    # ------------------------------------------------------------------ #
    def process(self, frame_rgb: np.ndarray) -> list[dict]:
        """
        Returns a list of dicts (one per detected face), each with keys:
            face_bbox  : (x,y,w,h) | None
            left_eye   : (lx,ly)   | None
            right_eye  : (rx,ry)   | None
            distance   : {"cm": float|None, "status": str, "color": tuple, "label": str}
            pupil      : {"left_rel": float, "right_rel": float, "mean_rel": float} | None
        Returns an empty list if no faces found.
        """
        h, w = frame_rgb.shape[:2]
        res = self._mesh.process(frame_rgb)

        if not res.multi_face_landmarks:
            return []

        faces = []
        for fi, face_lm in enumerate(res.multi_face_landmarks):
            lm  = face_lm.landmark
            pts = np.array([[l.x * w, l.y * h] for l in lm], dtype=np.float32)

            out = {
                "face_bbox": None,
                "left_eye":  None,
                "right_eye": None,
                "distance":  {"cm": None, "status": "no_face",
                              "color": (160, 160, 160), "label": ""},
                "pupil":     None,
            }

            # -- face bbox from mesh extent
            xs, ys = pts[:, 0], pts[:, 1]
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            out["face_bbox"] = (x1, y1, x2 - x1, y2 - y1)

            # -- iris centres
            if len(lm) >= 478:
                out["left_eye"]  = (int(pts[468, 0]), int(pts[468, 1]))
                out["right_eye"] = (int(pts[473, 0]), int(pts[473, 1]))

                # -- pupillometry
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

            faces.append(out)

        return faces

    def reset(self):
        self._dist_ema.clear()
        self._l_iris_ema.clear()
        self._r_iris_ema.clear()

    # fallback: simple face detector when FaceMesh fails
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

def _draw_eye_arrows(frame, left_eye, right_eye, yaw, pitch, length=120):
    dx = int(math.sin(yaw)    * length)
    dy = int(-math.sin(pitch) * length)
    for (ex, ey) in [p for p in (left_eye, right_eye) if p]:
        # Glowing iris dot
        cv2.circle(frame, (ex, ey), 6, (0, 255, 200), -1)
        cv2.circle(frame, (ex, ey), 8, (0, 200, 150), 1)
        # Thick gaze arrow
        end = (ex + dx, ey + dy)
        cv2.arrowedLine(frame, (ex, ey), end, (0, 230, 60), 3, tipLength=0.25)
        cv2.arrowedLine(frame, (ex, ey), end, (100, 255, 130), 1, tipLength=0.25)


def _draw_gaze_arrow(frame, cx, cy, yaw, pitch, length=150):
    dx = int(math.sin(yaw)    * length)
    dy = int(-math.sin(pitch) * length)
    end = (cx + dx, cy + dy)
    cv2.arrowedLine(frame, (cx, cy), end, (0, 230, 60), 4, tipLength=0.22)
    cv2.arrowedLine(frame, (cx, cy), end, (100, 255, 130), 2, tipLength=0.22)


def _draw_gaze_dot(frame, sx, sy):
    cv2.circle(frame, (sx, sy), 22, (0, 230, 60), 2)
    cv2.circle(frame, (sx, sy), 14, (0, 230, 60), 1)
    cv2.circle(frame, (sx, sy),  5, (0, 255, 80), -1)


def _draw_hud(frame, yaw, pitch, dist_info, pupil_info, show_dist, show_pupil):
    """Draw a semi-transparent HUD panel with gaze metrics."""
    h, w = frame.shape[:2]
    yaw_d = math.degrees(yaw)
    pitch_d = math.degrees(pitch)

    # Build info lines
    lines = [f"Yaw  {yaw_d:+6.1f}", f"Pitch{pitch_d:+6.1f}"]

    if show_dist and dist_info.get("cm") is not None:
        lines.append(f"Dist  {dist_info['cm']:.0f}cm")
    if show_pupil and pupil_info is not None:
        lines.append(f"Iris  {pupil_info['mean_rel']:.3f}")

    # Panel dimensions
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    line_h = 22
    pad = 8
    panel_w = 160
    panel_h = pad * 2 + line_h * len(lines)

    # Semi-transparent background (top-left)
    x0, y0 = 6, 6
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Border
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h),
                  (0, 200, 60), 1)

    # Text
    for i, line in enumerate(lines):
        ty = y0 + pad + (i + 1) * line_h - 4
        cv2.putText(frame, line, (x0 + pad, ty), font, font_scale,
                    (0, 230, 80), thickness, cv2.LINE_AA)

    # Distance alarm (center bottom, only if alarming)
    if show_dist and dist_info.get("label"):
        label = dist_info["label"]
        color = dist_info["color"]
        sz = cv2.getTextSize(label, font, 0.8, 2)[0]
        tx = (w - sz[0]) // 2
        ty = h - 18
        # Background pill
        cv2.rectangle(frame, (tx - 8, ty - sz[1] - 6), (tx + sz[0] + 8, ty + 6),
                      (0, 0, 0), -1)
        cv2.putText(frame, label, (tx, ty), font, 0.8, color, 2, cv2.LINE_AA)


def _draw_face_box(frame, bbox, face_id=0):
    """Draw a styled face bounding box."""
    x, y, bw, bh = bbox
    # Corner brackets instead of full rectangle
    corner_len = min(20, bw // 4, bh // 4)
    color = (0, 220, 180)

    # Top-left
    cv2.line(frame, (x, y), (x + corner_len, y), color, 2)
    cv2.line(frame, (x, y), (x, y + corner_len), color, 2)
    # Top-right
    cv2.line(frame, (x + bw, y), (x + bw - corner_len, y), color, 2)
    cv2.line(frame, (x + bw, y), (x + bw, y + corner_len), color, 2)
    # Bottom-left
    cv2.line(frame, (x, y + bh), (x + corner_len, y + bh), color, 2)
    cv2.line(frame, (x, y + bh), (x, y + bh - corner_len), color, 2)
    # Bottom-right
    cv2.line(frame, (x + bw, y + bh), (x + bw - corner_len, y + bh), color, 2)
    cv2.line(frame, (x + bw, y + bh), (x + bw, y + bh - corner_len), color, 2)


# Keep old functions as stubs for gui.py imports
def _draw_distance(frame, info):
    pass

def _draw_pupil(frame, info):
    pass


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    model_path:    str,
    camera_id:     int   = 0,
    video_path:    str | None = None,
    calib_file:    str | None = None,
    run_calib:     bool  = False,
    arrow_length:  int   = 120,
    ema_alpha:     float = 0.3,
    show_distance: bool  = True,
    show_pupil:    bool  = True,
    loop_video:    bool  = True,
):
    device  = _get_device()
    print(f"[infer] Device: {device}")

    model   = load_model(model_path, device)
    mesh    = FaceMeshAnalyzer()
    ema     = EMA(alpha=ema_alpha)

    video_mode = video_path is not None

    # -- calibration
    calibrator = None
    calib_path = calib_file or "calibration.pkl"

    def _model_fn(face_bgr):
        return predict(model, face_bgr, device)

    def _detector_fn(frame_rgb):
        faces = mesh.process(frame_rgb)
        return faces[0]["face_bbox"] if faces else None

    if run_calib and not video_mode:
        from calibration import run_calibration
        calibrator = run_calibration(_model_fn, _detector_fn,
                                     camera_id=camera_id, save_path=calib_path)
    elif Path(calib_path).exists():
        from calibration import GazeCalibrator
        calibrator = GazeCalibrator.load(calib_path)

    print("[infer] Calibration:", "active" if calibrator else "none (arrow mode)")

    # -- video source
    if video_mode:
        cap = cv2.VideoCapture(video_path)
        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        vid_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[infer] Video mode: {Path(video_path).name}  "
              f"{vid_total} frames, {vid_fps:.1f} fps")
    else:
        cap = cv2.VideoCapture(camera_id)
        vid_fps = 0.0

    if not cap.isOpened():
        src = video_path or f"camera {camera_id}"
        print(f"[infer] Cannot open {src}")
        return

    controls = "Q/ESC quit | R reset | D dist | P pupil"
    if video_mode:
        controls += " | SPACE pause | LEFT/RIGHT seek"
    else:
        controls += " | C calib"
    print(f"[infer] Running — {controls}")

    _show_dist  = show_distance
    _show_pupil = show_pupil
    _paused     = False
    target_dt   = 1.0 / vid_fps if video_mode else 0.0

    while True:
        t0 = time.perf_counter()

        # Handle pause in video mode
        if video_mode and _paused:
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                _paused = False
            elif key == 81 or key == 2:   # LEFT arrow
                pos = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 30)
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                _paused = False
            elif key == 83 or key == 3:   # RIGHT arrow
                pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) + 30
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                _paused = False
            continue

        ret, frame = cap.read()
        if not ret:
            if video_mode and loop_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ema.reset()
                mesh.reset()
                continue
            elif video_mode:
                _paused = True
                continue
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces     = mesh.process(frame_rgb)
        info      = faces[0] if faces else {}

        face_bbox  = info.get("face_bbox")
        left_eye   = info.get("left_eye")
        right_eye  = info.get("right_eye")

        if face_bbox is not None:
            x, y, w, h = face_bbox
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x+w), min(frame.shape[0], y+h)

            if x2 > x1 and y2 > y1:
                face_bgr = frame[y1:y2, x1:x2]
                raw_yaw, raw_pitch = predict(model, face_bgr, device)
                smoothed = ema.update(np.array([raw_yaw, raw_pitch]))
                yaw, pitch = float(smoothed[0]), float(smoothed[1])

                cx, cy = (x1+x2)//2, (y1+y2)//2

                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

                if left_eye or right_eye:
                    _draw_eye_arrows(frame, left_eye, right_eye,
                                     yaw, pitch, arrow_length)
                else:
                    _draw_gaze_arrow(frame, cx, cy, yaw, pitch, arrow_length)

                if calibrator:
                    sx, sy = calibrator.predict(yaw, pitch)
                    _draw_gaze_dot(frame, sx, sy)

                yaw_d   = math.degrees(yaw)
                pitch_d = math.degrees(pitch)
                cv2.putText(frame, f"Yaw {yaw_d:+.1f}  Pitch {pitch_d:+.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 230, 60), 2)
                vx = math.sin(yaw)*math.cos(pitch)
                vy = -math.sin(pitch)
                vz = -math.cos(yaw)*math.cos(pitch)
                cv2.putText(frame, f"vec ({vx:+.2f}, {vy:+.2f}, {vz:+.2f})",
                            (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (180, 230, 180), 1)
        else:
            cv2.putText(frame, "No face", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2)
            ema.reset()

        if _show_dist:
            _draw_distance(frame, info.get("distance", {}))
        if _show_pupil:
            _draw_pupil(frame, info.get("pupil"))

        # Video progress bar
        if video_mode:
            fh, fw = frame.shape[:2]
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            progress = pos / max(vid_total, 1)
            bar_y = fh - 4
            cv2.rectangle(frame, (0, bar_y), (fw, fh), (60, 60, 60), -1)
            cv2.rectangle(frame, (0, bar_y), (int(fw * progress), fh),
                          (0, 200, 60), -1)
            cur_s = pos / max(vid_fps, 1)
            tot_s = vid_total / max(vid_fps, 1)
            time_str = (f"{int(cur_s)//60}:{int(cur_s)%60:02d} / "
                        f"{int(tot_s)//60}:{int(tot_s)%60:02d}")
            cv2.putText(frame, time_str, (fw - 130, fh - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        help_text = controls
        if video_mode and _paused:
            help_text = "PAUSED — " + help_text
        cv2.putText(frame, help_text,
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

        cv2.imshow("Gaze Estimation", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" ") and video_mode:
            _paused = True
        elif key == ord("c") and not video_mode:
            from calibration import run_calibration
            try:
                calibrator = run_calibration(_model_fn, _detector_fn,
                                             camera_id=camera_id,
                                             save_path=calib_path)
                ema.reset()
            except KeyboardInterrupt:
                print("[infer] Calibration cancelled")
        elif key == ord("r"):
            ema.reset()
            mesh.reset()
            print("[infer] Reset")
        elif key == ord("d"):
            _show_dist  = not _show_dist
        elif key == ord("p"):
            _show_pupil = not _show_pupil
        elif video_mode and (key == 81 or key == 2):   # LEFT
            pos = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 30)
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ema.reset()
        elif video_mode and (key == 83 or key == 3):   # RIGHT
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) + 30
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ema.reset()

        # Pace to original FPS in video mode
        if video_mode and target_dt > 0:
            elapsed = time.perf_counter() - t0
            if target_dt > elapsed:
                time.sleep(target_dt - elapsed)

    cap.release()
    cv2.destroyAllWindows()
    print("[infer] Stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(description="Real-time gaze estimation")
    p.add_argument("--model",       default=None,
                   help="Path to .pt checkpoint (auto-finds latest if omitted)")
    p.add_argument("--camera",      type=int,   default=0)
    p.add_argument("--video",       default=None,
                   help="Path to video file (test mode — replaces camera)")
    p.add_argument("--no-loop",     action="store_true",
                   help="Do not loop video at end")
    p.add_argument("--calibrate",   action="store_true")
    p.add_argument("--calib_file",  default=None)
    p.add_argument("--arrow",       type=int,   default=120)
    p.add_argument("--ema",         type=float, default=0.3)
    p.add_argument("--no_distance", action="store_true")
    p.add_argument("--no_pupil",    action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()

    model_path = args.model
    if model_path is None:
        ckpts = sorted(Path("checkpoints").glob("**/best*.pt"))
        if not ckpts:
            print("[infer] No checkpoint found. Provide --model path.")
            raise SystemExit(1)
        model_path = str(ckpts[-1])
        print(f"[infer] Auto-selected: {model_path}")

    run(
        model_path    = model_path,
        camera_id     = args.camera,
        video_path    = args.video,
        calib_file    = args.calib_file,
        run_calib     = args.calibrate,
        arrow_length  = args.arrow,
        ema_alpha     = args.ema,
        show_distance = not args.no_distance,
        show_pupil    = not args.no_pupil,
        loop_video    = not args.no_loop,
    )
