"""
9-point gaze calibration.

Maps raw (yaw, pitch) gaze angles -> screen (x, y) pixel coordinates
using a 2nd-degree polynomial fit (6 coefficients per axis).

Usage (standalone):
    python calibration.py --model checkpoints/best.pt

Usage (from code):
    from calibration import run_calibration, GazeCalibrator
    calibrator = run_calibration(model_fn, detector_fn)
    sx, sy = calibrator.predict(yaw, pitch)
"""
from __future__ import annotations
import pickle
from pathlib import Path

import cv2
import numpy as np

# 9 calibration points as (rel_x, rel_y) fractions of screen
CALIB_POINTS_9 = [
    (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
    (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
    (0.1, 0.9), (0.5, 0.9), (0.9, 0.9),
]

CALIB_PATH_DEFAULT = "calibration.pkl"


# ---------------------------------------------------------------------------
# Polynomial feature builder
# ---------------------------------------------------------------------------

def _poly2(yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """Build degree-2 polynomial feature matrix from yaw/pitch arrays."""
    ones = np.ones_like(yaw, dtype=np.float64)
    return np.column_stack([
        ones,
        yaw, pitch,
        yaw ** 2, yaw * pitch, pitch ** 2,
    ])


# ---------------------------------------------------------------------------
# Calibrator class
# ---------------------------------------------------------------------------

class GazeCalibrator:
    """Fits and applies a 2D polynomial mapping: (yaw, pitch) -> (sx, sy)."""

    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self._samples: list[tuple[float, float, float, float]] = []
        self._cx: np.ndarray | None = None
        self._cy: np.ndarray | None = None
        self.is_fitted = False

    # ------------------------------------------------------------------
    def collect(self, yaw: float, pitch: float, sx: float, sy: float):
        self._samples.append((float(yaw), float(pitch), float(sx), float(sy)))

    # ------------------------------------------------------------------
    def fit(self):
        if len(self._samples) < 6:
            raise RuntimeError(
                f"Need at least 6 calibration samples, got {len(self._samples)}."
            )
        arr = np.array(self._samples, dtype=np.float64)
        yaws, pitches, sxs, sys_ = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        A = _poly2(yaws, pitches)
        self._cx, _, _, _ = np.linalg.lstsq(A, sxs, rcond=None)
        self._cy, _, _, _ = np.linalg.lstsq(A, sys_, rcond=None)
        self.is_fitted = True
        # Report residual
        px = A @ self._cx
        py = A @ self._cy
        err = np.sqrt(((px - sxs) ** 2 + (py - sys_) ** 2)).mean()
        print(f"[calib] Fit residual: {err:.1f} px  ({len(self._samples)} points)")

    # ------------------------------------------------------------------
    def predict(self, yaw: float, pitch: float) -> tuple[int, int]:
        if not self.is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        A = _poly2(np.array([yaw]), np.array([pitch]))
        px = float(A @ self._cx)
        py = float(A @ self._cy)
        return (
            int(np.clip(px, 0, self.screen_w - 1)),
            int(np.clip(py, 0, self.screen_h - 1)),
        )

    # ------------------------------------------------------------------
    def save(self, path: str = CALIB_PATH_DEFAULT):
        with open(path, "wb") as f:
            pickle.dump({
                "cx": self._cx,
                "cy": self._cy,
                "screen_w": self.screen_w,
                "screen_h": self.screen_h,
                "samples": self._samples,
            }, f)
        print(f"[calib] Calibration saved -> {path}")

    @classmethod
    def load(cls, path: str = CALIB_PATH_DEFAULT) -> "GazeCalibrator":
        with open(path, "rb") as f:
            d = pickle.load(f)
        c = cls(d["screen_w"], d["screen_h"])
        c._cx = d["cx"]
        c._cy = d["cy"]
        c._samples = d.get("samples", [])
        c.is_fitted = True
        print(f"[calib] Calibration loaded <- {path}")
        return c


# ---------------------------------------------------------------------------
# Interactive calibration routine
# ---------------------------------------------------------------------------

def run_calibration(
    model_fn,
    detector_fn,
    camera_id: int = 0,
    dwell_s: float = 2.5,
    save_path: str = CALIB_PATH_DEFAULT,
) -> GazeCalibrator:
    """
    Full interactive calibration.

    Parameters
    ----------
    model_fn    : callable(face_bgr: np.ndarray) -> (yaw: float, pitch: float)
    detector_fn : callable(frame_rgb: np.ndarray) -> (x, y, w, h) | None
    camera_id   : webcam index
    dwell_s     : seconds to collect samples per dot
    save_path   : where to save the calibration file

    Returns
    -------
    Fitted GazeCalibrator
    """
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    # Detect screen size
    dummy = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.imshow("Calibration", dummy)
    cv2.waitKey(1)
    rect = cv2.getWindowImageRect("Calibration")
    sw, sh = rect[2], rect[3]
    if sw <= 0 or sh <= 0:
        sw, sh = 1920, 1080

    calibrator = GazeCalibrator(sw, sh)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    # -- instruction screen
    canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
    msg = "Eye Gaze Calibration"
    sub = "Keep your head still. Follow each white dot with your eyes."
    sub2 = "Press SPACE to start   |   ESC to cancel"
    cv2.putText(canvas, msg,  (sw // 2 - 280, sh // 2 - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
    cv2.putText(canvas, sub,  (sw // 2 - 380, sh // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 1)
    cv2.putText(canvas, sub2, (sw // 2 - 260, sh // 2 + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 100), 1)
    cv2.imshow("Calibration", canvas)
    while True:
        k = cv2.waitKey(30) & 0xFF
        if k == 32:   # SPACE
            break
        if k == 27:   # ESC
            cap.release()
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Calibration cancelled by user.")

    # -- dot loop
    for idx, (rx, ry) in enumerate(CALIB_POINTS_9):
        px, py = int(rx * sw), int(ry * sh)
        collected: list[tuple[float, float]] = []
        t_start = cv2.getTickCount()
        freq = cv2.getTickFrequency()

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            elapsed = (cv2.getTickCount() - t_start) / freq
            frac = min(elapsed / dwell_s, 1.0)

            # -- draw
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            # progress ring
            cv2.ellipse(canvas, (px, py), (24, 24), -90,
                        0, int(360 * frac), (80, 210, 80), 3)
            # dot
            cv2.circle(canvas, (px, py), 13, (255, 255, 255), -1)
            cv2.circle(canvas, (px, py),  4, (0,   0,   0),   -1)
            # counter
            cv2.putText(
                canvas,
                f"Point {idx + 1} / {len(CALIB_POINTS_9)}",
                (sw // 2 - 80, sh - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (160, 160, 160), 1,
            )
            cv2.imshow("Calibration", canvas)

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                cap.release()
                cv2.destroyAllWindows()
                raise KeyboardInterrupt("Calibration cancelled by user.")

            # collect samples after initial 40% of dwell
            if elapsed > dwell_s * 0.4:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                det = detector_fn(frame_rgb)
                if det is not None:
                    x, y, w, h = det
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                    if x2 > x1 and y2 > y1:
                        yaw, pitch = model_fn(frame[y1:y2, x1:x2])
                        collected.append((yaw, pitch))

            if elapsed >= dwell_s:
                break

        if collected:
            mean_yaw   = float(np.mean([s[0] for s in collected]))
            mean_pitch = float(np.mean([s[1] for s in collected]))
            calibrator.collect(mean_yaw, mean_pitch, float(px), float(py))
            print(
                f"[calib] Point {idx + 1}: ({px},{py}) "
                f"yaw={mean_yaw:+.3f} pitch={mean_pitch:+.3f} "
                f"({len(collected)} samples)"
            )
        else:
            print(f"[calib] Point {idx + 1}: no face detected — skipped")

    cap.release()
    cv2.destroyAllWindows()

    calibrator.fit()
    calibrator.save(save_path)
    print("[calib] Done!")
    return calibrator


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run 9-point gaze calibration")
    p.add_argument("--model",  required=True, help="Path to best.pt checkpoint")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--dwell",  type=float, default=2.5,
                   help="Seconds to collect per dot (default: 2.5)")
    p.add_argument("--out",    default=CALIB_PATH_DEFAULT,
                   help="Output calibration file")
    args = p.parse_args()

    import torch
    from models.gaze_model import GazeDINO

    device = (
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )

    ckpt = torch.load(args.model, map_location=device)
    model = GazeDINO().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    import torchvision.transforms as T
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def model_fn(face_bgr):
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        inp = transform(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inp)[0].cpu().numpy()
        return float(out[0]), float(out[1])

    import mediapipe as mp
    fd = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )

    def detector_fn(frame_rgb):
        h, w = frame_rgb.shape[:2]
        res = fd.process(frame_rgb)
        if not res.detections:
            return None
        best = max(res.detections, key=lambda d: d.score[0])
        bb = best.location_data.relative_bounding_box
        return (
            max(0, int(bb.xmin * w)),
            max(0, int(bb.ymin * h)),
            int(bb.width * w),
            int(bb.height * h),
        )

    run_calibration(model_fn, detector_fn,
                    camera_id=args.camera,
                    dwell_s=args.dwell,
                    save_path=args.out)
