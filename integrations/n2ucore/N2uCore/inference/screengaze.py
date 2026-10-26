import numpy as np


def screen_point(yaw, pitch, calibration):
    x, y = float(yaw), float(pitch)
    features = [1, x, y, x * x, x * y, y * y, x * x * x, x * x * y, x * y * y, y * y * y]
    cx, cy = calibration["cx"], calibration["cy"]
    n = len(cx)
    if n not in (6, 10) or len(cy) != n:
        raise ValueError("Calibration must have 6 or 10 coefficients per axis")
    values = np.asarray([*cx, *cy], dtype=float)
    w, h = calibration["screen_w"], calibration["screen_h"]
    if not np.isfinite(values).all() or w <= 0 or h <= 0:
        raise ValueError("Invalid screen calibration")
    px, py = np.dot(features[:n], cx), np.dot(features[:n], cy)
    if not np.isfinite([px, py]).all():
        raise ValueError("Non-finite screen prediction")
    return [int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1))]


def validate_calibration(value):
    if not isinstance(value, dict):
        raise ValueError("Calibration must be a JSON object")
    result = {k: value[k] for k in ("cx", "cy", "screen_w", "screen_h")}
    for k in ("screen_w", "screen_h"):
        if (
            isinstance(result[k], bool)
            or not isinstance(result[k], (int, float))
            or not np.isfinite(result[k])
            or result[k] < 2
            or int(result[k]) != result[k]
        ):
            raise ValueError("Screen dimensions must be positive integers >= 2")
        result[k] = int(result[k])
    for k in ("cx", "cy"):
        coeff = np.asarray(result[k], dtype=float)
        if coeff.ndim != 1 or not np.isfinite(coeff).all():
            raise ValueError("Invalid calibration coefficients")
        result[k] = coeff.tolist()
    screen_point(0, 0, result)
    return result


class ScreenCalibration:
    def __init__(self):
        self.calibration = None
        self.targets = []
        self.active = None
        self.size = None
        self.last_error = None

    def begin(self, width, height):
        validate_calibration({"cx": [0] * 6, "cy": [0] * 6, "screen_w": width, "screen_h": height})
        self.size = (int(width), int(height))
        self.targets = []
        self.active = None
        self.last_error = None

    def target(self, x, y, now, source_after_us):
        if self.size is None:
            raise ValueError("Begin calibration first")
        if self.active is not None:
            raise ValueError("Finish the active target first")
        if len(self.targets) >= 32:
            raise ValueError("Maximum 32 targets per calibration")
        if not np.isfinite([x, y]).all() or not (0 <= x < self.size[0] and 0 <= y < self.size[1]):
            raise ValueError("Target outside the calibrated screen")
        self.active = {
            "xy": [float(x), float(y)],
            "samples": [],
            "start": now,
            "source_after_us": source_after_us,
        }

    def feed(self, yaw, pitch, now, source_ts_us):
        t = self.active
        if (
            t is not None
            and 0.98 <= now - t["start"] <= 2.8
            and source_ts_us >= t["source_after_us"] + 980000
            and np.isfinite([yaw, pitch]).all()
            and len(t["samples"]) < 128
        ):
            t["samples"].append([float(yaw), float(pitch)])

    def finish_target(self):
        t = self.active
        if t is None:
            raise ValueError("No active target")
        self.active = None
        if len(t["samples"]) < 3:
            raise ValueError("At least 3 fresh face predictions are required; retry this target")
        arr = np.asarray(t["samples"])
        med = np.median(arr, axis=0)
        distances = np.linalg.norm(arr - med, axis=1)
        mask = distances <= max(1e-6, 1.5 * np.std(distances))
        if mask.sum() >= 3:
            arr = arr[mask]
        self.targets.append([*arr.mean(axis=0), *t["xy"]])
        return {"targets": len(self.targets), "samples": len(arr)}

    def fit(self):
        if self.size is None or self.active is not None or len(self.targets) < 10:
            raise ValueError("Finish at least 10 valid targets before fitting")
        a = np.asarray(self.targets, dtype=float)
        yaw, pitch = a[:, 0], a[:, 1]
        center = np.mean(a[:, :2], axis=0)
        scale = np.std(a[:, :2], axis=0)
        if np.any(scale < 1e-6):
            raise ValueError("Insufficient variation of gaze angles")

        def features(x, y):
            return np.column_stack(
                [np.ones_like(x), x, y, x * x, x * y, y * y, x**3, x * x * y, x * y * y, y**3]
            )

        matrix = features((yaw - center[0]) / scale[0], (pitch - center[1]) / scale[1])
        if np.linalg.matrix_rank(matrix) < 10 or np.linalg.cond(matrix) > 1e7:
            raise ValueError("Degenerate calibration; repeat with distinct targets")
        coeff = np.linalg.lstsq(matrix, a[:, 2:], rcond=None)[0]
        # Convert normalized coefficients back to raw-angle coordinates.
        from numpy.polynomial import Polynomial as P

        x = P([-center[0] / scale[0], 1 / scale[0]])
        y = P([-center[1] / scale[1], 1 / scale[1]])
        powers = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]
        raw = np.zeros((10, 2))
        for i, (xp, yp) in enumerate(powers):
            for j, vx in enumerate((x**xp).coef):
                for k, vy in enumerate((y**yp).coef):
                    raw[powers.index((j, k))] += coeff[i] * vx * vy
        candidate = validate_calibration(
            {
                "cx": raw[:, 0].tolist(),
                "cy": raw[:, 1].tolist(),
                "screen_w": self.size[0],
                "screen_h": self.size[1],
            }
        )
        residual = float(np.linalg.norm(matrix @ coeff - a[:, 2:], axis=1).mean())
        self.calibration = candidate
        self.size = None
        return {
            "targets": len(self.targets),
            "fit_residual_px": residual,
            "note": "Training residual, not independent validation error",
        }

    def cancel(self):
        self.active = None
        self.size = None
        self.targets = []
