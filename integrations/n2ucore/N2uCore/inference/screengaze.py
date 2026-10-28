import numpy as np
from numpy.polynomial import Polynomial

from N2uCore import log

from N2uCore.obj import BaseObject


def screenPoint(yaw, pitch, calibration):
    try:
        x, y = float(yaw), float(pitch)
        features = [1, x, y, x * x, x * y, y * y, x * x * x, x * x * y, x * y * y, y * y * y]
        cx, cy = calibration["cx"], calibration["cy"]
        n = len(cx)
        if n not in (6, 10) or len(cy) != n:
            log.err("Calibration must have 6 or 10 coefficients per axis")
            return None
        values = np.asarray([*cx, *cy], dtype=float)
        w, h = calibration["screen_w"], calibration["screen_h"]
        if not np.isfinite(values).all() or w <= 0 or h <= 0:
            log.err("Invalid screen calibration")
            return None
        px, py = np.dot(features[:n], cx), np.dot(features[:n], cy)
        if not np.isfinite([px, py]).all():
            log.err("Non-finite screen prediction")
            return None
        return [int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1))]
    except Exception as exc:
        log.err("screenPoint: {}".format(exc))
        return None


def validateCalibration(value):
    try:
        if not isinstance(value, dict):
            log.err("Calibration must be a JSON object")
            return None
        result = {k: value[k] for k in ("cx", "cy", "screen_w", "screen_h")}
        for k in ("screen_w", "screen_h"):
            if (
                isinstance(result[k], bool)
                or not isinstance(result[k], (int, float))
                or not np.isfinite(result[k])
                or result[k] < 2
                or int(result[k]) != result[k]
            ):
                log.err("Screen dimensions must be positive integers >= 2")
                return None
            result[k] = int(result[k])
        for k in ("cx", "cy"):
            coeff = np.asarray(result[k], dtype=float)
            if coeff.ndim != 1 or not np.isfinite(coeff).all():
                log.err("Invalid calibration coefficients")
                return None
            result[k] = coeff.tolist()
        if screenPoint(0, 0, result) is None:
            return None
        return result
    except Exception as exc:
        log.err("validateCalibration: {}".format(exc))
        return None


class ScreenCalibration(BaseObject):
    def __init__(self):
        super().__init__()
        self._calibration = None
        self._targets = []
        self._active = None
        self._size = None

    def calibration(self):
        return self._calibration

    def status(self):
        return {
            "calibrated": self._calibration is not None,
            "collecting": self._size is not None,
            "targets": len(self._targets),
            "active_samples": len(self._active["samples"]) if self._active else 0,
        }

    def begin(self, width, height):
        try:
            if (
                validateCalibration({"cx": [0] * 6, "cy": [0] * 6, "screen_w": width, "screen_h": height})
                is None
            ):
                return False
            self._size = (int(width), int(height))
            self._targets = []
            self._active = None
            return True
        except Exception as exc:
            self.err("begin: {}".format(exc))
            return None

    def target(self, x, y, now, sourceAfterUs):
        try:
            if self._size is None:
                self.err("Begin calibration first")
                return None
            if self._active is not None:
                self.err("Finish the active target first")
                return None
            if len(self._targets) >= 16:
                self.err("Maximum 16 targets per calibration")
                return None
            if not np.isfinite([x, y]).all() or not (0 <= x < self._size[0] and 0 <= y < self._size[1]):
                self.err("Target outside the calibrated screen")
                return None
            self._active = {
                "xy": [float(x), float(y)],
                "samples": [],
                "start": now,
                "sourceAfterUs": sourceAfterUs,
            }
            return True
        except Exception as exc:
            self.err("target: {}".format(exc))
            return None

    def feed(self, yaw, pitch, now, sourceTsUs):
        t = self._active
        if (
            t is not None
            and 0.98 <= now - t["start"] <= 2.8
            and sourceTsUs >= t["sourceAfterUs"] + 980000
            and np.isfinite([yaw, pitch]).all()
            and len(t["samples"]) < 128
        ):
            t["samples"].append([float(yaw), float(pitch)])

    def finishTarget(self):
        try:
            t = self._active
            if t is None:
                self.err("No active target")
                return None
            self._active = None
            if len(t["samples"]) < 3:
                self.err("At least 3 fresh face predictions are required; retry this target")
                return None
            arr = np.asarray(t["samples"])
            med = np.median(arr, axis=0)
            distances = np.linalg.norm(arr - med, axis=1)
            mask = distances <= max(1e-6, 1.5 * np.std(distances))
            if mask.sum() >= 3:
                arr = arr[mask]
            self._targets.append([*arr.mean(axis=0), *t["xy"]])
            return {"targets": len(self._targets), "samples": len(arr)}
        except Exception as exc:
            self.err("finishTarget: {}".format(exc))
            return None

    def fit(self):
        try:
            if self._size is None or self._active is not None or len(self._targets) != 16:
                self.err("Finish all 16 valid targets before fitting")
                return None
            a = np.asarray(self._targets, dtype=float)
            yaw, pitch = a[:, 0], a[:, 1]
            center = np.mean(a[:, :2], axis=0)
            scale = np.std(a[:, :2], axis=0)
            if np.any(scale < 1e-6):
                self.err("Insufficient variation of gaze angles")
                return None

            def _features(x, y):
                return np.column_stack(
                    [np.ones_like(x), x, y, x * x, x * y, y * y, x**3, x * x * y, x * y * y, y**3]
                )

            matrix = _features((yaw - center[0]) / scale[0], (pitch - center[1]) / scale[1])
            if np.linalg.matrix_rank(matrix) < 10 or np.linalg.cond(matrix) > 1e7:
                self.err("Degenerate calibration; repeat with distinct targets")
                return None
            coeff = np.linalg.lstsq(matrix, a[:, 2:], rcond=None)[0]
            # Convert normalized coefficients back to raw-angle coordinates.
            x = Polynomial([-center[0] / scale[0], 1 / scale[0]])
            y = Polynomial([-center[1] / scale[1], 1 / scale[1]])
            powers = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]
            raw = np.zeros((10, 2))
            for i, (xp, yp) in enumerate(powers):
                for j, vx in enumerate((x**xp).coef):
                    for k, vy in enumerate((y**yp).coef):
                        raw[powers.index((j, k))] += coeff[i] * vx * vy
            candidate = validateCalibration(
                {
                    "cx": raw[:, 0].tolist(),
                    "cy": raw[:, 1].tolist(),
                    "screen_w": self._size[0],
                    "screen_h": self._size[1],
                }
            )
            if candidate is None:
                return None
            residual = float(np.linalg.norm(matrix @ coeff - a[:, 2:], axis=1).mean())
            self._calibration = candidate
            self._size = None
            return {
                "targets": len(self._targets),
                "fit_residual_px": residual,
                "note": "Training residual, not independent validation error",
            }
        except Exception as exc:
            self.err("fit: {}".format(exc))
            return None

    def cancel(self):
        self._active = None
        self._size = None
        self._targets = []
        return True
