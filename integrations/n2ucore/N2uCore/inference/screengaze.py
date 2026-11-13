from __future__ import annotations

import numpy as np
from numpy.polynomial import Polynomial

from N2uCore import log

from N2uCore.obj import BaseObject

def screenPoint(gaze, calibration):
    """Screen pixel of the gaze [yaw, pitch, *iris] with the calibration, clipped to the screen: polynomial in the
    angles plus a linear term in the iris."""
    x, y, iris = gaze[0], gaze[1], gaze[2:]
    features = [1, x, y, x * x, x * y, y * y, x * x * x, x * x * y, x * y * y, y * y * y]
    cx, cy = calibration["cx"], calibration["cy"]
    n = len(cx)
    px = np.dot(features[:n], cx) + np.dot(iris, calibration["cx_iris"])
    py = np.dot(features[:n], cy) + np.dot(iris, calibration["cy_iris"])
    return [int(np.clip(px, 0, calibration["screen_w"] - 1)), int(np.clip(py, 0, calibration["screen_h"] - 1))]


def headPose(sample, imageSize):
    # [eyes x, eyes y, eye span] in image-width units; the eye span follows the camera distance, the face box
    # does not (17% closer gave a 3% bigger box)
    x, y = sample["eye_center_px"]
    return [x / imageSize["w"], y / imageSize["w"], sample["eye_span_px"] / imageSize["w"]]


def validateCalibration(value):
    try:
        if not isinstance(value, dict):
            log.err("Calibration must be a JSON object")
            return None
        result = {k: value[k] for k in ("cx", "cy", "cx_iris", "cy_iris", "screen_w", "screen_h")}
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
        for k in ("cx", "cy", "cx_iris", "cy_iris"):
            coeff = np.asarray(result[k], dtype=float)
            if coeff.ndim != 1 or not np.isfinite(coeff).all():
                log.err("Invalid calibration coefficients")
                return None
            result[k] = coeff.tolist()
        if len(result["cx"]) not in (6, 10) or len(result["cy"]) != len(result["cx"]):
            log.err("Calibration must have 6 or 10 coefficients per axis")
            return None
        if len(result["cx_iris"]) != 3 or len(result["cy_iris"]) != 3:
            log.err("Calibration must have 3 iris coefficients per axis")
            return None
        return result
    except Exception as exc:
        log.err("validateCalibration: {}".format(exc))
        return None


class ScreenCalibration(BaseObject):
    POWERS = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]
    # polynomial degree -> terms (indices in POWERS), horizontal and vertical screen axis. The vertical line keeps
    # yaw^2 and yaw^2 * pitch: a straight line puts the top corners toward the centre (GazeDINO squeezes the pitch
    # looking up, more sideways), yaw^2 alone also pulled the bottom centre down. Three recordings, left out of the
    # fit: top corners 150 -> 118 px, centre 76 -> 64 px, bottom centre 67 px as before, bottom corners 60 -> 80 px
    AXIS_TERMS = ({1: [0, 1, 2], 2: list(range(6)), 3: list(range(10))},
                  {1: [0, 1, 2, 3, 7], 2: [0, 1, 2, 3, 4, 5, 7], 3: list(range(10))})
    # how far the head may move from where it was during calibration, measured on a recording (error 65 px in place):
    # 17-26% closer/farther gave 240-330 px, 0.5 eye spans down 160 px, 0.9 eye spans sideways only 77 px
    DRIFT_SCALE = 0.08          # relative eye span change (closer / farther)
    DRIFT_SHIFT_Y = 0.25        # vertical shift of the eyes, in eye spans
    DRIFT_SHIFT_X = 1.0         # sideways shift of the eyes, in eye spans (it matters little)
    COLLECT_SECS = (0.7, 2.0)   # samples of a target: from when the gaze settled to the end of the 2 s target
    # how much the iris counts against the angles (both standardized on the targets, ridge RIDGE on the iris terms
    # only, the angles stay least squares): on three recordings 0.25 lowered the error 15-30%, most with the head
    # lower (277 -> 175 px); 0.5 got worse sideways
    IRIS_WEIGHT = 0.25
    RIDGE = 0.5

    def __init__(self):
        super().__init__()
        self._calibration = None
        self._reference = None      # median head pose [eyes x, eyes y, eye span] during the calibration in use
        self._targets = []
        self._heads = []            # median head pose of each completed target
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
            empty = {"cx": [0] * 6, "cy": [0] * 6, "cx_iris": [0] * 3, "cy_iris": [0] * 3}
            if validateCalibration({**empty, "screen_w": width, "screen_h": height}) is None:
                return False
            self._size = (int(width), int(height))
            self._targets = []
            self._heads = []
            self._active = None
            return True
        except Exception as exc:
            self.err("begin: {}".format(exc))
            return None

    def target(self, x, y, now):
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
                "heads": [],
                "start": now,
            }
            return True
        except Exception as exc:
            self.err("target: {}".format(exc))
            return None

    def feed(self, gaze, ts, head):
        t = self._active
        if (
            t is not None
            and self.COLLECT_SECS[0] <= ts - t["start"] <= self.COLLECT_SECS[1]
            and len(t["samples"]) < 128
        ):
            t["samples"].append([float(v) for v in gaze])
            t["heads"].append(head)

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
            # outliers by the angles; the target keeps the mean of angles and iris
            med = np.median(arr[:, :2], axis=0)
            distances = np.linalg.norm(arr[:, :2] - med, axis=1)
            # robust cut (median + 2.5 scaled MAD): the old 1.5 * std of distances kept only ~40% of good samples
            mad = 1.4826 * np.median(np.abs(distances - np.median(distances)))
            mask = distances <= np.median(distances) + max(1e-6, 2.5 * mad)
            if mask.sum() >= 3:
                arr = arr[mask]
            self._targets.append([*arr.mean(axis=0), *t["xy"]])
            self._heads.append(np.median(np.asarray(t["heads"]), axis=0).tolist())
            return {"targets": len(self._targets), "samples": len(arr)}
        except Exception as exc:
            self.err("finishTarget: {}".format(exc))
            return None

    @classmethod
    def _features(cls, x, y, iris, terms):
        """The ``terms`` (indices in POWERS) of normalized yaw x and pitch y, then the weighted normalized iris."""
        return np.column_stack([x**cls.POWERS[i][0] * y**cls.POWERS[i][1] for i in terms] + [iris])

    @classmethod
    def _solve(cls, matrix, values, terms):
        """Coefficients with the iris terms shrunk (ridge), or None for degenerate angles (rank or conditioning)."""
        angles = matrix[:, :terms]
        if np.linalg.matrix_rank(angles) < terms or np.linalg.cond(angles) > 1e7:
            return None
        penalty = np.zeros((matrix.shape[1], matrix.shape[1]))
        penalty[terms:, terms:] = cls.RIDGE * np.eye(matrix.shape[1] - terms)
        return np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ values)

    def _crossValidate(self, x, y, iris, values, terms):
        """Leave-one-out pixel errors along one screen axis of a fit with ``terms`` angle terms, or None if a fold is
        degenerate."""
        errors = []
        for i in range(len(values)):
            keep = np.arange(len(values)) != i
            coeff = self._solve(self._features(x[keep], y[keep], iris[keep], terms), values[keep], len(terms))
            if coeff is None:
                return None
            predicted = self._features(x[i:i + 1], y[i:i + 1], iris[i:i + 1], terms) @ coeff
            errors.append(abs(predicted[0] - values[i]))
        return np.asarray(errors)

    def fit(self):
        # for each screen axis the polynomial degree (1-3) with the lowest leave-one-out error wins: with 16
        # targets a cubic can follow the noise of single targets
        try:
            if self._size is None or self._active is not None or len(self._targets) != 16:
                self.err("Finish all 16 valid targets before fitting")
                return None
            a = np.asarray(self._targets, dtype=float)
            center = np.mean(a[:, :2], axis=0)
            scale = np.std(a[:, :2], axis=0)
            if np.any(scale < 1e-6):
                self.err("Insufficient variation of gaze angles")
                return None
            x, y, screen = (a[:, 0] - center[0]) / scale[0], (a[:, 1] - center[1]) / scale[1], a[:, 5:]
            irisCenter = np.mean(a[:, 2:5], axis=0)
            irisScale = np.std(a[:, 2:5], axis=0)
            irisScale[irisScale < 1e-6] = np.inf     # a steady iris feature carries nothing: its term is 0
            iris = (a[:, 2:5] - irisCenter) / irisScale * self.IRIS_WEIGHT
            best = []
            for axis, degreeTerms in enumerate(self.AXIS_TERMS):
                choice = None
                for degree, terms in degreeTerms.items():
                    errors = self._crossValidate(x, y, iris, screen[:, axis], terms)
                    if errors is not None and (choice is None or errors.mean() < choice[1].mean()):
                        choice = (degree, errors)
                if choice is None:
                    self.err("Degenerate calibration; repeat with distinct targets")
                    return None
                best.append(choice)
            degrees = [best[0][0], best[1][0]]
            errors = np.hypot(best[0][1], best[1][1])
            coeff = np.zeros((10, 2))
            rawIris = np.zeros((3, 2))
            fitted = np.zeros_like(screen)
            for axis, degree in enumerate(degrees):
                terms = self.AXIS_TERMS[axis][degree]
                matrix = self._features(x, y, iris, terms)
                solved = self._solve(matrix, screen[:, axis], len(terms))
                coeff[terms, axis] = solved[:len(terms)]
                # iris terms back to raw iris values
                rawIris[:, axis] = solved[len(terms):] * self.IRIS_WEIGHT / irisScale
                fitted[:, axis] = matrix @ solved
            # Convert normalized coefficients back to raw-angle coordinates.
            px = Polynomial([-center[0] / scale[0], 1 / scale[0]])
            py = Polynomial([-center[1] / scale[1], 1 / scale[1]])
            raw = np.zeros((10, 2))
            for i, (xp, yp) in enumerate(self.POWERS):
                for j, vx in enumerate((px**xp).coef):
                    for k, vy in enumerate((py**yp).coef):
                        raw[self.POWERS.index((j, k))] += coeff[i] * vx * vy
            # the iris centering into the constant term
            raw[0] -= irisCenter @ rawIris
            stored = 6 if not raw[6:].any() else 10   # screenPoint reads 6 (quadratic) or 10 (cubic) terms
            candidate = validateCalibration(
                {
                    "cx": raw[:stored, 0].tolist(),
                    "cy": raw[:stored, 1].tolist(),
                    "cx_iris": rawIris[:, 0].tolist(),
                    "cy_iris": rawIris[:, 1].tolist(),
                    "screen_w": self._size[0],
                    "screen_h": self._size[1],
                }
            )
            if candidate is None:
                return None
            residual = float(np.linalg.norm(fitted - screen, axis=1).mean())
            self._calibration = candidate
            self._reference = np.median(np.asarray(self._heads), axis=0).tolist()
            self._size = None
            return {
                "targets": len(self._targets),
                "degree": max(degrees),
                "degrees": degrees,     # [horizontal, vertical]
                "fit_residual_px": residual,
                "cv_error_px": float(errors.mean()),
                "cv_p90_px": float(np.percentile(errors, 90)),
                "note": "cv_error_px: mean error at each target left out of the fit (leave-one-out)",
            }
        except Exception as exc:
            self.err("fit: {}".format(exc))
            return None

    def reference(self):
        return self._reference

    def drift(self, head):
        # the hint is for the move furthest beyond its threshold, from the user's point of view: the image is
        # not mirrored, so eyes moving right in the image mean the user moved to their left ("move_right")
        cx, cy, size = self._reference
        dx, dy = (head[0] - cx) / size, (head[1] - cy) / size
        ratio = head[2] / size
        over = {
            "move_closer" if ratio < 1 else "move_back": abs(ratio - 1) / self.DRIFT_SCALE,
            "move_up" if dy > 0 else "move_down": abs(dy) / self.DRIFT_SHIFT_Y,
            "move_right" if dx > 0 else "move_left": abs(dx) / self.DRIFT_SHIFT_X,
        }
        hint = max(over, key=over.get)
        if over[hint] <= 1:
            hint = "ok"
        return {
            "drifted": hint != "ok",
            "hint": hint,
            "dx": float(dx),
            "dy": float(dy),
            "scale": float(ratio),
            "reference": [cx, cy, size],
            "current": [float(v) for v in head],
        }

    def cancel(self):
        self._active = None
        self._size = None
        self._targets = []
        self._heads = []
        return True
