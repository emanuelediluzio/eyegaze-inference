"""The statistics of RULE-R1.md: calibrated error per person, its bootstrap, the paired difference between two
models over people, the gap to the teacher, and the bands. Pure numpy, no model.

Angles are (yaw, pitch) in radians, as the engine gives them and as eval_mpii.py turns the labels.
"""
import hashlib

import numpy as np

CAL_FRAMES, DRAWS, RESAMPLES = 16, 20, 10_000
YAW_BANDS = [(0, 10), (10, 20), (20, 30), (30, 180)]            # |yaw| in degrees
PITCH_BANDS = [(-90, -10), (-10, 10), (10, 90)]                 # degrees, < -10 is down
SIZE_BANDS = [(0, 0.25), (0.25, 0.4), (0.4, 10)]                # longer face box side / frame width
BAND_MIN = 20                                                   # crops of a person in a band to count the person


def vectors(a):
    return np.stack([np.cos(a[:, 1]) * np.sin(a[:, 0]), np.sin(a[:, 1]), np.cos(a[:, 1]) * np.cos(a[:, 0])], 1)


def angularError(pred, true):
    """Degrees between the directions of two (N, 2) arrays of (yaw, pitch): an error, or the gap between models."""
    return np.degrees(np.arccos(np.clip((vectors(pred) * vectors(true)).sum(1), -1, 1)))


def calibrationDraws(dataset, person, n, draws=DRAWS, k=CAL_FRAMES):
    """The 16 calibration frames of each draw: they depend only on dataset, person, n and the draw, never on the
    model. Seed: the first 8 bytes of sha256("eyegaze-r1-cal:<dataset>:<person>:<draw>"), big-endian."""
    out = []
    for d in range(draws):
        digest = hashlib.sha256(f"eyegaze-r1-cal:{dataset}:{person}:{d}".encode()).digest()
        out.append(np.random.default_rng(int.from_bytes(digest[:8], "big")).choice(n, k, replace=False))
    return out


def calibratedError(pred, true, draws):
    """(the person's error: mean over draws of the mean error on the frames not drawn, per-frame errors: each frame's
    mean over the draws where it was not a calibration frame). A line per axis is fitted on the drawn frames."""
    n = len(pred)
    total, count, means = np.zeros(n), np.zeros(n), []
    for cal in draws:
        test = np.setdiff1d(np.arange(n), cal)
        A = np.linalg.lstsq(np.column_stack([np.ones(len(cal)), pred[cal]]), true[cal], rcond=None)[0]
        err = angularError(np.column_stack([np.ones(len(test)), pred[test]]) @ A, true[test])
        means.append(err.mean())
        total[test] += err
        count[test] += 1
    frames = np.divide(total, count, out=np.full(n, np.nan), where=count > 0)
    return float(np.mean(means)), frames[count > 0]


def bootstrapMean(values, resamples=RESAMPLES, seed=0):
    """(mean, 2.5th, 97.5th percentile of the resampled means)."""
    values = np.asarray(values, np.float64)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (resamples, len(values)))].mean(1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pairedDifference(a, b):
    """Mean over people of a - b (dicts person -> error, same people) with its bootstrap interval over people."""
    people = sorted(a)
    if set(people) != set(b):
        raise ValueError("the two models must be scored on the same people")
    return bootstrapMean([a[p] - b[p] for p in people])


def evenlySpaced(n, at_most):
    """Indices of at most `at_most` of n items, evenly spaced: np.linspace rounded down, as eval_mpii.py."""
    return np.arange(n) if n <= at_most else np.linspace(0, n - 1, at_most).astype(int)


def bands(values, edges, absolute=False):
    v = np.abs(values) if absolute else values
    return [(lo, hi, (v >= lo) & (v < hi)) for lo, hi in edges]


def perBand(errors, people, values, edges, absolute=False):
    """[(lo, hi, mean over people of their mean error in the band, number of people)]; a person counts in a band
    only with at least BAND_MIN items in it."""
    out = []
    for lo, hi, mask in bands(values, edges, absolute):
        means = [errors[mask & (people == p)].mean() for p in np.unique(people)
                 if (mask & (people == p)).sum() >= BAND_MIN]
        out.append((lo, hi, float(np.mean(means)) if means else float("nan"), len(means)))
    return out
