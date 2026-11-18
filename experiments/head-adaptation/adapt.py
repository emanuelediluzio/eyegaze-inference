"""Per-person adaptation of GazeDINO from a 16-point calibration, on GazeGene subjects held out of training.

    python adapt.py features.npz [--per-target N]

One simulated session = one subject seen by one camera (200 frames). The calibration takes the frames closest to a
4x4 grid over that session's gaze range (N frames per grid point, averaged as the real calibration averages its
samples); the error is measured on all the other frames. Methods, all predicting the true (yaw, pitch):

  raw         GazeDINO as it is
  offset      plus the mean calibration error
  affine      a straight line per axis from GazeDINO's angles (what the screen calibration does today)
  last layer  the last head layer (128 -> 2) refitted: ridge toward the trained weights (fine-tuning that layer
              with weight decay toward the original, solved in closed form)
  cls ridge   a correction of GazeDINO's angles from the 768 DINOv2 features (ridge)
  affine+ll   affine, then the last layer on what is left

Ridge strengths are chosen per session by leave-one-out on the 16 calibration points, as the screen calibration
chooses its degree.
"""
import argparse

import numpy as np

LAMBDAS = np.logspace(-3, 4, 15)


def angularError(pred, true):
    """Degrees between gaze directions given as (yaw, pitch)."""
    def vec(a):
        return np.stack([np.cos(a[:, 1]) * np.sin(a[:, 0]), np.sin(a[:, 1]), np.cos(a[:, 1]) * np.cos(a[:, 0])], 1)
    c = np.clip((vec(pred) * vec(true)).sum(1), -1, 1)
    return np.degrees(np.arccos(c))


def ridge(X, R, lam):
    """Ridge with an unpenalized intercept: coefficients for [1, X] fitting R."""
    mx, mr = X.mean(0), R.mean(0)
    Xc = X - mx
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ (R - mr))
    return np.vstack([mr - mx @ w, w])


def withOne(X):
    return np.column_stack([np.ones(len(X)), X])


def ridgeLoo(X, R, lambdas=LAMBDAS):
    """The ridge strength with the lowest leave-one-out squared error."""
    best = None
    for lam in lambdas:
        err = 0.0
        for i in range(len(X)):
            keep = np.arange(len(X)) != i
            err += np.sum((withOne(X[i:i + 1]) @ ridge(X[keep], R[keep], lam) - R[i]) ** 2)
        if best is None or err < best[1]:
            best = (lam, err)
    return best[0]


def calibrationSet(true, perTarget):
    """Indices of the frames closest to a 4x4 grid over the gaze range (perTarget frames per point)."""
    lo, hi = np.percentile(true, 5, axis=0), np.percentile(true, 95, axis=0)
    span = hi - lo
    grid = [(lo[0] + span[0] * (0.08 + 0.84 * i / 3), lo[1] + span[1] * (0.08 + 0.84 * j / 3))
            for j in range(4) for i in range(4)]
    groups, used = [], set()
    for g in grid:
        d = np.linalg.norm((true - g) / span, axis=1)
        order = [i for i in np.argsort(d) if i not in used][:perTarget]
        used.update(order)
        groups.append(order)
    return groups


def session(F, perTarget):
    """Errors (deg) of each method on one session's non-calibration frames; also the pitch gain."""
    groups = calibrationSet(F["true"], perTarget)
    cal = np.concatenate(groups)
    test = np.setdiff1d(np.arange(len(F["true"])), cal)
    avg = lambda a: np.array([a[g].mean(0) for g in groups])          # a calibration target = mean of its frames
    T, P, H, C = avg(F["true"]), avg(F["pred"]), avg(F["h128"]), avg(F["cls"])
    tT, tP, tH, tC = F["true"][test], F["pred"][test], F["h128"][test], F["cls"][test]
    out = {"raw": tP, "offset": tP + (T - P).mean(0)}

    A = [np.linalg.lstsq(withOne(P), T[:, k], rcond=None)[0] for k in range(2)]
    affine = lambda p: np.column_stack([withOne(p) @ a for a in A])
    out["affine"] = affine(tP)

    # last layer: the residual of the trained layer, ridge on its 128 inputs (= weight decay toward the trained layer)
    lam = ridgeLoo(H, T - P)
    out["last layer"] = tP + withOne(tH) @ ridge(H, T - P, lam)

    mu, sd = C.mean(0), C.std(0) + 1e-6
    Cs, tCs = (C - mu) / sd, (tC - mu) / sd
    lam = ridgeLoo(Cs, T - P)
    out["cls ridge"] = tP + withOne(tCs) @ ridge(Cs, T - P, lam)

    lam = ridgeLoo(H, T - affine(P))
    out["affine+ll"] = affine(tP) + withOne(tH) @ ridge(H, T - affine(P), lam)

    res = {}
    for name, pred in out.items():
        e = angularError(pred, tT)
        res[name] = (e.mean(), np.degrees(np.abs(pred[:, 1] - tT[:, 1])).mean(),
                     np.polyfit(tT[:, 1], pred[:, 1], 1)[0])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("--per-target", type=int, default=3)
    args = ap.parse_args()
    D = np.load(args.features)
    keys = sorted(set(zip(D["subject"], D["camera"])))
    allRes = {}
    for subj, cam in keys:
        m = (D["subject"] == subj) & (D["camera"] == cam)
        F = {k: D[k][m] for k in ("true", "pred", "h128", "cls")}
        for name, r in session(F, args.per_target).items():
            allRes.setdefault(name, {}).setdefault(subj, []).append(r)
    subjects = sorted({s for s, _ in keys})
    print(f"{len(keys)} sessions (subject x camera), {args.per_target} frames per calibration point")
    print("mean angular error (deg) per subject, on the frames not used for calibration; pitch error; pitch gain")
    print(f"{'method':12s} " + " ".join(f"{s:>9s}" for s in subjects) + "      all  pitch  gain")
    for name, per in allRes.items():
        rows = np.array([r for s in subjects for r in per[s]])
        print(f"{name:12s} " + " ".join(f"{np.mean([r[0] for r in per[s]]):9.2f}" for s in subjects)
              + f"  {rows[:, 0].mean():7.2f} {rows[:, 1].mean():6.2f} {rows[:, 2].mean():5.2f}")


if __name__ == "__main__":
    main()
