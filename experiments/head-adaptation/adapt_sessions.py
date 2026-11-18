"""Head adaptation of GazeDINO on real calibration sessions (webcam frames), against N2uCore's screen calibration.

    python adapt_sessions.py N2UCORE_DIR SESSION_DIR [SESSION_DIR ...]

Each SESSION_DIR has the features.npz of extract_session.py. The calibration is the real ScreenCalibration of
N2uCore, fed as the ScreenPointOfGaze node feeds it (16 targets, angles + iris, head pose). The adaptations add to
it a correction in screen pixels learnt on the same 16 targets, from GazeDINO's own features:

  poly             the screen calibration alone (Iride today)
  poly+h128        plus ridge from the 128 inputs of the last head layer (= refitting that layer, toward the
                   screen, with weight decay toward the trained one)
  poly+cls         plus ridge from the 768 DINOv2 features
  poly+h128 frames as poly+h128, learnt on the single frames instead of the target means
  h128 direct      ridge from the 128 features straight to the screen, without the calibration

Ridge strengths by leave-one-target-out on the calibration targets. The features are smoothed as the engine smooths
the angles. Errors on targets never used in the fit: 9 check targets in the calibration position ("still") and the
same targets after moving the head (back, left, closer, down). "target" is the error of the median point of each
target, "frame" the mean error of the single frames (from 0.7 s, when the eyes have settled).
"""
import sys
from pathlib import Path

import numpy as np

LAMBDAS = np.logspace(-3, 4, 15)
SETTLED = 0.7
PHASES = ("still", "back", "left", "closer", "down")
IMAGE_W = 640


def smooth(F, mono, ok, alpha=0.25, fps=15):
    """Exponential smoothing in time as GazeDinoEngine (restarts where the face was lost)."""
    out, ema, last = np.full_like(F, np.nan), None, None
    for i in range(len(F)):
        if not ok[i] or np.isnan(F[i]).any():
            if not ok[i]:
                ema = None
            out[i] = ema if ema is not None else np.nan
            continue
        a = 1.0 if ema is None else 1 - (1 - alpha) ** ((mono[i] - last) * fps)
        ema = F[i] if ema is None else a * F[i] + (1 - a) * ema
        last = mono[i]
        out[i] = ema
    return out


def ridge(X, R, lam):
    mx, mr = X.mean(0), R.mean(0)
    w = np.linalg.solve((X - mx).T @ (X - mx) + lam * np.eye(X.shape[1]), (X - mx).T @ (R - mr))
    return lambda Z: (Z - mx) @ w + mr


def ridgeLoo(X, R, groups):
    """Ridge with the strength of the lowest leave-one-group-out error."""
    best = None
    for lam in LAMBDAS:
        err = 0.0
        for g in np.unique(groups):
            out = groups == g
            err += np.sum((ridge(X[~out], R[~out], lam)(X[out]) - R[out]) ** 2)
        if best is None or err < best[1]:
            best = (lam, err)
    return ridge(X, R, best[0]), best[0]


def polyPoint(G, C):
    """screenPoint without the clip (the correction is added before clipping)."""
    x, y = G[:, 0], G[:, 1]
    f = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y, x**3, x * x * y, x * y * y, y**3])
    n = len(C["cx"])
    return np.column_stack([f[:, :n] @ C["cx"] + G[:, 2:5] @ C["cx_iris"], f[:, :n] @ C["cy"] + G[:, 2:5] @ C["cy_iris"]])


def runSession(path, ScreenCalibration):
    D = dict(np.load(Path(path) / "features.npz"))
    W, H = (int(v) for v in D["screen"])
    G = np.column_stack([D["yaw"], D["pitch"], D["iris"]])
    head = np.column_stack([D["eye_center"] / IMAGE_W, D["eye_span"] / IMAGE_W])
    ok, fresh, phase, k, t = D["ok"], D["fresh"], D["phase"], D["k"], D["t"]
    S = np.column_stack([D["x"], D["y"]])
    feats = {"h128": smooth(D["h128"], D["mono"], ok), "cls": smooth(D["cls"], D["mono"], ok)}

    cal = ScreenCalibration()
    cal.begin(W, H)
    calRows = []
    for target in range(16):
        rows = np.flatnonzero((phase == "calib") & (k == target))
        cal.target(*S[rows[0]], D["mono"][rows[0]] - t[rows[0]])
        for i in rows[ok[rows]]:
            cal.feed(G[i].tolist(), D["mono"][i], head[i].tolist())
        cal.finishTarget()
        calRows.append(rows[fresh[rows] & (t[rows] >= 0.7) & (t[rows] <= 2.0)])
    report = cal.fit()
    C = cal.calibration()

    # calibration targets: mean features and the calibration's own fit
    TS = np.array([S[r[0]] for r in calRows])
    TG = np.array([v[:5] for v in cal._targets])          # the target means the calibration was fitted on
    TR = TS - polyPoint(TG, C)
    groupsF = np.concatenate([[g] * len(r) for g, r in enumerate(calRows)])
    frames = np.concatenate(calRows)

    corr = {}
    for name in ("h128", "cls"):
        F = feats[name]
        TF = np.array([np.nanmean(F[r], 0) for r in calRows])
        mu, sd = TF.mean(0), TF.std(0) + 1e-6
        model, lam = ridgeLoo((TF - mu) / sd, TR, np.arange(16))
        corr["poly+" + name] = (lambda m, mu, sd, F: lambda idx: m((F[idx] - mu) / sd))(model, mu, sd, F)
    F = feats["h128"]
    mu, sd = np.nanmean(F[frames], 0), np.nanstd(F[frames], 0) + 1e-6
    frameR = S[frames] - polyPoint(G[frames], C)
    model, _ = ridgeLoo((F[frames] - mu) / sd, frameR, groupsF)
    corr["poly+h128 frames"] = (lambda idx: model((F[idx] - mu) / sd))
    TF = np.array([np.nanmean(F[r], 0) for r in calRows])
    mu2, sd2 = TF.mean(0), TF.std(0) + 1e-6
    direct, _ = ridgeLoo((TF - mu2) / sd2, TS, np.arange(16))

    methods = {
        "poly": lambda idx: polyPoint(G[idx], C),
        **{n: (lambda c: lambda idx: polyPoint(G[idx], C) + c(idx))(c) for n, c in corr.items()},
        "h128 direct": lambda idx: direct((F[idx] - mu2) / sd2),
    }
    res = {}
    for name, predict in methods.items():
        for ph in PHASES:
            tgt, frm = [], []
            for target in np.unique(k[phase == ph]):
                rows = np.flatnonzero((phase == ph) & (k == target) & fresh & (t >= SETTLED))
                rows = rows[~np.isnan(feats["cls"][rows]).any(1)]
                if len(rows) < 3:
                    continue
                P = np.clip(predict(rows), 0, [W - 1, H - 1])
                tgt.append(np.linalg.norm(np.median(P, 0) - S[rows[0]]))
                frm.append(np.linalg.norm(P - S[rows[0]], axis=1).mean())
            res[(name, ph)] = (np.mean(tgt), np.mean(frm))
    return W, H, report, res, methods


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    sys.path.insert(0, sys.argv[1])
    from N2uCore.inference.screengaze import ScreenCalibration

    allRes = []
    for path in sys.argv[2:]:
        W, H, report, res, methods = runSession(path, ScreenCalibration)
        print(f"== {Path(path).name} {W}x{H}: calibration cv {report['cv_error_px']:.0f} px, degrees {report['degrees']}")
        print(f"   {'method':17s}" + "".join(f"{p:>14s}" for p in PHASES) + "   (target / frame px)")
        for name in methods:
            print(f"   {name:17s}" + "".join(f"{res[(name, p)][0]:7.0f}/{res[(name, p)][1]:<6.0f}" for p in PHASES))
        allRes.append(res)
    print("== mean over the sessions")
    for name in methods:
        print(f"   {name:17s}" + "".join(f"{np.mean([r[(name, p)][0] for r in allRes]):7.0f}/"
                                          f"{np.mean([r[(name, p)][1] for r in allRes]):<6.0f}" for p in PHASES))


if __name__ == "__main__":
    main()
