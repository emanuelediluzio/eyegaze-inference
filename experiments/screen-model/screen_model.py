"""Screen model vs polynomial calibration of the gaze on the screen.

    python screen_model.py synthetic                 # the model recovers a known geometry
    python screen_model.py sessions DIR [DIR ...]    # recorded sessions (samples.json or samples_angles.json)

Screen model: screen plane, x right / y down in px; the eye at distance D in front of the point (ex, ey).
GazeDINO angles -> true angles
    a = sx (yaw - cx)                          gain 1: D comes from the horizontal
    b = gy (pitch - cy) + k (pitch - cy)^2     vertical gain and squeeze free
and the gaze hits the screen at
    x = ex + D tan(a),   y = ey + D tan(b) / cos(a)
plus linear iris terms as in the polynomial fit. Limits of a real desk: eye in front of the middle 60% of the
width, from 0.3 H above the screen to 0.8 H down, D 0.6-4 W, pitch offset within 0.6 rad of the mean gaze,
vertical gain 0.3-3.

Polynomials as ScreenCalibration (N2uCore/inference/screengaze.py): normalized yaw x and pitch y, standardized iris
with weight 0.25 and ridge 0.5. "linear" is the fit before 2026-11-12; "x2 + x2*pitch" adds yaw^2 and yaw^2 * pitch
to the vertical axis (the candidate under test on the Mac mini).

Recorded session format: a list of samples {"phase", "k", "x", "y", "mono", "t", "sample": {"yaw_rad",
"pitch_rad", "iris"?}}; phase "calib" has the 16 calibration targets on a 4x4 grid at 12.5/37.5/62.5/87.5%,
phase "still" 9 verification targets at 25/50/75% never used in the fit. Without "iris" the iris terms drop out.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

POWERS = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]
IRIS_WEIGHT, RIDGE = 0.25, 0.5
COLLECT_SECS = (0.7, 2.0)
NAMES = ["ex", "ey", "D", "cx", "cy", "gy", "k"]


# ---------------------------------------------------------------- the two models

def irisScaler(G):
    mu, sd = G[:, 2:5].mean(0), G[:, 2:5].std(0)
    sd[sd < 1e-6] = np.inf
    return lambda g: (g[:, 2:5] - mu) / sd * IRIS_WEIGHT


def polyFit(G, S, terms):
    """Least squares on POWERS[terms] per screen axis (angles) + ridge on the iris terms."""
    mu, sd = G[:, :2].mean(0), G[:, :2].std(0)
    iris = irisScaler(G)

    def feats(g, idx):
        x, y = (g[:, 0] - mu[0]) / sd[0], (g[:, 1] - mu[1]) / sd[1]
        return np.column_stack([x**POWERS[i][0] * y**POWERS[i][1] for i in idx] + [iris(g)])

    coefs = []
    for axis, idx in enumerate(terms):
        m = feats(G, idx)
        penalty = np.zeros((m.shape[1],) * 2)
        penalty[len(idx):, len(idx):] = RIDGE * np.eye(m.shape[1] - len(idx))
        coefs.append(np.linalg.solve(m.T @ m + penalty, m.T @ S[:, axis]))
    return lambda g: np.column_stack([feats(g, idx) @ c for idx, c in zip(terms, coefs)])


def screenFit(G, S, W, H, verbose=False):
    """The screen model; with verbose, the parameters and those that ended on a limit."""
    iris = irisScaler(G)
    I = iris(G)
    sx = np.sign(np.polyfit(G[:, 0], S[:, 0], 1)[0])
    slopeY = np.polyfit(G[:, 1], S[:, 1], 1)[0]

    def predict(p, g, ir):
        ex, ey, D, cx, cy, gy, k = p[:7]
        a = sx * (g[:, 0] - cx)
        dp = g[:, 1] - cy
        b = gy * dp + k * dp * dp
        return np.column_stack((ex + D * np.tan(a), ey + D * np.tan(b) / np.cos(a))) + ir @ p[7:13].reshape(3, 2)

    def resid(p):
        priors = [(p[0] - W / 2) / (0.2 * W) * 30, (p[1] - 0.25 * H) / (0.4 * H) * 30]
        return np.concatenate(((predict(p, G, I) - S).ravel(), np.sqrt(RIDGE) * p[7:13], priors))

    sg = np.sign(slopeY)
    gLo, gHi = sorted((0.3 * sg, 3.0 * sg))
    lo = [0.2 * W, -0.3 * H, 0.6 * W, -np.inf, G[:, 1].mean() - 0.6, gLo, -np.inf] + [-np.inf] * 6
    hi = [0.8 * W, 0.8 * H, 4.0 * W, np.inf, G[:, 1].mean() + 0.6, gHi, np.inf] + [np.inf] * 6
    D0 = float(np.clip(abs(np.polyfit(G[:, 0], S[:, 0], 1)[0]), 0.61 * W, 3.99 * W))
    p0 = np.array([W / 2, 0.25 * H, D0, G[:, 0].mean() - sx * np.arctan((S[:, 0].mean() - W / 2) / D0),
                   G[:, 1].mean(), np.clip(slopeY / D0, gLo + 1e-3, gHi - 1e-3), 0.0, *np.zeros(6)])
    p = least_squares(resid, p0, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=4000).x
    if verbose:
        atLimit = [n for n, v, l, h in zip(NAMES, p[:7], lo, hi)
                   if np.isclose(v, l, rtol=1e-3) or np.isclose(v, h, rtol=1e-3)]
        print("    " + "  ".join(f"{n}={v:.3g}" for n, v in zip(NAMES, p[:7]))
              + f"  (D/W={p[2] / W:.2f}, ex/W={p[0] / W:.2f}, ey/H={p[1] / H:.2f})"
              + (f"  ON THE LIMIT: {atLimit}" if atLimit else ""))
    return lambda g: predict(p, g, iris(g))


MODELS = {
    "linear": lambda G, S, W, H: polyFit(G, S, ([0, 1, 2], [0, 1, 2])),
    "x2 + x2*pitch": lambda G, S, W, H: polyFit(G, S, ([0, 1, 2], [0, 1, 2, 3, 7])),
    "screen model": lambda G, S, W, H: screenFit(G, S, W, H),
}


def leaveOneOut(make, G, S, W, H):
    out = np.zeros_like(S)
    for i in range(len(S)):
        keep = np.arange(len(S)) != i
        out[i] = make(G[keep], S[keep], W, H)(G[i:i + 1])[0] - S[i]
    return out


# ---------------------------------------------------------------- synthetic check

def synthetic(W=1440, H=900, D=1.2, eye=(0.5, 0.15)):
    """16 targets of a perfect geometry (GazeDINO angles = true angles), no iris."""
    xs = ys = [0.08, 0.36, 0.64, 0.92]
    S = np.array([(W * x, H * y) for y in ys for x in xs])
    a = np.arctan((S[:, 0] - W * eye[0]) / (D * W))
    b = np.arctan((S[:, 1] - H * eye[1]) * np.cos(a) / (D * W))
    return np.column_stack((a, b, np.tile([0.45, 0.05, 0.3], (16, 1)))), S


def runSynthetic():
    print("Perfect geometry: leave-one-out error (px); the fitted screen model parameters")
    for D, eye in [(1.0, (0.5, 0.15)), (1.6, (0.5, 0.15)), (0.8, (0.4, 0.0)), (2.5, (0.55, 0.3))]:
        G, S = synthetic(D=D, eye=eye)
        print(f"  true D/W={D}, eye=({eye[0]} W, {eye[1]} H)")
        screenFit(G, S, 1440, 900, verbose=True)
        for name in ("linear", "screen model"):
            e = np.linalg.norm(leaveOneOut(MODELS[name], G, S, 1440, 900), axis=1)
            print(f"    {name:13s} mean {e.mean():5.1f}  max {e.max():5.1f}")


# ---------------------------------------------------------------- recorded sessions

def loadSession(path):
    path = Path(path)
    file = path / "samples.json" if (path / "samples.json").exists() else path / "samples_angles.json"
    data = json.loads(file.read_text())
    gaze = lambda s: [s["yaw_rad"], s["pitch_rad"], *s.get("iris", [0.0, 0.0, 0.0])]
    calib = [r for r in data if r["phase"] == "calib"]
    still = [r for r in data if r["phase"] == "still" and r["sample"].get("status", "ok") == "ok"]
    W, H = round(max(r["x"] for r in calib) * 8 / 7), round(max(r["y"] for r in calib) * 8 / 7)
    G, S = [], []
    for k in sorted({r["k"] for r in calib}):
        rows = [r for r in calib if r["k"] == k]
        start = rows[0]["mono"] - rows[0]["t"]
        arr = np.array([gaze(r["sample"]) for r in rows
                        if r["sample"].get("status", "ok") == "ok"
                        and COLLECT_SECS[0] <= r["mono"] - start <= COLLECT_SECS[1]])
        # outliers by the angles (median + 2.5 scaled MAD), as ScreenCalibration.finishTarget
        dist = np.linalg.norm(arr[:, :2] - np.median(arr[:, :2], axis=0), axis=1)
        mad = 1.4826 * np.median(np.abs(dist - np.median(dist)))
        keep = dist <= np.median(dist) + max(1e-6, 2.5 * mad)
        G.append((arr[keep] if keep.sum() >= 3 else arr).mean(axis=0))
        S.append((rows[0]["x"], rows[0]["y"]))
    keys = sorted({r["k"] for r in still})
    VG = np.array([np.median([gaze(r["sample"]) for r in still if r["k"] == k], axis=0) for k in keys])
    VS = np.array([next((r["x"], r["y"]) for r in still if r["k"] == k) for k in keys])
    return W, H, np.array(G), np.array(S, dtype=float), VG, VS


def runSessions(paths):
    head = ("LOO all | inner | top corners | bottom corners | bottom centre"
            " || check all | centre | bottom centre | top centre")
    rows = {m: [] for m in MODELS}
    for path in paths:
        W, H, G, S, VG, VS = loadSession(path)
        print(f"== {Path(path).name} ({W}x{H}), screen model parameters:")
        screenFit(G, S, W, H, verbose=True)
        xs, ys = sorted(set(S[:, 0])), sorted(set(S[:, 1]))
        inner = np.isin(S[:, 0], xs[1:3]) & np.isin(S[:, 1], ys[1:3])
        topC = (S[:, 1] == ys[0]) & np.isin(S[:, 0], [xs[0], xs[3]])
        botC = (S[:, 1] == ys[3]) & np.isin(S[:, 0], [xs[0], xs[3]])
        botMid = (S[:, 1] == ys[3]) & np.isin(S[:, 0], xs[1:3])
        vx, vy = sorted(set(VS[:, 0])), sorted(set(VS[:, 1]))
        at = lambda i, j: (VS[:, 0] == vx[i]) & (VS[:, 1] == vy[j])
        print("   " + head)
        for name, make in MODELS.items():
            n = np.linalg.norm(leaveOneOut(make, G, S, W, H), axis=1)
            v = np.linalg.norm(make(G, S, W, H)(VG) - VS, axis=1)
            r = [n.mean(), n[inner].mean(), n[topC].mean(), n[botC].mean(), n[botMid].mean(),
                 v.mean(), v[at(1, 1)].mean(), v[at(1, 2)].mean(), v[at(1, 0)].mean()]
            rows[name].append(r)
            print(f"   {name:13s} " + " ".join(f"{x:5.0f}" for x in r))
    print("== mean over the sessions\n   " + head)
    for name in MODELS:
        print(f"   {name:13s} " + " ".join(f"{x:5.0f}" for x in np.mean(rows[name], axis=0)))


if __name__ == "__main__":
    if sys.argv[1:2] == ["synthetic"]:
        runSynthetic()
    elif sys.argv[1:2] == ["sessions"] and len(sys.argv) > 2:
        runSessions(sys.argv[2:])
    else:
        sys.exit(__doc__)
