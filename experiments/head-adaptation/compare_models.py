"""Screen error of two GazeDINO models on the same recorded sessions, each with its own screen calibration.

    python compare_models.py N2UCORE_DIR NAME.npz[,NAME.npz ...] SESSION_DIR [SESSION_DIR ...]

The npz files are outputs of extract_session.py (e.g. features.npz for the teacher, student.npz for the distilled
model). Per model and session: N2uCore's ScreenCalibration on the 16 targets (as the ScreenPointOfGaze node feeds it),
its leave-one-out error, then the error on targets never used in the fit, as in adapt_sessions.py.
"""
import sys
from pathlib import Path

import numpy as np

from adapt_sessions import IMAGE_W, PHASES, SETTLED, polyPoint


def evaluate(path, ScreenCalibration):
    D = dict(np.load(path))
    W, H = (int(v) for v in D["screen"])
    G = np.column_stack([D["yaw"], D["pitch"], D["iris"]])
    head = np.column_stack([D["eye_center"] / IMAGE_W, D["eye_span"] / IMAGE_W])
    ok, phase, k, t = D["ok"], D["phase"], D["k"], D["t"]
    S = np.column_stack([D["x"], D["y"]])
    cal = ScreenCalibration()
    cal.begin(W, H)
    for target in range(16):
        rows = np.flatnonzero((phase == "calib") & (k == target))
        cal.target(*S[rows[0]], D["mono"][rows[0]] - t[rows[0]])
        for i in rows[ok[rows]]:
            cal.feed(G[i].tolist(), D["mono"][i], head[i].tolist())
        cal.finishTarget()
    report = cal.fit()
    C = cal.calibration()
    res = {"cv": report["cv_error_px"]}
    for ph in PHASES:
        tgt, frm = [], []
        for target in np.unique(k[phase == ph]):
            rows = np.flatnonzero((phase == ph) & (k == target) & ok & (t >= SETTLED))
            if len(rows) < 3:
                continue
            P = np.clip(polyPoint(G[rows], C), 0, [W - 1, H - 1])
            tgt.append(np.linalg.norm(np.median(P, 0) - S[rows[0]]))
            frm.append(np.linalg.norm(P - S[rows[0]], axis=1).mean())
        res[ph] = (np.mean(tgt), np.mean(frm))
    return res


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    sys.path.insert(0, sys.argv[1])
    from N2uCore.inference.screengaze import ScreenCalibration

    names, sessions = sys.argv[2].split(","), sys.argv[3:]
    head = f"   {'model':14s}{'cv':>6s}" + "".join(f"{p:>14s}" for p in PHASES) + "   (target / frame px)"
    allRes = {n: [] for n in names}
    for s in sessions:
        print(f"== {Path(s).name}\n{head}")
        for n in names:
            r = evaluate(Path(s) / n, ScreenCalibration)
            allRes[n].append(r)
            print(f"   {n:14s}{r['cv']:6.0f}" + "".join(f"{r[p][0]:7.0f}/{r[p][1]:<6.0f}" for p in PHASES))
    print(f"== mean over the sessions\n{head}")
    for n in names:
        R = allRes[n]
        print(f"   {n:14s}{np.mean([r['cv'] for r in R]):6.0f}"
              + "".join(f"{np.mean([r[p][0] for r in R]):7.0f}/{np.mean([r[p][1] for r in R]):<6.0f}" for p in PHASES))


if __name__ == "__main__":
    main()
