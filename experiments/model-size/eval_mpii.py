"""Angular error of GazeDINO variants on MPIIFaceGaze (real laptop webcams, 15 people), through the engine Iride uses.

    python eval_mpii.py N2UCORE_DIR CHECKPOINT MPII_DIR [--student CKPT | --coreml PKG [--units ALL]] [--per-subject 300]
                        [--out NAME.npz]

MPII_DIR has pXX/pXX.txt and pXX/dayYY/NNNN.jpg (the dataset as released; per line: image, screen point, 6
landmarks, head rotation and translation, face centre, gaze target, eye). The label of a frame is the direction from
the face centre to the gaze target, in camera coordinates, turned into (yaw, pitch) as GazeGene's labels. Each frame
goes alone through GazeDinoEngine (FaceMesh, face crop, model; the smoothing is reset, so the angle is the model's).
--per-subject frames per person, evenly spaced. Errors, in degrees:

  raw          the model's angles as they are
  calibrated   after a straight line per axis fitted on 16 frames of the same person (what the screen calibration
               does with 16 targets), on that person's other frames; mean of 20 random draws of the 16 frames
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "head-adaptation"))
from extract_session import CoreMLModel, studentModel  # noqa: E402


def angles(vec):
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    return np.stack([np.arctan2(x, -z), np.arctan2(-y, np.sqrt(x**2 + z**2))], axis=-1)


def angularError(pred, true):
    def vec(a):
        return np.stack([np.cos(a[:, 1]) * np.sin(a[:, 0]), np.sin(a[:, 1]), np.cos(a[:, 1]) * np.cos(a[:, 0])], 1)
    return np.degrees(np.arccos(np.clip((vec(pred) * vec(true)).sum(1), -1, 1)))


def calibrated(pred, true, rng, n=16, draws=20):
    errs = []
    for _ in range(draws):
        cal = rng.choice(len(pred), n, replace=False)
        test = np.setdiff1d(np.arange(len(pred)), cal)
        X = np.column_stack([np.ones(n), pred[cal]])
        A = np.linalg.lstsq(X, true[cal], rcond=None)[0]
        errs.append(angularError(np.column_stack([np.ones(len(test)), pred[test]]) @ A, true[test]).mean())
    return np.mean(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n2ucore")
    ap.add_argument("checkpoint")
    ap.add_argument("mpii")
    ap.add_argument("--student")
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--coreml")
    ap.add_argument("--units", default="ALL")
    ap.add_argument("--per-subject", type=int, default=300)
    ap.add_argument("--out")
    args = ap.parse_args()
    sys.path.insert(0, args.n2ucore)
    import torch
    from N2uCore.inference.gazedinoengine import GazeDinoEngine

    engine = GazeDinoEngine(args.checkpoint)
    if not engine.load():
        sys.exit("cannot load GazeDINO")
    if args.student:
        engine._model = studentModel(args.student, args.backbone).eval().to(engine._device, engine._dtype)
    if args.coreml:
        engine._model, engine._device, engine._dtype = CoreMLModel(args.coreml, args.units), torch.device("cpu"), torch.float32

    rows = {k: [] for k in ("subject", "image", "pred", "true")}
    for subjectDir in sorted(Path(args.mpii).glob("p[0-9][0-9]")):
        lines = [l.split() for l in open(subjectDir / f"{subjectDir.name}.txt")]
        for l in [lines[i] for i in np.linspace(0, len(lines) - 1, args.per_subject).astype(int)]:
            rgb = cv2.cvtColor(cv2.imread(str(subjectDir / l[0])), cv2.COLOR_BGR2RGB)
            engine.reset()
            s = engine.analyze(rgb, 0) or {}
            if s.get("status") != "ok":
                continue
            fc, gt = np.array(l[21:24], float), np.array(l[24:27], float)
            rows["subject"].append(subjectDir.name); rows["image"].append(l[0])
            rows["pred"].append([s["yaw_rad"], s["pitch_rad"]]); rows["true"].append(angles(gt - fc))
        print(subjectDir.name, sum(s == subjectDir.name for s in rows["subject"]), flush=True)
    engine.close()

    D = {k: np.asarray(v) for k, v in rows.items()}
    if args.out:
        np.savez_compressed(args.out, **D)
    rng = np.random.default_rng(0)
    raw, cal = [], []
    for s in np.unique(D["subject"]):
        m = D["subject"] == s
        raw.append(angularError(D["pred"][m], D["true"][m]).mean())
        cal.append(calibrated(D["pred"][m], D["true"][m], rng))
    print(f"{len(D['pred'])} frames, {len(raw)} people   raw {np.mean(raw):.2f} deg   calibrated {np.mean(cal):.2f} deg")
    print("per person, calibrated: " + " ".join(f"{c:.1f}" for c in cal))


if __name__ == "__main__":
    main()
