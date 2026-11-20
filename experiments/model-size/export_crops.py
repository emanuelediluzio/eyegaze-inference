"""Face crops of real frames, cut as Iride cuts them, for distillation on unlabeled real faces (distill_gaze_v2.py).

    python export_crops.py N2UCORE_DIR CHECKPOINT OUT_DIR --mpii MPII_DIR --subjects p00-p09 [--per-subject 3000]
    python export_crops.py N2UCORE_DIR CHECKPOINT OUT_DIR --frames DIR [--stride 2]

Every frame goes through GazeDinoEngine; the crop handed to the model is saved as OUT_DIR/<source>_<n>.jpg, 224x224,
the size the model sees. Frames without a face (or with the eyes closed) give no crop. No label is kept: the teacher
labels the crops during distillation.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking_landmarks import TrackingLandmarks  # noqa: E402


def subjects(spec):
    first, last = (int(s.lstrip("p")) for s in spec.split("-"))
    return [f"p{i:02d}" for i in range(first, last + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n2ucore")
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--mpii")
    ap.add_argument("--subjects", default="p00-p09")
    ap.add_argument("--per-subject", type=int, default=3000)
    ap.add_argument("--frames", help="a folder of frames (jpg/png), e.g. a recorded session's frames/")
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()
    sys.path.insert(0, args.n2ucore)
    from N2uCore.inference.gazedinoengine import GazeDinoEngine

    engine = GazeDinoEngine(args.checkpoint)
    if not engine.load():
        sys.exit("cannot load GazeDINO")
    landmarks = TrackingLandmarks(args.n2ucore)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    crop = {}
    predict = engine._predictCrop

    def keep(rgb):
        crop["rgb"] = rgb
        return predict(rgb)

    engine._predictCrop = keep

    if args.mpii:
        sources = []
        for s in subjects(args.subjects):
            lines = [l.split()[0] for l in open(Path(args.mpii) / s / f"{s}.txt")]
            picks = np.linspace(0, len(lines) - 1, min(args.per_subject, len(lines))).astype(int)
            sources += [(f"{s}_{i:04d}", Path(args.mpii) / s / lines[i]) for i in picks]
    else:
        frames = sorted(p for p in Path(args.frames).iterdir() if p.suffix.lower() in (".jpg", ".png"))
        sources = [(f"{Path(args.frames).parent.name}_{i:06d}", p) for i, p in enumerate(frames[::args.stride])]

    saved = 0
    for n, (name, path) in enumerate(sources):
        crop.clear()
        engine.reset()
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        engine.analyze(rgb, landmarks(rgb), 0)
        if "rgb" in crop:
            face = cv2.resize(cv2.cvtColor(crop["rgb"], cv2.COLOR_RGB2BGR), (224, 224), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / f"{name}.jpg"), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
        if n % 1000 == 0:
            print(n, len(sources), saved, flush=True)
    engine.close()
    print(f"{saved} crops of {len(sources)} frames in {out}")


if __name__ == "__main__":
    main()
