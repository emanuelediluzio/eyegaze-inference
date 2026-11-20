"""Step time and landmark statistics of the gaze engine on a recorded session.

    python bench_engine.py N2UCORE_DIR CHECKPOINT SESSION_DIR [--frames 400]

Since N2uCore_dev a5c5e98 the landmarks come from the tracking node (`Landmarks` channel), so a step has two parts,
timed apart: FaceLandmarker as the tracking node runs it (../tracking_landmarks.py), and GazeDinoEngine.analyze
(crop and model) in the gaze node. Every second frame (15 FPS, as Iride live) of the first --frames goes through both
in order. Prints the median and p90 time of each part, the share of frames with a face, and the eye opening (eye
aspect ratio) of the frames, which the blink threshold reads. The FaceMesh vs FaceLandmarker results of the README
were measured with the engine running MediaPipe itself (eyegaze-inference 181b8e5).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking_landmarks import TrackingLandmarks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n2ucore")
    ap.add_argument("checkpoint")
    ap.add_argument("session")
    ap.add_argument("--frames", type=int, default=400)
    args = ap.parse_args()
    sys.path.insert(0, args.n2ucore)
    from N2uCore.inference.gazedinoengine import GazeDinoEngine

    engine = GazeDinoEngine(args.checkpoint)
    if not engine.load():
        sys.exit("cannot load GazeDINO")
    landmarks = TrackingLandmarks(args.n2ucore)
    session = Path(args.session)
    log = [json.loads(line) for line in open(session / "log.jsonl")][:args.frames:2]
    frames = [cv2.cvtColor(cv2.imread(str(session / "frames" / f"{r['i']:06d}.jpg")), cv2.COLOR_BGR2RGB) for r in log]
    times, ok, opening = {"landmarks": [], "engine": []}, 0, []
    for i, (r, rgb) in enumerate(zip(log, frames)):
        start = time.perf_counter()
        mesh = landmarks(rgb)
        middle = time.perf_counter()
        s = engine.analyze(rgb, mesh, r["mono"] * 1e6) or {}
        if i >= 20:
            times["landmarks"].append(middle - start)
            times["engine"].append(time.perf_counter() - middle)
        if s.get("status") == "ok":
            ok += 1
            opening.append(s["iris"][2])
    engine.close()
    landmarks.close()
    steps = "   ".join(f"{k} median {np.median(1000 * np.array(v)):6.1f} ms p90 {np.percentile(1000 * np.array(v), 90):6.1f} ms"
                       for k, v in times.items())
    print(f"{steps}   face {ok / len(log):5.1%}   eye opening median {np.median(opening):.3f}"
          f"  p5 {np.percentile(opening, 5):.3f}")


if __name__ == "__main__":
    main()
