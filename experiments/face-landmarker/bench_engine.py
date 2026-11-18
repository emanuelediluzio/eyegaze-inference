"""Engine step time and landmark statistics of GazeDinoEngine on a recorded session, for FaceMesh vs FaceLandmarker.

    python bench_engine.py N2UCORE_DIR CHECKPOINT SESSION_DIR [--max-faces N] [--frames 400]

N2UCORE_DIR picks the engine (a checkout with FaceMesh or with FaceLandmarker). Every second frame (15 FPS, as Iride
live) of the first --frames goes through analyze in order. Prints the median and p90 time of a step, the share of
frames with a face, and the eye opening (eye aspect ratio) of the frames, which the blink threshold reads.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n2ucore")
    ap.add_argument("checkpoint")
    ap.add_argument("session")
    ap.add_argument("--max-faces", type=int)
    ap.add_argument("--frames", type=int, default=400)
    args = ap.parse_args()
    sys.path.insert(0, args.n2ucore)
    from N2uCore.inference.gazedinoengine import GazeDinoEngine

    if args.max_faces:
        GazeDinoEngine.MAX_FACES = args.max_faces
    engine = GazeDinoEngine(args.checkpoint)
    if not engine.load():
        sys.exit("cannot load GazeDINO")
    session = Path(args.session)
    log = [json.loads(line) for line in open(session / "log.jsonl")][:args.frames:2]
    frames = [cv2.cvtColor(cv2.imread(str(session / "frames" / f"{r['i']:06d}.jpg")), cv2.COLOR_BGR2RGB) for r in log]
    times, ok, opening = [], 0, []
    for i, (r, rgb) in enumerate(zip(log, frames)):
        start = time.perf_counter()
        s = engine.analyze(rgb, r["mono"] * 1e6) or {}
        if i >= 20:
            times.append(time.perf_counter() - start)
        if s.get("status") == "ok":
            ok += 1
            opening.append(s["iris"][2])
    engine.close()
    t = 1000 * np.array(times)
    print(f"{Path(args.n2ucore).name:28s} max faces {GazeDinoEngine.MAX_FACES}   step median {np.median(t):6.1f} ms"
          f"  p90 {np.percentile(t, 90):6.1f} ms   face {ok / len(log):5.1%}   eye opening median"
          f" {np.median(opening):.3f}  p5 {np.percentile(opening, 5):.3f}")


if __name__ == "__main__":
    main()
