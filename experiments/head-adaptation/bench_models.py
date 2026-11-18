"""Speed of GazeDINO and a distilled student, the model alone and the whole engine step, on this machine.

    python bench_models.py N2UCORE_DIR TEACHER_CKPT STUDENT_CKPT SESSION_DIR [--backbone dinov2_vits14]

Model alone: one 224x224 crop at a time (as the engine runs), on the engine's device and precision (fp16 on a GPU)
and on the CPU in fp32. Engine step: GazeDinoEngine.analyze on 200 recorded frames (FaceMesh + crop + model).
Times are medians after a warm-up.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from extract_session import studentModel


def timeModel(model, device, dtype, runs=60):
    model = model.eval().to(device, dtype)
    x = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)
    times = []
    with torch.inference_mode():
        for i in range(runs + 10):
            start = time.perf_counter()
            model(x).cpu()
            if i >= 10:
                times.append(time.perf_counter() - start)
    return 1000 * np.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n2ucore")
    ap.add_argument("teacher")
    ap.add_argument("student")
    ap.add_argument("session")
    ap.add_argument("--backbone", default="dinov2_vits14")
    args = ap.parse_args()
    sys.path.insert(0, args.n2ucore)
    from N2uCore.inference.gazedinoengine import GazeDinoEngine

    session = Path(args.session)
    log = [json.loads(line) for line in open(session / "log.jsonl")][:400:2]
    frames = [cv2.cvtColor(cv2.imread(str(session / "frames" / f"{r['i']:06d}.jpg")), cv2.COLOR_BGR2RGB) for r in log]

    for name in ("teacher", "student"):
        engine = GazeDinoEngine(args.teacher)
        if not engine.load():
            sys.exit("cannot load GazeDINO")
        if name == "student":
            engine._model = studentModel(args.student, args.backbone).eval().to(engine._device, engine._dtype)
        params = sum(p.numel() for p in engine._model.parameters()) / 1e6
        gpu = timeModel(engine._model, engine._device, engine._dtype)
        steps = []
        for i, (r, rgb) in enumerate(zip(log, frames)):
            start = time.perf_counter()
            engine.analyze(rgb, r["mono"] * 1e6)
            if i >= 20:
                steps.append(time.perf_counter() - start)
        device = engine._device.type
        cpu = timeModel(engine._model, torch.device("cpu"), torch.float32)
        engine.close()
        print(f"{name:8s} {params:5.1f} M params   model {device} {gpu:6.1f} ms   model cpu fp32 {cpu:6.1f} ms"
              f"   engine step {1000 * np.median(steps):6.1f} ms ({1 / np.median(steps):4.1f} FPS)", flush=True)


if __name__ == "__main__":
    main()
