"""Speed of Core ML packages (export_coreml.py) on each compute unit of this machine, and their drift from fp32.

    python bench_coreml.py PKG_DIR [--units CPU_ONLY,CPU_AND_GPU,CPU_AND_NE,ALL] [--runs 60]

Per package and compute unit: load (compile) time and median time of one prediction (1x3x224x224, after a warm-up).
"drift" is the largest difference of the angles from the fp16 package of the same model on the CPU, over 20 random
inputs, in degrees (the accuracy on real frames is measured with extract_session.py --coreml).
"""
import argparse
import time
from pathlib import Path

import coremltools as ct
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkgs")
    ap.add_argument("--units", default="CPU_ONLY,CPU_AND_GPU,CPU_AND_NE,ALL")
    ap.add_argument("--runs", type=int, default=60)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    inputs = [rng.standard_normal((1, 3, 224, 224)).astype(np.float32) for _ in range(20)]
    reference = {}
    print(f"{'package':18s}{'MB':>7s}  {'units':12s}{'load s':>8s}{'ms':>8s}{'drift deg':>11s}")
    for path in sorted(Path(args.pkgs).glob("*.mlpackage")):
        mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
        name, tag = path.stem.rsplit("_", 1)
        for units in args.units.split(","):
            start = time.perf_counter()
            model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit[units])
            load = time.perf_counter() - start
            outs = np.array([model.predict({"image": x})["gaze"][0] for x in inputs])
            if tag == "fp16" and units == "CPU_ONLY":
                reference[name] = outs
            times = []
            for i in range(args.runs + 10):
                start = time.perf_counter()
                model.predict({"image": inputs[i % len(inputs)]})
                if i >= 10:
                    times.append(time.perf_counter() - start)
            drift = np.degrees(np.abs(outs - reference[name]).max()) if name in reference else float("nan")
            print(f"{path.stem:18s}{mb:7.1f}  {units:12s}{load:8.1f}{1000 * np.median(times):8.1f}{drift:11.3f}",
                  flush=True)
            del model


if __name__ == "__main__":
    main()
