"""Measure local GazeDINO inference using one existing RGB image, without a camera."""
import argparse
import hashlib
import json
import platform
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image

from N2uCore.dyn.wrk.memprovider import resolveComputeResources
from N2uCore.inference.gazedinoengine import GazeDinoEngine


def benchmark(checkpoint, imagePath, deviceName, frames=30, warmup=5, maxFps=0):
    if frames < 1 or warmup < 0 or maxFps < 0:
        raise ValueError("frames must be positive; warmup and maxFps must be nonnegative")
    resources = resolveComputeResources(deviceName, GazeDinoEngine.SUPPORTED_DEVICES)
    if resources is None:
        return None
    deviceName, _ = resources
    imagePath = Path(imagePath)
    with Image.open(imagePath) as image:
        rgb = np.asarray(image.convert("RGB"))
    process = psutil.Process()
    baselineRss = process.memory_info().rss
    peakRss = baselineRss
    stop = threading.Event()

    def sampleMemory():
        nonlocal peakRss
        while not stop.wait(.05):
            peakRss = max(peakRss, process.memory_info().rss)

    def deviceMemory():
        if deviceName == "mps":
            return {
                "tensor_bytes": torch.mps.current_allocated_memory(),
                "driver_bytes": torch.mps.driver_allocated_memory(),
            }
        return None

    engine = GazeDinoEngine(checkpoint, deviceName=deviceName)
    sampler = threading.Thread(target=sampleMemory, daemon=True)
    sampler.start()
    try:
        started = time.perf_counter()
        if not engine.load():
            return None
        loadSecs = time.perf_counter() - started
        loadedRss = process.memory_info().rss
        for _ in range(warmup):
            engine.analyze(rgb)
        latencies = []
        statuses = Counter()
        peakDevice = deviceMemory()
        cpuBefore = process.cpu_times()
        started = time.perf_counter()
        for _ in range(frames):
            frameStart = time.perf_counter()
            result = engine.analyze(rgb)
            latencies.append(time.perf_counter() - frameStart)
            statuses[result["status"]] += 1
            memory = deviceMemory()
            if memory is not None:
                peakDevice = {key: max(peakDevice[key], memory[key]) for key in memory}
            if maxFps:
                stop.wait(max(0, 1 / maxFps - (time.perf_counter() - frameStart)))
        elapsed = time.perf_counter() - started
        cpuAfter = process.cpu_times()
        peakRss = max(peakRss, process.memory_info().rss)
        return {
            "machine": platform.machine(), "os": platform.platform(), "processor": platform.processor(),
            "torch": torch.__version__, "device": deviceName,
            "logical_cpus": psutil.cpu_count(), "system_ram_bytes": psutil.virtual_memory().total,
            "image_sha256": hashlib.sha256(imagePath.read_bytes()).hexdigest(),
            "image_size_px": [rgb.shape[1], rgb.shape[0]],
            "frames": frames, "warmup_frames": warmup, "max_fps": maxFps or None,
            "load_seconds": loadSecs, "elapsed_seconds": elapsed,
            "achieved_fps": frames / elapsed,
            "analysis_ms": {"mean": float(np.mean(latencies) * 1000),
                            "p50": float(np.percentile(latencies, 50) * 1000),
                            "p95": float(np.percentile(latencies, 95) * 1000)},
            "cpu_percent_one_core": 100 * ((cpuAfter.user + cpuAfter.system) - (cpuBefore.user + cpuBefore.system)) / elapsed,
            "rss_baseline_bytes": baselineRss, "rss_loaded_bytes": loadedRss,
            "rss_peak_bytes": peakRss, "rss_peak_delta_bytes": peakRss - baselineRss,
            "mps_peak_bytes": peakDevice, "statuses": dict(statuses),
            "scope": "Full engine analysis of a repeated RGB image; excludes camera, video encoding, Flow transport and browser. MPS uses unified RAM; do not add its counters to RSS. No power measurement.",
        }
    finally:
        try:
            engine.close()
        finally:
            stop.set()
            sampler.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-fps", type=float, default=0, help="0 measures maximum throughput")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark(args.checkpoint, args.image, args.device, args.frames, args.warmup, args.max_fps)
    if report is None:
        parser.exit(1, "GazeDINO benchmark initialization failed.\n")
    data = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(data)
    print(data, end="")


if __name__ == "__main__":
    main()
