"""GazeDINO features of a recorded calibration session (record.py), through the engine Iride uses.

    python extract_session.py N2UCORE_DIR CHECKPOINT SESSION_DIR [--stride 2] [--student CKPT | --coreml PKG] [--out NAME.npz]

SESSION_DIR has frames/NNNNNN.jpg, log.jsonl (frame time, phase, target) and screen.json. Every --stride-th frame
(2: about 15 FPS, as Iride live) goes through N2uCore's GazeDinoEngine (landmarks as the tracking node gives them,
face crop, GazeDINO, smoothing, iris), so the angles are the ones the screen calibration gets. Hooks on the head read, for the same frame, the DINOv2
CLS token (768), the input of the last head layer (128) and the unsmoothed angles. Frames where the model did not
run (no face, blink: the engine repeats the last estimate) have fresh = False. Output: SESSION_DIR/features.npz.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking_landmarks import TrackingLandmarks  # noqa: E402


def studentModel(checkpoint, backbone):
    """A distilled GazeDINO: the engine's head on a smaller DINOv2, from the local Torch Hub cache."""
    from collections import OrderedDict

    import torch
    from torch import nn

    trunk = torch.hub.load(str(Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"), backbone,
                           source="local", pretrained=False)
    dim = trunk.embed_dim
    head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 512), nn.GELU(), nn.Dropout(0.3), nn.Linear(512, 128),
                         nn.GELU(), nn.Dropout(0.15), nn.Linear(128, 2))
    model = nn.Sequential(OrderedDict([("backbone", trunk), ("head", head)]))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"], strict=True)
    return model


class CoreMLModel(object):
    """A Core ML package (export_coreml.py) in place of the engine's model: tensor in, tensor out, on the CPU side."""

    def __init__(self, path, units="ALL"):
        import coremltools as ct

        self._model = ct.models.MLModel(path, compute_units=ct.ComputeUnit[units])

    def __call__(self, x):
        import torch

        return torch.from_numpy(self._model.predict({"image": x.float().numpy()})["gaze"].astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n2ucore")
    ap.add_argument("checkpoint")
    ap.add_argument("session")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--student", help="distilled checkpoint (distill_gaze.py) run in place of the engine's model")
    ap.add_argument("--backbone", default="dinov2_vits14", help="the student's DINOv2 backbone")
    ap.add_argument("--coreml", help="Core ML package (export_coreml.py) run in place of the engine's model")
    ap.add_argument("--units", default="ALL", help="Core ML compute units: ALL, CPU_AND_NE, CPU_AND_GPU, CPU_ONLY")
    ap.add_argument("--out", default="features.npz")
    args = ap.parse_args()
    sys.path.insert(0, args.n2ucore)
    import torch
    from N2uCore.inference.gazedinoengine import GazeDinoEngine

    engine = GazeDinoEngine(args.checkpoint)
    if not engine.load():
        sys.exit("cannot load GazeDINO")
    if args.student:
        engine._model = studentModel(args.student, args.backbone).eval().to(engine._device, engine._dtype)
    landmarks = TrackingLandmarks(args.n2ucore)
    seen = {}
    if args.coreml or not hasattr(engine._model, "head"):   # Core ML: no head to hook, only the angles are saved
        if args.coreml:
            engine._model, engine._device, engine._dtype = CoreMLModel(args.coreml, args.units), torch.device("cpu"), torch.float32
        head = None
    else:
        head = engine._model.head
        head[0].register_forward_pre_hook(lambda m, inp: seen.__setitem__("cls", inp[0][0].float().cpu().numpy()))
        head[-1].register_forward_pre_hook(lambda m, inp: seen.__setitem__("h128", inp[0][0].float().cpu().numpy()))
        head[-1].register_forward_hook(lambda m, inp, out: seen.__setitem__("raw", out[0].float().cpu().numpy()))

    session = Path(args.session)
    log = [json.loads(line) for line in open(session / "log.jsonl")][::args.stride]
    rows = {k: [] for k in ("i", "t", "phase", "k", "x", "y", "mono", "ok", "fresh", "yaw", "pitch", "iris",
                            "eye_center", "eye_span", "raw", "cls", "h128")}
    for n, r in enumerate(log):
        seen.clear()
        rgb = cv2.cvtColor(cv2.imread(str(session / "frames" / f"{r['i']:06d}.jpg")), cv2.COLOR_BGR2RGB)
        s = engine.analyze(rgb, landmarks(rgb), r["mono"] * 1e6) or {"status": "error"}
        ok = s.get("status") == "ok"
        for key in ("i", "t", "phase", "k", "x", "y", "mono"):
            rows[key].append(r[key])
        rows["ok"].append(ok)
        rows["fresh"].append(ok and "h128" in seen)
        rows["yaw"].append(s.get("yaw_rad", np.nan)); rows["pitch"].append(s.get("pitch_rad", np.nan))
        rows["iris"].append(s.get("iris", [np.nan] * 3))
        rows["eye_center"].append(s.get("eye_center_px", [np.nan] * 2)); rows["eye_span"].append(s.get("eye_span_px", np.nan))
        rows["raw"].append(seen.get("raw", np.full(2, np.nan)))
        rows["cls"].append(seen.get("cls", np.full(head[0].normalized_shape[0] if head else 1, np.nan, np.float32)))
        rows["h128"].append(seen.get("h128", np.full(128, np.nan, np.float32)))
        if n % 500 == 0:
            print(session.name, n, len(log), flush=True)
    screen = json.loads((session / "screen.json").read_text())
    np.savez_compressed(session / args.out, **{k: np.asarray(v) for k, v in rows.items()},
                        screen=np.array([screen["w"], screen["h"]]))
    engine.close()


if __name__ == "__main__":
    main()
