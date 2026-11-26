"""Gap from students to the teacher on the crops of export_crops_video.py shards, per person and per band
(RULE-R1.md: checkpoint selection on AVCAffe val.txt, the gap on Joel, the pilot check).

    python gap_eval.py N2UCORE_DIR SHARD_DIR --models teacher=T.pt,v3=S.pt[,r1=R.pt] [--people a,b] [--per-person 500]
                       [--out gaps.npz]

The first model is the reference. Every crop goes through GazeDinoEngine._predictCrop, the engine's own
preprocessing and model (.pt or .mlpackage), as in Iride. Per person, at most --per-person crops, evenly spaced in
the shard's order (by task, then time), the same for every model.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import r1stats as s


def loadShard(path, at_most):
    z = np.load(path)
    picks = s.evenlySpaced(len(z["offsets"]) - 1, at_most)
    jpeg, offsets = z["jpeg"], z["offsets"]
    crops = (cv2.cvtColor(cv2.imdecode(jpeg[offsets[i]:offsets[i + 1]], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
             for i in picks)
    return picks, crops, {k: z[k][picks] for k in ("task", "frame", "face_size", "yaw", "pitch")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n2ucore")
    ap.add_argument("shards")
    ap.add_argument("--models", required=True, help="name=path,... the first is the reference (the teacher)")
    ap.add_argument("--people", help="comma separated (default: every shard)")
    ap.add_argument("--per-person", type=int, default=500)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out")
    args = ap.parse_args()
    sys.path.insert(0, args.n2ucore)
    from N2uCore.inference.gazedinoengine import GazeDinoEngine

    models = [m.split("=", 1) for m in args.models.split(",")]
    engines = {}
    for name, path in models:
        engines[name] = GazeDinoEngine(path, deviceName=args.device)
        if not engines[name].load():
            sys.exit(f"cannot load {name}: {path}")
    shards = Path(args.shards)
    people = args.people.split(",") if args.people else sorted(p.stem for p in shards.glob("*.npz")
                                                               if not p.name.startswith(".") and ".partial" not in p.name)
    rows = {k: [] for k in ("person", "index", "task", "frame", "face_size", "yaw", "pitch")}
    preds = {name: [] for name, _ in models}
    for person in people:
        picks, crops, fields = loadShard(shards / f"{person}.npz", args.per_person)
        kept = 0
        for j, (i, rgb) in enumerate(zip(picks, crops)):
            out = {name: engine._predictCrop(rgb) for name, engine in engines.items()}
            if any(v is None for v in out.values()):
                continue                         # scored only on crops valid for every model
            for name, v in out.items():
                preds[name].append(v)
            rows["person"].append(person)
            rows["index"].append(i)
            for k in fields:
                rows[k].append(fields[k][j])
            kept += 1
        print(f"{person}: {kept} crops", flush=True)
    for engine in engines.values():
        engine.close()
    data = {k: np.array(v) for k, v in rows.items()} | {f"pred_{n}": np.array(v) for n, v in preds.items()}
    if args.out:
        np.savez(args.out, **data)
    report(data, [name for name, _ in models])


def report(data, names):
    reference, people = names[0], data["person"]
    gaps = {n: s.angularError(data[f"pred_{n}"], data[f"pred_{reference}"]) for n in names[1:]}
    perPerson = {n: {p: float(g[people == p].mean()) for p in np.unique(people)} for n, g in gaps.items()}
    print(f"\ngap to {reference}, degrees: mean over people [95% bootstrap over people], median, worst 3")
    for n, pp in perPerson.items():
        values = np.array(list(pp.values()))
        mean, lo, hi = s.bootstrapMean(values)
        worst = sorted(pp.items(), key=lambda kv: -kv[1])[:3]
        print(f"  {n}: {mean:.2f} [{lo:.2f}, {hi:.2f}]  median {np.median(values):.2f}  worst "
              + ", ".join(f"{p} {v:.2f}" for p, v in worst) + f"  ({len(values)} people, {len(people)} crops)")
    if len(names) == 3:
        mean, lo, hi = s.pairedDifference(perPerson[names[2]], perPerson[names[1]])
        print(f"  {names[2]} - {names[1]}: {mean:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    for title, values, edges, absolute in [("|yaw|", data["yaw"], s.YAW_BANDS, True),
                                           ("pitch", data["pitch"], s.PITCH_BANDS, False),
                                           ("face size", data["face_size"], s.SIZE_BANDS, False)]:
        print(f"\nper {title} band: mean over people (people with >= {s.BAND_MIN} crops)")
        for n, g in gaps.items():
            print(f"  {n}: " + "  ".join(f"[{lo:g},{hi:g}) {m:.2f} ({k})"
                                          for lo, hi, m, k in s.perBand(g, people, values, edges, absolute)))


if __name__ == "__main__":
    main()
