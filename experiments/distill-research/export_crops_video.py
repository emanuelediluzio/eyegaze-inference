"""Face crops of videos, cut as Iride cuts them, one shard per person, for the R1 distillation (RULE-R1.md).

    python export_crops_video.py N2UCORE_DIR VIDEO_ROOT OUT_DIR [--people aiim005,aiim006] [--every 2.0]
    python export_crops_video.py N2UCORE_DIR FRAMES_DIR OUT_DIR --frames NAME [--every 2.0 --fps 15]

VIDEO_ROOT has one folder per person with that person's videos (AVCAffe: aiimNNN/task_K.mp4; the macOS ._* files
are skipped). With --frames, FRAMES_DIR is a folder of numbered frames (a recorded session's frames/) and NAME the
person; --fps gives the frame rate to turn --every into a frame step.

One frame every --every seconds goes through the tracking node's FaceLandmarker (tracking_landmarks.py) and the
engine's face box (GazeDinoEngine._faceBox, no model loaded); the box is resized to 224x224 as the engine does.
Dropped, and counted: no face, eyes closed (eye aspect ratio under BLINK_EAR), face box at the frame edge.

OUT_DIR/<person>.npz holds, per kept crop:
  jpeg, offsets   the JPEG bytes (quality 95) of all crops in one uint8 array, crop i is jpeg[offsets[i]:offsets[i+1]]
  task, frame     the video index in the person's sorted list (0 with --frames) and the frame number
  face_size       longer face box side / frame width
  yaw, pitch      head pose in degrees from FaceLandmarker's transformation matrix: yaw > 0 turned towards the image
                  right, pitch > 0 up
and the counts: sampled, no_face, edge, closed; per task, task_names and task_sampled (frames sampled). A person with a shard is skipped (rerun to resume). The shard is
written to a temporary name and renamed, so a stopped run leaves no half shard.
"""
import argparse
import math
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking_landmarks import TrackingLandmarks  # noqa: E402

SIZE = 224


def headPose(matrix):
    """(yaw, pitch) in degrees of the face's forward axis (canonical +z) in camera space (x right, y up)."""
    forward = matrix[:3, 2]
    return math.degrees(math.atan2(forward[0], forward[2])), math.degrees(math.atan2(forward[1], math.hypot(forward[0], forward[2])))


def natural(path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def videoFrames(path, every):
    """(frame number, BGR frame) every `every` seconds, reading the video once in order."""
    capture = cv2.VideoCapture(str(path))
    step = max(1, round(every * (capture.get(cv2.CAP_PROP_FPS) or 25)))
    n = 0
    while capture.grab():
        if n % step == 0:
            ok, frame = capture.retrieve()
            if ok:
                yield n, frame
        n += 1
    capture.release()


def folderFrames(folder, every, fps):
    frames = sorted((p for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg", ".png")), key=natural)
    for p in frames[:: max(1, round(every * fps))]:
        yield int(re.sub(r"\D", "", p.stem) or 0), cv2.imread(str(p))


class Cropper:
    def __init__(self, n2ucore):
        sys.path.insert(0, n2ucore)
        from N2uCore.inference.gazedinoengine import GazeDinoEngine

        self._engine = GazeDinoEngine("unused")    # only the face box and the eye opening: no model is loaded
        self._landmarks = TrackingLandmarks(n2ucore, matrices=True)

    def __call__(self, bgr):
        """(status, crop BGR 224x224, face size, yaw, pitch); status is ok, no_face, edge or closed."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mesh, matrix = self._landmarks.detect(rgb)
        if not mesh.any() or matrix is None:
            return "no_face", None, 0, 0, 0
        height, width = rgb.shape[:2]
        box, atEdge = self._engine._faceBox(mesh, width, height)
        if box is None:
            return "no_face", None, 0, 0, 0
        if atEdge:
            return "edge", None, 0, 0, 0
        if self._engine._eyeOpening(mesh, width, height) < self._engine.BLINK_EAR:
            return "closed", None, 0, 0, 0
        x1, y1, x2, y2 = box
        crop = cv2.resize(bgr[y1:y2, x1:x2], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        return ("ok", crop, max(x2 - x1, y2 - y1) / width, *headPose(matrix))

    def close(self):
        self._landmarks.close()


def exportPerson(cropper, sources, out):
    """sources: (task, name, iterator of (frame number, BGR)). Writes out (an .npz) and returns the counts."""
    counts = dict(sampled=0, no_face=0, edge=0, closed=0)
    jpegs, fields, names, sampled = [], [], [], []
    for task, name, frames in sources:
        names.append(name)
        sampled.append(0)
        for number, bgr in frames:
            counts["sampled"] += 1
            sampled[-1] += 1
            status, crop, size, yaw, pitch = cropper(bgr)
            if status != "ok":
                counts[status] += 1
                continue
            jpegs.append(cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].ravel())
            fields.append((task, number, size, yaw, pitch))
    offsets = np.cumsum([0] + [len(j) for j in jpegs]).astype(np.int64)
    f = np.array(fields, np.float64).reshape(-1, 5)
    temporary = out.with_name(out.stem + ".partial.npz")
    np.savez(temporary, jpeg=np.concatenate(jpegs) if jpegs else np.zeros(0, np.uint8), offsets=offsets,
             task=f[:, 0].astype(np.int16), frame=f[:, 1].astype(np.int32), face_size=f[:, 2].astype(np.float32),
             yaw=f[:, 3].astype(np.float32), pitch=f[:, 4].astype(np.float32),
             task_names=np.array(names), task_sampled=np.array(sampled, np.int32),
             **{k: np.int64(v) for k, v in counts.items()})
    os.replace(temporary, out)
    return counts | {"kept": len(jpegs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n2ucore")
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--people", help="comma separated person folders (default: all)")
    ap.add_argument("--every", type=float, default=2.0, help="seconds between sampled frames")
    ap.add_argument("--frames", help="person name: SOURCE is a folder of frames, not of people")
    ap.add_argument("--fps", type=float, default=15.0, help="frame rate of --frames")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cropper = Cropper(args.n2ucore)
    if args.frames:
        jobs = [(args.frames, lambda: [(0, Path(args.source).name, folderFrames(args.source, args.every, args.fps))])]
    else:
        people = args.people.split(",") if args.people else sorted(p.name for p in Path(args.source).iterdir()
                                                                   if p.is_dir() and not p.name.startswith("."))
        def videos(person):
            files = sorted((v for v in (Path(args.source) / person).iterdir()
                            if v.suffix.lower() == ".mp4" and not v.name.startswith("._")), key=natural)
            return [(i, v.stem, videoFrames(v, args.every)) for i, v in enumerate(files)]
        jobs = [(p, lambda p=p: videos(p)) for p in people]
    for person, sources in jobs:
        target = out / f"{person}.npz"
        if target.exists():
            print(f"{person}: already done", flush=True)
            continue
        start = time.time()
        counts = exportPerson(cropper, sources(), target)
        print(f"{person}: " + " ".join(f"{k} {v}" for k, v in counts.items()) + f"  {time.time() - start:.0f}s", flush=True)
    cropper.close()


if __name__ == "__main__":
    main()
