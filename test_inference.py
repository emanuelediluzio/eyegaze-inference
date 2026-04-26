"""Quick headless test: run inference on a video, save annotated output."""
from __future__ import annotations
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from infer_gaze import load_model, predict, FaceMeshAnalyzer, EMA, _get_device
from infer_gaze import _draw_eye_arrows, _draw_gaze_arrow, _draw_distance, _draw_pupil
import math


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best.pt"
    video_path = sys.argv[2] if len(sys.argv) > 2 else "test_muschio.mp4"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "output_test.mp4"
    max_frames = int(sys.argv[4]) if len(sys.argv) > 4 else 150  # ~5s

    device = _get_device()
    print(f"Device: {device}")
    model = load_model(model_path, device)
    mesh = FaceMeshAnalyzer(max_faces=5)
    ema = EMA(alpha=0.3)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Input: {video_path} ({w}x{h}, {fps:.0f}fps)")

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    frame_count = 0
    face_count = 0
    t0 = time.time()

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = mesh.process(frame_rgb)
        info = faces[0] if faces else {}

        face_bbox = info.get("face_bbox")
        left_eye = info.get("left_eye")
        right_eye = info.get("right_eye")

        if face_bbox is not None:
            face_count += 1
            x, y, bw, bh = face_bbox
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)

            if x2 > x1 and y2 > y1:
                face_bgr = frame[y1:y2, x1:x2]
                raw_yaw, raw_pitch = predict(model, face_bgr, device)
                smoothed = ema.update(np.array([raw_yaw, raw_pitch]))
                yaw, pitch = float(smoothed[0]), float(smoothed[1])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

                if left_eye or right_eye:
                    _draw_eye_arrows(frame, left_eye, right_eye, yaw, pitch, 120)
                else:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    _draw_gaze_arrow(frame, cx, cy, yaw, pitch, 120)

                yaw_d = math.degrees(yaw)
                pitch_d = math.degrees(pitch)
                cv2.putText(frame, f"Yaw {yaw_d:+.1f}  Pitch {pitch_d:+.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 230, 60), 2)
        else:
            cv2.putText(frame, "No face", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 220), 2)

        _draw_distance(frame, info.get("distance", {"cm": None, "label": ""}))
        _draw_pupil(frame, info.get("pupil"))

        writer.write(frame)
        frame_count += 1

        if frame_count % 30 == 0:
            elapsed = time.time() - t0
            print(f"  Frame {frame_count}/{max_frames}  "
                  f"faces_detected={face_count}/{frame_count}  "
                  f"speed={frame_count/elapsed:.1f} fps")

    elapsed = time.time() - t0
    cap.release()
    writer.release()

    print(f"\nDone! {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} fps)")
    print(f"Face detection rate: {face_count}/{frame_count} ({100*face_count/max(frame_count,1):.0f}%)")
    print(f"Output saved: {out_path}")


if __name__ == "__main__":
    main()
