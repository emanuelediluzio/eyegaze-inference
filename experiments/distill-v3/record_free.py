"""Record unlabeled webcam frames in the head poses where distillation v2 loses: closer, lower, turned, tilted.

    python record_free.py OUT_DIR [camera] [WxH]

No calibration and no labels: the teacher labels the crops during distillation (export_crops.py --frames, then
distill_gaze_v3). A dot moves slowly over the whole screen and the eyes follow it, so the gaze covers the screen in
every head pose. 10 phases of 30 s, about 5 minutes with the pauses. Camera as in the pipeline: 640x480.
Output: <out>/frames/NNNNNN.jpg, <out>/log.jsonl (frame, phase, dot position), <out>/screen.json.
Space continues at each pause, Esc aborts (only at the pauses).
"""
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

out = Path(sys.argv[1]); (out / "frames").mkdir(parents=True, exist_ok=True)
CAMERA = int(sys.argv[2]) if len(sys.argv) > 2 else 0
cam = cv2.VideoCapture(CAMERA, cv2.CAP_AVFOUNDATION); cam.set(3, 640); cam.set(4, 480)
for _ in range(30): cam.grab()
cv2.namedWindow("rec", cv2.WND_PROP_FULLSCREEN); cv2.setWindowProperty("rec", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
for _ in range(40):                     # the fullscreen transition takes a moment on macOS
    cv2.imshow("rec", np.full((10, 10, 3), 247, np.uint8)); cv2.waitKey(100)
    _, _, W, H = cv2.getWindowImageRect("rec")
    if W >= 800: break
if len(sys.argv) > 3: W, H = map(int, sys.argv[3].split("x"))
if W < 800: W, H = 1440, 900
SECONDS = 30
PHASES = [
    ("normal", "Siediti come al solito."),
    ("closer", "AVVICINATI alla camera di circa 15 cm."),
    ("closest", "Avvicinati ANCORA, finche' il viso resta tutto nell'anteprima."),
    ("back", "Torna INDIETRO, schiena allo schienale."),
    ("down", "Scivola GIU' con la testa di circa 10 cm."),
    ("chin_down", "Testa normale ma MENTO IN BASSO, come se leggessi un foglio."),
    ("up", "Raddrizzati e alza un po' la testa (mento in su)."),
    ("turn_left", "GIRA la testa a sinistra (circa 20 gradi), gli occhi seguono il punto."),
    ("turn_right", "GIRA la testa a destra (circa 20 gradi), gli occhi seguono il punto."),
    ("free", "Muovi la testa come vuoi: piega, appoggiati alla mano, avvicinati e allontanati."),
]
(out / "screen.json").write_text(json.dumps({"w": W, "h": H, "camera": CAMERA, "seconds": SECONDS,
                                             "phases": [p[0] for p in PHASES]}))
log = open(out / "log.jsonl", "w"); n = 0


def text(img, s, y, scale=0.9):
    size = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
    cv2.putText(img, s, ((W - size[0]) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (40, 40, 40), 2)


def pause(lines):
    """Camera preview and the instruction until Space (the face must be in view)."""
    while True:
        ok, frame = cam.read()
        img = np.full((H, W, 3), 247, np.uint8)
        if ok:
            prev = cv2.resize(cv2.flip(frame, 1), (320, 240)); img[60:300, (W - 320) // 2:(W + 320) // 2] = prev
        for i, s in enumerate(lines): text(img, s, H // 2 + 40 * i)
        text(img, "Premi SPAZIO quando sei in posizione (Esc annulla)", H // 2 + 40 * len(lines) + 30, 0.7)
        cv2.imshow("rec", img); key = cv2.waitKey(1)
        if key == 27: sys.exit("aborted")
        if key == 32: return


def dot(t, phase):
    """A slow Lissajous path over the whole screen, a different start per phase."""
    a = 0.13 * t + 0.7 * phase
    return (W * (0.5 + 0.45 * math.sin(2 * a)), H * (0.5 + 0.43 * math.sin(3 * a + 0.5)))


for p, (phase, instruction) in enumerate(PHASES):
    pause([f"{p + 1}/{len(PHASES)}  {instruction}", "Poi segui il punto con gli occhi per 30 secondi."])
    start = time.monotonic()
    while (t := time.monotonic() - start) < SECONDS:
        ok, frame = cam.read()
        x, y = dot(t, p)
        img = np.full((H, W, 3), 247, np.uint8)
        cv2.circle(img, (int(x), int(y)), 14, (48, 120, 215), 3); cv2.circle(img, (int(x), int(y)), 4, (20, 20, 20), -1)
        text(img, f"{phase}  {SECONDS - int(t)} s - segui il punto", H - 30, 0.7)
        cv2.imshow("rec", img); cv2.waitKey(1)          # Esc only at the pauses
        if ok:
            cv2.imwrite(str(out / "frames" / f"{n:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            log.write(json.dumps({"i": n, "t": t, "phase": phase, "x": x, "y": y, "mono": time.monotonic()}) + "\n"); n += 1
log.close(); cam.release(); cv2.destroyAllWindows()
print("frames", n, "screen", W, H)
