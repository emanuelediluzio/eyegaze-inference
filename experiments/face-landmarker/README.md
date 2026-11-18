# MediaPipe FaceLandmarker instead of FaceMesh in the gaze engine

**Status (2026-11-18): adopted in the integration.**

The Jetson runs mediapipe 1.2.3, which no longer has the legacy FaceMesh, so `GazeDinoEngine` could not start there. The engine now uses FaceLandmarker, set up as in N2uCore's tracking nodes:

- the model file `models/face_landmarker.task`;
- image mode;
- one face per frame.

Its results are the same as with FaceMesh. It runs with mediapipe 0.10.21 and with 1.2.3.

## Results

Two recorded sessions (`../head-adaptation`) and MPIIFaceGaze (`../model-size/eval_mpii.py`), with GazeDINO ViT-B in PyTorch.

**Engine step** (`bench_engine.py`, M1 MacBook under load):

| Landmarks | Step, median | Step, p90 | Frames with a face |
|---|---|---|---|
| FaceMesh | 81–86 ms | 93–137 ms | 100% |
| FaceLandmarker | 84–86 ms | 98–101 ms | 100% |

With 1 or with 2 faces the time is the same.

**MPIIFaceGaze, 15 people:**

| Landmarks | Frames with a face | Calibrated error |
|---|---|---|
| FaceMesh | 4279 | 6.35° |
| FaceLandmarker | 4489 | 6.32° |

**Recorded sessions:** mean of the two sessions, target / frame px.

| Landmarks | cv | still | back | left | closer | down |
|---|---|---|---|---|---|---|
| FaceMesh | 77 | 71/77 | 114/121 | 150/153 | 97/102 | 194/196 |
| FaceLandmarker | 93 | 65/72 | 112/119 | 134/145 | 104/108 | 186/190 |
| FaceLandmarker + Core ML int8 | 94 | 64/72 | 114/120 | 140/150 | 106/110 | 188/192 |

On the check targets, FaceLandmarker is better in 4 of 5 positions. The calibration's own leave-one-out estimate is higher (session 3: 121 against 97 px) and is to be confirmed live.

**Eye opening (eye aspect ratio):**

- FaceMesh: median 0.32–0.33, 5th percentile 0.27–0.28.
- FaceLandmarker: median 0.34, 5th percentile 0.30–0.31.

The blink threshold of 0.15 stays well below both, so it is unchanged.

**mediapipe 1.2.3:** the gaze tests pass, and the engine finds the face in 100% of the frames with the same eye-opening figures. The FaceMesh engine does not load.

## Later

The tracking node already runs FaceLandmarker on the same camera. Once its `Landmarks` channel publishes the 478 raw normalized landmarks (irises kept, no metric transform) with the frame's packID, the gaze node can use them and stop running MediaPipe itself. This is to be done with the tracking node's maintainers.
