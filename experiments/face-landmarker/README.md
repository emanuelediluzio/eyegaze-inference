# MediaPipe FaceLandmarker instead of FaceMesh in the gaze engine

**Status (2026-11-19): adopted in the integration; since N2uCore_dev a5c5e98 the landmarks come from the tracking node.**

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
| FaceLandmarker of the tracking node (`Landmarks` channel) | 4456 | 6.31° |

**Recorded sessions:** mean of the two sessions, target / frame px.

| Landmarks | cv | still | back | left | closer | down |
|---|---|---|---|---|---|---|
| FaceMesh | 77 | 71/77 | 114/121 | 150/153 | 97/102 | 194/196 |
| FaceLandmarker | 93 | 65/72 | 112/119 | 134/145 | 104/108 | 186/190 |
| FaceLandmarker + Core ML int8 | 94 | 64/72 | 114/120 | 140/150 | 106/110 | 188/192 |
| FaceLandmarker of the tracking node | 91 | 63/70 | 115/121 | 136/147 | 102/105 | 180/183 |

On the check targets, FaceLandmarker is better in 4 of 5 positions. The calibration's own leave-one-out estimate is higher (session 3: 121 against 97 px) and is to be confirmed live.

**Eye opening (eye aspect ratio):**

- FaceMesh: median 0.32–0.33, 5th percentile 0.27–0.28.
- FaceLandmarker: median 0.34, 5th percentile 0.30–0.31.

The blink threshold of 0.15 stays well below both, so it is unchanged.

**mediapipe 1.2.3:** the gaze tests pass, and the engine finds the face in 100% of the frames with the same eye-opening figures. The FaceMesh engine does not load.

## Landmarks from the tracking node (N2uCore_dev a5c5e98)

The tracking node already runs FaceLandmarker on the same camera. Its `Landmarks` channel now publishes the 478 raw normalized landmarks (irises kept, no metric transform, all zeros without a face) with the frame's packID. The gaze node pairs them with the frame and no longer runs MediaPipe itself, so MediaPipe runs once per frame.

The scripts that use the engine without a tracking node get the same points from `../tracking_landmarks.py`: FaceLandmarker set up as the tracking node, with its default confidences (detection 0.7, tracking 0.5, against 0.5 / 0.5 in the engine before). The rows "of the tracking node" above were measured this way, with the float32 joblib in PyTorch.

- **Accuracy:** the same. MPIIFaceGaze calibrated 6.31° against 6.32°, with 33 fewer frames with a face because of the higher detection confidence. On the recorded sessions the error is the same or slightly lower.
- **Time** (`bench_engine.py`, M1 MacBook under load): FaceLandmarker takes 16–18 ms a frame (median) in the tracking node, which runs anyway. In the gaze node, crop and model take 69–80 ms with PyTorch.
- The gaze node no longer works without a tracking node on the same camera.
