# eyegaze-inference — GazeDINO Real-Time Inference

Real-time gaze estimation powered by **GazeDINO** (DINOv2 ViT-B/14 + MLP head).
No training code, no dataset dependency. Drop in the checkpoint and run.

Best model: **3.24° mean angular error**.

---

## Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [GUI](#gui)
- [Inference Pipeline](#inference-pipeline)
- [9-Point Calibration](#9-point-calibration)
- [Headless Test Script](#headless-test-script)
- [CLI Reference](#cli-reference)
- [Live Metrics](#live-metrics)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [License](#license)

---

## Architecture

```
Input: (B, 3, 224, 224)  <- face crop, ImageNet-normalised
          |
  DINOv2 ViT-B/14  ->  CLS token  ->  [768 dim]
  (86M params, self-supervised pre-training on 142M images)
          |
  LayerNorm(768)
  -> Linear(768 -> 512) -> GELU -> Dropout(0.30)
  -> Linear(512 -> 128) -> GELU -> Dropout(0.15)
  -> Linear(128 -> 2)
          |
  Output: (yaw, pitch) in radians
```

The model is trained with **angular loss** (mean angular error between 3D gaze vectors) on the GazeGene dataset (56 subjects, 9 cameras, ~90K samples). Training repo: [eye-gaze](https://github.com/emanuelediluzio/eye-gaze).

Backbone options (via `torch.hub`, `facebookresearch/dinov2`):

| Backbone | Params | Embed dim |
|----------|--------|-----------|
| `dinov2_vits14` | 21M | 384 |
| `dinov2_vitb14` | 86M | 768 (default) |
| `dinov2_vitl14` | 307M | 1024 |
| `dinov2_vitg14` | 1.1B | 1536 |

---

## Installation

### With uv (recommended)

```bash
git clone https://github.com/emanuelediluzio/eyegaze-inference.git
cd eyegaze-inference
uv sync --python python3.10
```

> **Note:** MediaPipe currently supports Python 3.10-3.11. Make sure you have a compatible version installed.

### With pip

```bash
git clone https://github.com/emanuelediluzio/eyegaze-inference.git
cd eyegaze-inference
pip install -r requirements.txt
```

### Dev dependencies (optional)

```bash
# with uv
uv sync --extra dev

# with pip
pip install pytest black flake8
```

---

## Quick Start

### 1. Download the checkpoint

Download the best checkpoint from [GitHub Releases v1.0-weights](https://github.com/emanuelediluzio/eye-gaze/releases/tag/v1.0-weights):

```bash
mkdir -p checkpoints
# Rename the downloaded file
mv best_run3_3.24deg.pt checkpoints/best.pt
```

### 2. Run the GUI (webcam)

```bash
python gui.py
```

### 3. Run on a video file

```bash
python gui.py --video path/to/video.mp4
```

> **macOS:** If the camera doesn't open, go to System Settings -> Privacy & Security -> Camera and authorise your Terminal app.

---

## GUI

Dark-mode interface built with CustomTkinter.

### Sidebar sections

| Section | Content |
|---------|---------|
| **MODEL** | Validation error of the loaded checkpoint |
| **GAZE** | Live yaw/pitch in degrees, screen coordinates (if calibrated) |
| **METRICS** | Distance (cm) with colour-coded status, iris ratio |
| **CALIBRATION** | Calibrate / Load / Clear buttons, status indicator |
| **OVERLAY** | Toggle switches: face box, gaze arrows, iris dots |
| **SOURCE** | Open Video / Camera buttons |
| **VIDEO** | Play/Pause, seek slider, timestamp (video mode only) |

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit |
| `C` | Start 9-point calibration |
| `Space` | Play/Pause (video mode) |

### Multi-face support

The GUI detects and tracks up to 5 faces simultaneously. Each face gets its own EMA smoother for stable gaze output. The sidebar displays metrics for the primary (first) face.

---

## Inference Pipeline

```
Frame (BGR from webcam or video)
      |
      v
MediaPipe FaceMesh (468 landmarks + iris refinement)
      |
      |-- Face bbox -> crop -> resize 224x224 -> GazeDINO -> (yaw, pitch) raw
      |         |
      |         +-- EMA smoothing (alpha=0.3) -> stable (yaw, pitch)
      |
      |-- Iris landmarks (468-477) -> pupillometry
      |         iris_ratio = iris_diameter_px / eye_width_px
      |
      |-- Cheekbone landmarks (234, 454) -> distance estimation
      |         d = (real_face_width * focal_length) / face_px_width
      |         pinhole camera model, comfort zone: 40-90 cm
      |
      |-- Eye landmarks -> EAR (Eye Aspect Ratio) -> blink detection
      |         threshold: 0.21
      |
      +-- 6-point solvePnP -> head pose (yaw, pitch, roll)
                nose tip, chin, eye corners, mouth corners

Without calibration: gaze arrow overlay on video
With calibration:    screen (x, y) pixel coordinates
```

### EMA Smoothing

All continuous values (gaze angles, distance, iris ratios) are smoothed with Exponential Moving Averages to reduce jitter:

| Signal | Alpha |
|--------|-------|
| Gaze (yaw, pitch) | 0.3 |
| Distance | 0.1 |
| Iris ratio | 0.2 |

---

## 9-Point Calibration

Maps raw `(yaw, pitch)` gaze angles to `(screen_x, screen_y)` pixel coordinates using a 2nd-degree polynomial fit.

### How it works

1. A fullscreen window shows 9 calibration points in a 3x3 grid
2. Follow each white dot with your eyes (2.5s per point), keep your head still
3. After the initial 40% of dwell time, the system collects gaze samples
4. Polynomial fit with `PolynomialFeatures(degree=2)` + least-squares regression
5. Requires at least 6 valid points (points with no face detection are skipped)
6. Saved to `calibration.pkl` for reuse across sessions

### Polynomial features

For each `(yaw, pitch)` pair, the feature vector is:

```
[1, yaw, pitch, yaw^2, yaw*pitch, pitch^2]
```

Two separate fits: one for screen X, one for screen Y.

### From the GUI

- Click **Calibrate** in the sidebar (or press `C`)
- Click **Load** to load a previously saved `calibration.pkl`
- Click **Clear** to remove the current calibration

### Standalone calibration

```bash
python calibration.py --model checkpoints/best.pt [--camera 0] [--dwell 2.5] [--out calibration.pkl]
```

### Tips for accurate calibration

- Uniform lighting, avoid strong backlighting
- Both eyes fully visible to the camera
- Keep your head completely still during calibration
- Sit at a comfortable distance (40-90 cm)
- The camera should be as close to eye level as possible

---

## Headless Test Script

Run inference on a video file without the GUI, producing an annotated output video:

```bash
python test_inference.py [model_path] [video_path] [output_path] [max_frames]
```

**Defaults:**
- `model_path`: `checkpoints/best.pt`
- `video_path`: `test_muschio.mp4`
- `output_path`: `output_test.mp4`
- `max_frames`: `150` (~5 seconds)

**Example:**

```bash
python test_inference.py checkpoints/best.pt my_video.mp4 result.mp4 300
```

The script prints progress every 30 frames with face detection rate and processing speed.

---

## CLI Reference

### gui.py

```bash
python gui.py [--model PATH] [--camera INT] [--video PATH]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `checkpoints/best*.pt` (auto-detected) | Path to the `.pt` checkpoint |
| `--camera` | `0` | Webcam index |
| `--video` | — | Path to a video file (overrides camera) |

If `--model` is not provided, the GUI automatically searches for `checkpoints/**/best*.pt`.

### calibration.py

```bash
python calibration.py --model PATH [--camera INT] [--dwell FLOAT] [--out PATH]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Path to the `.pt` checkpoint |
| `--camera` | `0` | Webcam index |
| `--dwell` | `2.5` | Seconds per calibration point |
| `--out` | `calibration.pkl` | Output calibration file |

---

## Live Metrics

| Metric | How it's computed |
|--------|-------------------|
| **Yaw / Pitch** | Direct model output converted to degrees. Yaw = horizontal, Pitch = vertical |
| **Distance (cm)** | `d = (14.0 cm * 554 px) / face_width_px` — pinhole camera model. Green (40-90 cm), yellow (>90 cm), red (<40 cm) |
| **Iris ratio** | `iris_diameter_px / eye_width_px` — proxy for pupil dilation. Smoothed with EMA (alpha=0.2) |
| **EAR** | Eye Aspect Ratio — average of vertical/horizontal eye landmark distances. Blink threshold: 0.21 |
| **Head pose** | solvePnP with 6 facial landmarks -> Euler angles (yaw, pitch, roll) in degrees |
| **Screen XY** | Only after calibration: polynomial mapping `(yaw, pitch)` -> pixel coordinates |

---

## Project Structure

```
eyegaze-inference/
  models/
    gaze_model.py         # GazeDINO: DINOv2 backbone + MLP head (768->512->128->2)
  infer_gaze.py           # Core: load_model, predict, FaceMeshAnalyzer, EMA,
                          #   BlinkTracker, drawing helpers, head pose
  gui.py                  # Entry point: CustomTkinter GUI with sidebar controls
  calibration.py          # 9-point calibration: GazeCalibrator, polynomial fit
  test_inference.py       # Headless video inference test script
  checkpoints/
    best.pt               # Checkpoint (download manually from Releases)
  pyproject.toml          # Project metadata and dependencies (uv/pip)
  uv.lock                 # Locked dependencies for reproducible installs
  requirements.txt        # Legacy pip requirements
```

---

## Requirements

- **Python** 3.10 or 3.11 (MediaPipe constraint)
- **Hardware:** Apple Silicon (MPS), NVIDIA GPU (CUDA), or CPU
- **Webcam** for live inference
- **OS:** macOS, Linux, Windows

### Core dependencies

| Package | Purpose |
|---------|---------|
| `torch` + `torchvision` | DINOv2 backbone, inference |
| `mediapipe` | Face mesh (468 landmarks + iris) |
| `opencv-python` | Video capture, drawing, image processing |
| `customtkinter` | Dark-mode GUI |
| `numpy` | Numerical operations |
| `Pillow` | Image conversion for Tkinter display |
| `scikit-learn` | Polynomial features for calibration |

---

## Troubleshooting

### Camera not authorised (macOS)

System Settings -> Privacy & Security -> Camera -> enable your Terminal / IDE.

### Size mismatch when loading checkpoint

Make sure you're using the checkpoint from [Releases v1.0-weights](https://github.com/emanuelediluzio/eye-gaze/releases/tag/v1.0-weights). The expected architecture is `768 -> 512 -> 128 -> 2`.

### MediaPipe not installing

MediaPipe only has wheels for Python 3.10-3.11. If you're on a newer Python version, use:

```bash
uv sync --python python3.10
```

### Calibration is inaccurate

- Ensure uniform lighting with no strong shadows
- Both eyes must be clearly visible throughout
- Keep your head completely still
- Sit at 40-90 cm from the camera
- Try re-calibrating if you change seating position

### xFormers warning

Cosmetic only — does not affect performance or accuracy. Can be safely ignored.

### Low FPS

- Use MPS (Apple Silicon) or CUDA for GPU acceleration
- Reduce `max_faces` if tracking multiple faces
- Close other applications using the camera

### No face detected

- Check lighting conditions
- Make sure the camera is not obstructed
- Face should be at least partially frontal
- Minimum detection confidence is 0.5

---

## References

- Training repo: [eye-gaze](https://github.com/emanuelediluzio/eye-gaze)
- [DINOv2](https://github.com/facebookresearch/dinov2) — Meta AI, Apache 2.0
- [MediaPipe](https://github.com/google/mediapipe) — Google, Apache 2.0
- [GazeGene Dataset](https://huggingface.co/datasets/vigil1917/GazeGene)

---

## License

MIT
