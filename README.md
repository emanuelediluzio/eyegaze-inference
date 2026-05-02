# EyeGaze Inference

Real-time eye gaze estimation using **GazeDINO** (DINOv2 ViT-B/14 + MLP head).

Inference-only package — no training code, no dataset dependencies. Just plug in a checkpoint and run.

## Features

| Feature | Description |
|---------|-------------|
| **Gaze Estimation** | Predicts yaw + pitch from face crops using GazeDINO (3.24 mean angular error) |
| **9-Point Calibration** | Maps gaze angles to screen coordinates via polynomial fit |
| **Distance Alarm** | Warns if you're too close or too far from the screen |
| **Pupillometry** | Relative iris size as a proxy for pupil dilation |
| **GUI** | CustomTkinter dark-mode interface with live metrics |
| **Video Mode** | Analyze video files with playback controls |

## Model

| | |
|---|---|
| **Architecture** | DINOv2 ViT-B/14 + MLP regression head |
| **Parameters** | 87M |
| **Best Val Error** | 3.24 mean angular error |
| **Output** | (yaw, pitch) in radians |
| **Training** | [eye-gaze](https://github.com/emanuelediluzio/eye-gaze) repo |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your checkpoint
mkdir checkpoints
cp /path/to/best.pt checkpoints/

# 3. Run (live camera)
python gui.py

# 4. Run on a video file
python gui.py --video path/to/video.mp4
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | auto | Path to `.pt` checkpoint |
| `--camera` | 0 | Camera index |
| `--video` | none | Path to video file |
| `--no-loop` | off | Don't loop video |

## Controls

The GUI provides toggle switches for all overlays (gaze arrows, distance, pupillometry), calibrate/reset buttons, and video playback controls with a seek slider.

## Project Structure

```
eyegaze-inference/
  models/
    gaze_model.py      # GazeDINO architecture (DINOv2 + MLP head)
  infer_gaze.py        # Core: model, FaceMesh analyzer, EMA, drawing helpers
  gui.py               # Main entry point — CustomTkinter GUI
  calibration.py       # 9-point gaze calibration
  requirements.txt
```

## Calibration

Press `C` during inference or use `--calibrate`:

1. Full-screen window shows 9 dots sequentially
2. Keep your head still, follow each dot with your eyes
3. Samples collected for 2.5s per dot
4. Polynomial fitted: (yaw, pitch) -> (screen_x, screen_y)
5. Saved to `calibration.pkl` for reuse

## Requirements

- Python 3.10+
- Apple Silicon (MPS), CUDA, or CPU
- Webcam (for live inference)

## License

MIT
