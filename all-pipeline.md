# EyeGaze — Complete Pipeline

Technical reference for the **GazeDINO** system, covering the model, training and dataset with diagrams and key figures.

---

## Contents

- [Introduction](#introduction)
- [1. Model architecture](#1-model-architecture)
- [2. Dataset — GazeGene](#2-dataset--gazegene)
- [3. Training pipeline](#3-training-pipeline)
- [4. Results](#4-results)

---

## Introduction

The system estimates gaze direction `(yaw, pitch)` from a single RGB face image. It consists of two modules:

- a **DINOv2 ViT-B/14 backbone** (~86 M parameters), pretrained by Meta AI using self-supervision on 142 M images, downloaded and used as a feature extractor
- an **MLP head** (~0.5 M parameters) trained to map DINOv2 features to `(yaw, pitch)`

Training used the **GazeGene** dataset (50 subjects, 9 cameras, ~90 K samples) in two stages: head warm-up with a frozen backbone, followed by full fine-tuning at a very low learning rate. The best model reaches **3.24° mean angular error** on the validation set.

This repository contains **inference code**: the desktop GUI accepts a webcam or video, and the N2uCore integration provides two Iride nodes. Training takes place in the separate [eye-gaze](https://github.com/emanuelediluzio/eye-gaze) repository.

The checkpoint diagram below describes the training format. The minimal inference joblib excludes optimizer state; see the [integration guide](integrations/n2ucore/README.md) for downloads and deployment.

---

## 1. Model architecture

```
                      ┌─────────────────────────────────────┐
                      │     INPUT IMAGE                     │
                      │     face crop RGB                   │
                      │     normalized ImageNet stats       │
                      │     mean = [0.485, 0.456, 0.406]    │
                      │     std  = [0.229, 0.224, 0.225]    │
                      └─────────────────┬───────────────────┘
                                        │
                                shape: (B, 3, 224, 224)
                                        │
                                        ▼
   ╔════════════════════════════════════════════════════════════════════╗
   ║                                                                    ║
   ║                  BACKBONE — DINOv2 ViT-B/14                        ║
   ║                                                                    ║
   ║   ┌────────────────────────────────────────────────────────────┐   ║
   ║   │  Patch Embedding 14×14                                     │   ║
   ║   │  → (16×16) = 256 patch token + 1 CLS token  +  pos.embed   │   ║
   ║   └────────────────────────────────────────────────────────────┘   ║
   ║                                │                                   ║
   ║                                ▼                                   ║
   ║   ┌────────────────────────────────────────────────────────────┐   ║
   ║   │  12 × Transformer Block                                    │   ║
   ║   │     ├─ MHSA (12 heads, dim 64)                             │   ║
   ║   │     ├─ LayerNorm + Residual                                │   ║
   ║   │     ├─ MLP (3072 hidden)                                   │   ║
   ║   │     └─ LayerNorm + Residual                                │   ║
   ║   └────────────────────────────────────────────────────────────┘   ║
   ║                                │                                   ║
   ║          (extract ONLY the CLS token; discard the 256 patch tokens)║
   ║                                │                                   ║
   ║                                ▼                                   ║
   ║                       CLS token: (B, 768)                          ║
   ║                                                                    ║
   ║              86 M parameters · self-supervised pretraining         ║
   ║              (Meta AI, on 142 M images, 0 labels)                  ║
   ╚════════════════════════════════════════════════════════════════════╝
                                        │
                                  (B, 768)
                                        │
                                        ▼
   ╔════════════════════════════════════════════════════════════════════╗
   ║                                                                    ║
   ║                    REGRESSION HEAD (MLP)                           ║
   ║                                                                    ║
   ║       ┌──────────────────────────────────┐                         ║
   ║       │  LayerNorm(768)                  │                         ║
   ║       └─────────────────┬────────────────┘                         ║
   ║                         ▼                                          ║
   ║       ┌──────────────────────────────────┐                         ║
   ║       │  Linear(768 → 512)               │                         ║
   ║       │  GELU                            │                         ║
   ║       │  Dropout(p = 0.30)               │                         ║
   ║       └─────────────────┬────────────────┘                         ║
   ║                         ▼                                          ║
   ║       ┌──────────────────────────────────┐                         ║
   ║       │  Linear(512 → 128)               │                         ║
   ║       │  GELU                            │                         ║
   ║       │  Dropout(p = 0.15)               │                         ║
   ║       └─────────────────┬────────────────┘                         ║
   ║                         ▼                                          ║
   ║       ┌──────────────────────────────────┐                         ║
   ║       │  Linear(128 → 2)                 │                         ║
   ║       └─────────────────┬────────────────┘                         ║
   ║                                                                    ║
   ║          ~ 0.5 M parameters · Xavier uniform init · bias = 0       ║
   ╚════════════════════════════════════════════════════════════════════╝
                                        │
                                  (B, 2)
                                        │
                                        ▼
                           ┌────────────────────────┐
                           │     OUTPUT             │
                           │  (yaw, pitch) radians  │
                           └────────────────────────┘

                       TOTAL: ~ 87 M parameters
```

### Parameter counts

| Component | Parameters |
|------------|----------:|
| Backbone DINOv2 ViT-B/14 | 86 580 480 |
| Head MLP (768→512→128→2) | 461 186 |
| **Total** | **87 041 666** |

### Output

- `yaw` ∈ ℝ — horizontal angle (positive = looking right)
- `pitch` ∈ ℝ — vertical angle (positive = looking up)
- both in **radians** (converted to degrees for display only)

---

## 2. Dataset — GazeGene

**Source:** [HuggingFace `vigil1917/GazeGene`](https://huggingface.co/datasets/vigil1917/GazeGene), ~27 GB. Published under a permissive license.

### Composition

| | |
|---|---|
| Subjects | 56 |
| Cameras per subject | 9 (different viewing angles) |
| Total combinations | 56 × 9 = 504 |
| Frames per camera | up to 200 (configurable) |
| Run 3 (best model) | 50 subjects × 9 × 200 = **90,000 samples** |
| Split train / val | 80 / 20 frame-level (seed 42) |
| Observed yaw range | −134° ↔ +124° |
| Observed pitch range | −87° ↔ +77° |

### Directory layout

```
gazegene_data/GazeGene_FaceCrops/
├── subject1/
│   ├── imgs/
│   │   ├── camera0/   subject1_0000.jpg … subject1_0199.jpg
│   │   ├── camera1/   …
│   │   └── camera8/
│   └── labels/
│       ├── gaze_label_camera0.pkl    ← gaze_C: (N, 3) unit 3D vectors
│       ├── gaze_label_camera1.pkl
│       └── …
├── subject2/
└── …
└── subject56/
```

### Converting 3D labels to angles

The pickle stores `gaze_C` as an `(N, 3)` array of 3D direction vectors. Conversion in `data/gazegene_loader.py`:

```python
# convention: x=right, y=down, z=forward
yaw   = arctan2(x, -z)
pitch = arctan2(-y, sqrt(x² + z²))
```

### Augmentation (training only)

| Transformation | Probability | Notes |
|----------------|------------:|------|
| Horizontal flip + yaw mirror | 0.5 | `label[0] = -label[0]` (critical) |
| ColorJitter | 0.7 | brightness 0.3, contrast 0.3, saturation 0.2, hue 0.05 |
| GaussianBlur | 0.3 | kernel 3, sigma 0.1–1.5 |
| Gaussian noise | 0.3 | σ = 0.02 |

Images are loaded **on demand** from disk using `cv2.imread` (lazy loading) to avoid exhausting RAM.

---

## 3. Training pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              PREPARATION                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  GazeGene HuggingFace (~27 GB)                                          │
│         │                                                               │
│         ▼                                                               │
│  data/download_gazegene.py                                              │
│         │  curl + zip merge + extract                                   │
│         ▼                                                               │
│  gazegene_data/GazeGene_FaceCrops/                                      │
│    └── subjectN/imgs/cameraK/*.jpg                                      │
│    └── subjectN/labels/gaze_label_cameraK.pkl                           │
│         │                                                               │
│         ▼                                                               │
│  data/gazegene_loader.py · load_gazegene()                              │
│    ├─ read .pkl files (3D vectors)                                      │
│    ├─ convert 3D → (yaw, pitch) radians: arctan2(...)                   │
│    ├─ split frame-level 80/20 (seed=42)                                 │
│    └─ return (paths_train, paths_val, y_train, y_val)                   │
│         │                                                               │
│         ▼                                                               │
│  data/gaze_dataset.py · LazyGazeDataset                                 │
│    ├─ read images on demand (cv2.imread)                                │
│    ├─ resize 224 + normalize ImageNet                                   │
│    └─ augmentation (training only): hflip+yaw mirror, jitter, blur, noise│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                              DataLoader (batch_size=32, num_workers=0)
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                               MODEL INIT                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   GazeDINO(backbone='dinov2_vitb14', dropout=0.3, freeze_backbone=True) │
│         │                                                               │
│         ▼                                                               │
│   torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')            │
│         │                                                               │
│         ▼                                                               │
│   Head MLP (init Xavier, bias=0)                                        │
│         │                                                               │
│         ▼                                                               │
│   .to(device)  · device = MPS / CUDA / CPU (auto)                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
╔═════════════════════════════════════════════════════════════════════════╗
║                  PHASE 1 · HEAD WARM-UP · 10 epochs                     ║
╠═════════════════════════════════════════════════════════════════════════╣
║                                                                         ║
║  backbone:  ❄ FROZEN  (unchanged DINOv2 weights)                        ║
║  head:      🟢 trainable  (~ 0.5 M params)                              ║
║                                                                         ║
║  optimizer  AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)        ║
║  scheduler  CosineAnnealingLR(T_max=10, eta_min=1e-5)                   ║
║                                                                         ║
║          LR:   1e-3  ─────────cosine─────────►  1e-5                    ║
║                                                                         ║
║  for each epoch:                                                        ║
║    ┌─────────────────────────────────────────────────────────────┐      ║
║    │ for batch in train_loader:                                  │      ║
║    │   pred = model(imgs)                                        │      ║
║    │   loss = angular_loss(pred, labels)                         │      ║
║    │   loss.backward()                                           │      ║
║    │   clip_grad_norm_(1.0)                                      │      ║
║    │   optimizer.step()                                          │      ║
║    │ # validation → val_err in degrees                            │     ║
║    │ if val_err < best: save_checkpoint(best.pt)                 │      ║
║    └─────────────────────────────────────────────────────────────┘      ║
║                                                                         ║
║  Run 1, end of phase 1: ~62.5° (head still unadapted but stable)        ║
╚═════════════════════════════════════════════════════════════════════════╝
                                    │
                              .unfreeze_backbone()
                                    │
                                    ▼
╔═════════════════════════════════════════════════════════════════════════╗
║              PHASE 2 · FULL FINE-TUNE · 50 epochs                       ║
╠═════════════════════════════════════════════════════════════════════════╣
║                                                                         ║
║  backbone:  🟢 trainable  (86 M params, gradients enabled)              ║
║  head:      🟢 trainable                                                ║
║                                                                         ║
║  optimizer  AdamW(model.parameters(), lr=5e-6, weight_decay=1e-4)       ║
║  scheduler  CosineAnnealingLR(T_max=50, eta_min=5e-8)                   ║
║                                                                         ║
║          LR:   5e-6  ─────────cosine─────────►  5e-8                    ║
║                                                                         ║
║          ↑                                                              ║
║          200× lower than Phase 1 — preserve the pretrained backbone;    ║
║          only a small adjustment is needed to fine-tune DINOv2          ║
║                                                                         ║
║  early stopping: patience=12 epochs                                     ║
║                                                                         ║
║  Run 1 (10 subj): best val 4.44° @ ep 47                                ║
║  Run 2 (50 subj): interrupted at 43/50                                  ║
║  Run 3 (50 subj, --resume from Run 2): best val 3.24° @ ep 49 ✅        ║
║                                                                         ║
╚═════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              LOSS & METRIC                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  angular_loss(pred, target)                                             │
│                                                                         │
│      convert (yaw, pitch) → unit 3D vector                              │
│      v = [sin(yaw)·cos(pitch), -sin(pitch), -cos(yaw)·cos(pitch)]       │
│                                                                         │
│      cos_sim = (v_pred · v_true).clamp(-1+ε, 1-ε)                       │
│      loss    = acos(cos_sim).mean()      ← in radians                   │
│                                                                         │
│      val_err_deg = rad2deg(loss)         ← for logging                  │
│                                                                         │
│  Geometrically correct: measures the angle between 3D vectors           │
│  on the unit sphere, not Euclidean error in the (yaw, pitch) plane.     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                                OUTPUT                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  checkpoints/gaze_YYYYMMDD_HHMMSS/best.pt                               │
│    {                                                                    │
│      "epoch":       49,                                                 │
│      "val_err_deg": 3.24,                                               │
│      "model":       state_dict,                                         │
│      "optimizer":   state_dict                                          │
│    }                                                                    │
│                                                                         │
│  logs/gaze_YYYYMMDD_HHMMSS/training_log.csv                             │
│    phase, epoch, loss, train_err, val_loss, val_err, lr                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Hyperparameter cheat sheet

| | Phase 1 | Phase 2 |
|---|--------:|--------:|
| Epochs | 10 | 50 |
| Backbone | ❄ frozen | 🟢 trainable |
| Trainable params | 0.5 M | 87 M |
| Initial LR | `1e-3` | `5e-6` |
| Final LR | `1e-5` | `5e-8` |
| Optimizer | AdamW | AdamW |
| Weight decay | `1e-4` | `1e-4` |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Gradient clip | `max_norm=1.0` | `max_norm=1.0` |
| Early stopping | patience 12 | patience 12 |
| Batch size | 32 | 32 |
| Loss | angular_loss | angular_loss |
| Augmentation | yes (training only) | yes (training only) |

### Hardware

All runs used **Apple Silicon (MPS backend)**. Full Run 3 duration: approximately five days on M1.

---

## 4. Results

| Run | Date | Subjects | Phase | Best Val Error | Notes |
|-----|------|---------:|-------|---------------:|------|
| Run 1 | 2026-03-14 | 10 | 1 + 2 | **4.44°** @ ep 47 | from scratch |
| Run 2 | 2026-04-11 | 50 | 2 only | ~3.94° | interrupted at 43/50 |
| **Run 3** | **2026-04-16** | **50** | 2 only (resumed from Run 2) | **3.24°** @ ep 49 | **best** |

**Improvement Run 1 → Run 3:** −1.20° (driven by 5× more data and iterative resumed training).

Checkpoints are published in [Releases v1.0-weights](https://github.com/emanuelediluzio/eye-gaze/releases/tag/v1.0-weights):
- `best_run3_3.24deg.pt` — best (Run 3)
- `best_run1_4.44deg.pt` — baseline (Run 1)

---

## References

- DINOv2 — Oquab et al., 2023 — [paper](https://arxiv.org/abs/2304.07193) · [repo](https://github.com/facebookresearch/dinov2) · Apache 2.0
- GazeGene Dataset — [HuggingFace](https://huggingface.co/datasets/vigil1917/GazeGene)
- Vision Transformers (ViT) — Dosovitskiy et al., 2020 — [paper](https://arxiv.org/abs/2010.11929)
- Training repo: [eye-gaze](https://github.com/emanuelediluzio/eye-gaze)
