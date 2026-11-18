# Head adaptation and distilled student: tested, not adopted

**Status (2026-11-18): neither is adopted.** Iride keeps GazeDINO (DINOv2 ViT-B/14) with the screen calibration of `ScreenCalibration` (`N2uCore/inference/screengaze.py`), as in N2uCore_dev d85f715.

- **Per-user head adaptation:** it does not beat the screen calibration on real sessions.
- **Distilled student (ViT-S/14):** it is about 2× faster but 20–25% less accurate on the screen.

## Data

- **GazeGene:** 6 subjects held out of training, 54 sessions (subject × camera), 200 frames each.
- **Real sessions:** two calibration sessions recorded on 2026-11-07 (`record.py`). Details:
  - one person, a webcam at 640 px wide, a MacBook at 1440×900 and an external screen at 1920×1080;
  - each session has 16 calibration targets, then 9 check targets in five phases: `still` (as at calibration), `back`, `left`, `closer` and `down` (head moved).
  - The check targets are never used in the fit.
  - Every second frame goes through N2uCore's `GazeDinoEngine` (about 15 FPS, as Iride live).

Errors are in screen px. "target" is the error of the median point of each target; "frame" is the mean error of the single frames, counted from 0.7 s when the eyes have settled.

## 1. Head adaptation

**Question:** does a correction learnt from GazeDINO's own features on the 16 calibration targets help beyond the screen calibration?

This is equivalent to refitting the last head layer, or a ridge on the DINOv2 CLS token. The ridge strength is chosen by leave-one-target-out.

**GazeGene** (`extract_features.py`, `adapt.py`): mean angular error on the frames not used for calibration (`results/adapt_gazegene.txt`).

| Method | Error (deg) |
|---|---|
| raw | 4.61 |
| affine (what the screen calibration does) | 4.04 |
| last layer | 4.03 |
| CLS ridge | 3.92 |

**Real sessions** (`extract_session.py`, `adapt_sessions.py`): mean of the two sessions, target / frame px (`results/adapt_sessions.txt`).

| Method | still | back | left | closer | down |
|---|---|---|---|---|---|
| poly (Iride today) | 72/77 | 114/121 | 150/153 | 97/102 | 194/196 |
| poly + last layer | 72/77 | 114/121 | 150/153 | 97/102 | 194/196 |
| poly + CLS | 72/78 | 113/120 | 148/151 | 98/102 | 194/196 |
| poly + last layer on frames | 84/88 | 106/114 | 159/161 | 115/117 | 209/210 |
| last layer straight to screen | 109/114 | 134/143 | 168/170 | 122/125 | 263/262 |

On the real sessions the leave-one-target-out error falls monotonically with the ridge strength. The chosen correction is therefore always about zero: 16 targets are not enough to learn anything from 128 or 768 features that the polynomial does not already capture.

## 2. Distilled student

**Training** (`distill_gaze.py`, a copy of the script in the eye-gaze training code). It runs there, next to `train_gaze.py`, at eye-gaze 619cfaf.

- Teacher: GazeDINO ViT-B/14. Student: DINOv2 ViT-S/14 with the same head.
- Training data: GazeGene, the same 50 subjects and split as the teacher.
- Loss:
  - 0.5 × angular loss to the labels;
  - 0.5 × angular loss to the teacher's angles;
  - 1 − cosine between the projected student CLS token and the teacher CLS token.
- The teacher runs online on the same augmented batch.
- Schedule:
  - phase 1: 10 epochs with the backbone frozen, lr 1e-3;
  - phase 2: 50 epochs unfrozen, lr 5e-6 with a cosine schedule.
- Run `distill_20261109_100553` on the Mac mini (`results/training_log.csv`, `results/distill_results.json`).

**Result:** 3.29° on the validation split, against 2.23° for the teacher.

The curve was still falling when the cosine schedule ended: 3.39° at epoch 40, 3.30° at epoch 50, best 3.29° at epoch 47. The CLS cosine had saturated at 0.998 from epoch 20.

**Screen accuracy** (`compare_models.py`): each model gets its own calibration on the same frames. Mean of the two sessions, target / frame px (`results/compare_screen.txt`).

| Model | calibration cv | still | back | left | closer | down |
|---|---|---|---|---|---|---|
| GazeDINO ViT-B | 77 | 71/77 | 114/121 | 150/153 | 97/102 | 194/196 |
| student ViT-S | 93 | 89/97 | 111/120 | 149/158 | 125/133 | 207/210 |

Frame by frame, the student's angles are about 2° from the teacher's (p90 4–5°).

**Speed** (`bench_models.py`): M1 MacBook, busy desktop, so the absolute times are pessimistic and the ratio is what counts (`results/bench_macbook.txt`). Model times are for batch 1.

| Model | Params | Model, MPS fp16 | Model, CPU fp32 | Engine step (FaceMesh + crop + model) |
|---|---|---|---|---|
| GazeDINO ViT-B | 87.0 M | 154 ms | 245 ms | 172 ms (5.8 FPS) |
| student ViT-S | 22.3 M | 71 ms | 128 ms | 86 ms (11.7 FPS) |

**Conclusion:** keep GazeDINO as the default. The student is only an option for machines where GazeDINO is too slow (CPU only, old hardware).

## Reproduce

```bash
# real sessions: features of each model, then the comparisons
python extract_session.py N2UCORE_DIR GAZEDINO.pt SESSION_DIR
python extract_session.py N2UCORE_DIR GAZEDINO.pt SESSION_DIR --student student.pt --out student.npz
python adapt_sessions.py N2UCORE_DIR SESSION_DIR...
python compare_models.py N2UCORE_DIR features.npz,student.npz SESSION_DIR...
python bench_models.py N2UCORE_DIR GAZEDINO.pt student.pt SESSION_DIR

# GazeGene held-out subjects
python extract_features.py DATASET_DIR GAZEDINO.pt features.npz
python adapt.py features.npz
```

The recordings and the student checkpoint are not in the repo (`~/n2u_training_data/gazecal-20261107-frames`, `~/n2u_training_data/gazedistill-20261117`).
