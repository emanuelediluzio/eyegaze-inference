# Model size and speed: Core ML, weight quantization, distilled student

**Status (2026-11-18):**

- **Adopted on macOS:** GazeDINO ViT-B/14 as Core ML with int8 weights. It weighs 87 MB instead of 324 MB, its accuracy is the same, and the model is about 10× faster.
- **Not adopted:** 4-bit palettization, which loses accuracy.
- **Not adopted:** the distilled ViT-S student, which loses on real faces. Distillation v2 is ready (`distill_gaze_v2.py`).

## Data

- **MPIIFaceGaze:** real laptop webcams, 15 people. It is the public dataset as released, at <https://perceptualui.org/research/datasets/MPIIFaceGaze/>.
  - 300 frames per person, each analysed alone through N2uCore's `GazeDinoEngine` (`eval_mpii.py`).
  - The label is the direction from the face centre to the gaze target, turned into GazeGene's (yaw, pitch).
  - "Calibrated" error: a straight line per axis is fitted on 16 frames of the same person, as the screen calibration does with its 16 targets, and the error is measured on the other frames.
- **Recorded sessions:** the two calibration sessions of `../head-adaptation`, in screen px, each model with its own `ScreenCalibration`.

## Results

Times are measured on an M1 MacBook, model alone, batch 1. Core ML times use all compute units; the Neural Engine does most of the work.

| Model | Size | Model time | MPII raw | MPII calibrated | Sessions: cv / still / down px |
|---|---|---|---|---|---|
| ViT-B, PyTorch (today) | 324 MB | 105 ms (MPS fp16) | 8.40° | 6.35° | 77 / 71 / 194 |
| ViT-B, Core ML fp16 | 172 MB | 11 ms | 8.43° | 6.36° | 78 / 72 / 194 |
| **ViT-B, Core ML int8 weights** | **87 MB** | **10 ms** | **8.41°** | **6.37°** | **78 / 72 / 193** |
| ViT-B, Core ML 6-bit palette | 65 MB | 10 ms | 8.45° | 6.36° | 78 / 71 / 205 |
| ViT-B, Core ML 4-bit palette | 44 MB | 10 ms | 9.86° | 7.61° | 107 / 107 / 227 |
| ViT-B, PyTorch int8 joblib (`export_eyegaze.py --int8`) | 81 MB | 105 ms (as float) | 8.30° | 6.33° | 93 / 65 / 186 |
| ViT-S student, PyTorch | 89 MB | 39 ms | 8.59° | 6.64° | 93 / 89 / 207 |
| ViT-S student, Core ML int8 | 22 MB | 4 ms | 8.58° | 6.64° | 93 / 88 / 206 |

The int8 joblib row was measured with FaceLandmarker (`../face-landmarker`). The float32 joblib with FaceLandmarker gives 8.32° / 6.32° on MPII and 93 / 65 / 186 px on the sessions, so int8 changes nothing there either. Frame by frame, the int8 angles are 0.25° from the float32 ones on average (max 0.6°).

On an M1 Mac mini, the whole engine step (face landmarks, crop and model) takes 19.7 ms with the int8 package, against 68.3 ms with PyTorch (`../face-landmarker/bench_engine.py`).

**Integration.** On macOS the gaze node loads `models/gazedino/model.mlpackage` when the file and coremltools are present (`N2uCore/dyn/eyegaze.py`, `GazeDinoEngine._loadCoreML`). Elsewhere it loads the PyTorch joblib, now int8 (81 MB): same quantization, rebuilt in float32 at load, so smaller but not faster.

**Distilled student.** On the 6 GazeGene subjects held out of training, the student equals the teacher: 4.57° against 4.58°. On real faces it loses about 0.3° (MPII) and about 20% in screen px. Distillation v2 targets that gap:

- the v1 student is trained further on real face crops cut by Iride (`export_crops.py`, MPIIFaceGaze people p00–p09), labelled by the teacher;
- patch-token feature matching;
- a longer schedule at a higher learning rate;
- an EMA of the weights;
- the best checkpoint is chosen on the gap to the teacher for p10–p14.

## Scripts

```bash
# Core ML packages (fp16, w8, p6, p4) of the teacher, or of a student
python export_coreml.py N2UCORE_DIR gazedino_inference.joblib OUT_DIR
python export_coreml.py N2UCORE_DIR x OUT_DIR --student student.pt --name student
python bench_coreml.py OUT_DIR                     # time per compute unit
python eval_mpii.py N2UCORE_DIR gazedino_inference.joblib MPII_DIR [--student CKPT | --coreml PKG]
python ../head-adaptation/extract_session.py N2UCORE_DIR CKPT SESSION_DIR --coreml PKG --out NAME.npz
# distillation v2
python export_crops.py N2UCORE_DIR CKPT crops/mpii_p00-p09 --mpii MPII_DIR --subjects p00-p09
python distill_gaze_v2.py --teacher best_run3_3.24deg.pt --init distill_v1/best.pt \
    --real_dir crops/mpii_p00-p09 --real_val_dir crops/mpii_p10-p14     # from the eye-gaze training code
```

Outputs are in `~/n2u_training_data/gazemodelsize-20261118`:

- `bench_coreml_macbook.txt`;
- `evals.txt`, the full MPII and session results;
- the `.mlpackage` files.
