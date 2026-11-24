# Distillation v3: real faces in the head poses where v2 loses

Status 2026-11-24. The v3 student (ViT-S) reaches the teacher (ViT-B) on MPIIFaceGaze and on the recorded sessions, including the poses where v2 was worse: head closer and looking down.

In N2uCore_dev since 48d5f63: the GazeDirection node runs this student when its device is CPU (`student.joblib`, or `student.mlpackage` on macOS) and the ViT-B when it is GPU. On an M1 CPU the student takes 40 ms a face against 133 ms for the ViT-B joblib. Tested live in Iride on the Mac mini with both devices.

## Why

v2 (`../model-size`) closed the gap to the teacher on real frontal faces, but on the recorded sessions it was worse than the teacher when the head came closer (146 px against 102 px) or looked down (243 px against 180 px). Its real crops came only from MPIIFaceGaze, frontal faces at a laptop.

## Data

Two free-pose webcam sessions recorded on the Mac mini with `record_free.py`, camera 640x480:

- 10 phases of 30 s: normal, closer, closest, back, down, chin_down, up, turn_left, turn_right, free;
- in each phase a dot moves over the screen and the eyes follow it;
- no labels: the teacher labels the face crops during distillation.

| Session | Frames | Use |
|---|---|---|
| `gazefree-20261123-1615-emanuelediluzio` | 8988 | training |
| `gazefree-20261123-1622-joelnovello` | 8987 | held out, model selection only |

Face crops come from `../model-size/export_crops.py --frames ... --stride 1`, with the tracking node's landmarks and Iride's crop. Every frame gave a crop.

## Training

`distill_gaze_v3.py` is `distill_gaze_v2.py` with these changes:

- `--real_dir` and `--real_val_dir` take comma-separated folders;
- the selection metric is the mean of the gaps to the teacher over the validation folders, so MPII p10–p14 and the held-out session weigh the same;
- `last.pt` is saved every epoch, and `--resume` continues an interrupted run in the same folder.

Run on the Mac mini (MPS), started from v2's best checkpoint:

- GazeGene, 50 subjects, 500 batches per epoch;
- real crops for training: MPII p00–p09 (28599) and the training session (8988);
- validation folders: MPII p10–p14 (2977) and the held-out session (8987);
- 30 epochs at most, patience 8, 2 loader workers.

It stopped by itself at epoch 10; the best checkpoint is epoch 2. Each epoch took about an hour.

Gap to the teacher on the validation crops:

| Checkpoint | Mean | MPII p10–p14 | Held-out session |
|---|---|---|---|
| v2 (start) | 2.42° | 2.02° | 2.82° |
| **v3, epoch 2** | **2.26°** | **1.99°** | **2.53°** |
| v3, epoch 10 | 2.44° | 2.09° | 2.79° |

After epoch 2 the gap grows again and the GazeGene validation error rises from 3.40° to 3.67° (teacher 2.22°), the same pattern as v2.

## Results

All with the tracking node's landmarks.

MPIIFaceGaze calibrated error (a line per axis fitted on 16 frames, mean of 20 draws). v2 and v3 were trained on crops of p00–p09, so only p10–p14 are unseen:

| Model | All 15 people | p10–p14 (unseen) | p00–p09 |
|---|---|---|---|
| Teacher ViT-B | 6.31° | 6.90° | 5.99° |
| Student v2 | 6.26° | 6.84° | 5.97° |
| **Student v3** | **6.24°** | **6.81°** | **5.95°** |

Recorded sessions 2 and 3, screen px, target error averaged over the two sessions:

| Model | cv | still | back | left | closer | down |
|---|---|---|---|---|---|---|
| Teacher ViT-B | 91 | 63 | 115 | 136 | 102 | 180 |
| Student v1 | 102 | 89 | 124 | 143 | 128 | 207 |
| Student v2 | 89 | 81 | 118 | 122 | 146 | 243 |
| **Student v3** | **79** | **69** | **119** | **119** | **108** | **196** |
| Student v3, Core ML int8 | 78 | 68 | 118 | 121 | 107 | 197 |

- Compared with v2, v3 is better when closer (146 to 108 px) and looking down (243 to 196 px). On session 3 these go from 176 to 121 px and from 280 to 206 px.
- Against the teacher, v3 is within 6 px when still and closer, within 16 px when looking down, and better on cv and left.
- On MPII it stays at the teacher's level.
- The Core ML int8 package (`../model-size/export_coreml.py`, `student_v3_w8.mlpackage`, 22 MB against 83 MB for the teacher's) keeps these results: 6.25° on MPII, and the session row above.

**Caveats:**

- Sessions 2 and 3 are of the same person as the training session. The sessions themselves were not used, but the face was. The unseen-person check is the held-out session, where the gap to the teacher went from 2.82° to 2.53°.
- There are two people and two sessions in all. More people would make the result more reliable.

## Files

- Mac mini: `~/training_mini/gazedistill/run_distill3.sh`, `log/distill3.log`, checkpoints in `code/checkpoints/distill3_20261124_002445`; crops in `~/n2u_training_data/gazedistill3-crops`; recordings in `~/n2u_training_data/gazefree-*`.
- MacBook: `~/n2u_training_data/gazedistill-20261124`, with `student_v3_best.pt`, `run_eval_v3.sh`, `eval_v3.txt` and `mpii_student_v3.npz`; the Core ML packages in `coreml/`, checked by `run_eval_v3_w8.sh` (`eval_v3_w8.txt`).
- Mac mini live test: `student_v3_w8.mlpackage` is copied to `N2uCore/res/models/gazedino/model.mlpackage`, the file the gaze node loads on macOS; the teacher's package is kept as `teacher_vitb_w8.mlpackage`.

```bash
python record_free.py OUT_DIR [camera]
python ../model-size/export_crops.py N2UCORE_DIR model.joblib crops/SESSION --frames SESSION/frames --stride 1
python distill_gaze_v3.py --teacher best_run3_3.24deg.pt --init distill_v2/best.pt \
    --real_dir crops/mpii_p00-p09,crops/SESSION_TRAIN --real_val_dir crops/mpii_p10-p14,crops/SESSION_HELDOUT \
    --epochs 30 --patience 8 --workers 2              # from the eye-gaze training code
python distill_gaze_v3.py ... --resume checkpoints/distill3_RUN/last.pt
```
