# Rule T1: a teacher fine-tuned on real labelled faces, and when it beats the current teacher

Written 2026-11-26, committed before any label of a real dataset is used for training. After the commit this file does not change. A change is a new rule, committed before the work it governs.

T1 comes after R1 and does not touch it: R1 is distilled from the current teacher and compared with v3 under `RULE-R1.md`.

## Models

- **Current teacher:** `best_run3_3.24deg.pt` (sha256 `b68131063b6475f8…`), GazeDINO ViT-B/14, trained on GazeGene only (`RULE-R1.md`, clarifications).
- **T1:** the same architecture, fine-tuned from the current teacher on GazeGene plus real faces with their true gaze labels.

## Data

| Group | People | Use in T1 |
|---|---|---|
| GazeGene | the teacher's 50 subjects and split | training and validation, as in `train_gaze.py` |
| MPIIFaceGaze | p00–p09 | training, with labels |
| RT-GENE | s000, s002 (with s014), s007, s008, s010 | training, with labels |
| RT-GENE | s001, s013 | checkpoint selection only, with labels |
| **Test** | MPIIFaceGaze p10–p14 and the 8 RT-GENE test people of `README.md` | final test only, as in R1 |
| AVCAffe | — | not used: it has no gaze labels |

- **RT-GENE selection people:** the last two of the 7 distillation people in the order of `split_rtgene.py`, sha256 of `eyegaze-r1:` plus the ID. The order is s000, s010, s007, s008, s002, s001, s013.
- **Test people:** they stay out of everything until the final test, as in R1. MPII p10–p14 included.

**Faces and frames.** Faces are cut as Iride cuts them, as in R1:

- RT-GENE: `inpainted/face_after_inpainting` only;
- MPIIFaceGaze: the frames as released.

The test frames, calibration draws, errors and bootstrap are those of `RULE-R1.md`, with the same seeds.

## Labels

Every label is turned into GazeGene's (yaw, pitch), as `data/gazegene_loader.py` of eye-gaze does:

- a unit gaze vector (x, y, z) in camera coordinates;
- yaw = atan2(x, −z), positive looking right;
- pitch = atan2(−y, √(x² + z²)), positive looking up.

**MPIIFaceGaze:** the direction from the face centre to the gaze target, as in `eval_mpii.py`.

**RT-GENE:**

- The gaze columns of `label_combined.txt` are turned into a camera-frame vector by the dataset's own convention (its code, `rt_gene`), then into (yaw, pitch) as above.
- **Check before any training**, on the 5 RT-GENE training people only:
  - the current teacher's predictions against the converted labels must show the same sign on both axes;
  - the fitted calibration slope per axis must be between 0.5 and 2;
  - the uncalibrated median error must be under 20°.
  
  If the check fails, the conversion is fixed and checked again, still on those 5 people. The check's numbers go in `README.md` before training.

## Training

- Fine-tuned from the current teacher, with a low learning rate (the phase-2 rate of `train_gaze.py`, 5e-6), all layers.
- Each batch is half GazeGene and half real faces. The real half is drawn so that every training person weighs the same.
- Same augmentation as `train_gaze.py`.
- At most 30 epochs, patience 8.
- **Checkpoint selection:**
  - the epoch kept has the lowest mean calibrated error over s001 and s013;
  - an epoch whose GazeGene validation error is more than 0.3° above the current teacher's (3.24°) is not eligible;
  - a tie goes to the earlier epoch.
- Runs on the Mac mini after R1, one job at a time.

## Criterion: T1 is better than the current teacher only if all four hold

1. **Main:** over the test people of MPIIFaceGaze and RT-GENE together, each person weighing the same, the 95% bootstrap interval of the mean per-person difference T1 − current teacher in calibrated angular error is entirely below 0.
2. **No dataset worse:** on MPIIFaceGaze p10–p14 alone and on the RT-GENE test people alone, the mean difference T1 − current teacher is at most +0.1°.
3. **Synthetic data:** T1's GazeGene validation error is at most 0.3° above 3.24°.
4. **No regression on the recorded sessions:** in screen px, on sessions 2 and 3 as in `../distill-v3`, no column (cv, still, back, left, closer, down) is more than 10% worse than the current teacher.

Also reported, not part of the criterion:

- the uncalibrated error on both test sets;
- the error per head pose band and per distance on the RT-GENE test people;
- Joel's session, as a control.

## After T1

If T1 is better, a student is distilled from it. It uses the same sources, selection and test as R1 (`RULE-R1.md`) and is compared with v3 and R1 on the same people. Its own short rule is committed before it is trained.

**Licenses:** T1 and its student are trained on non-commercial datasets (MPIIFaceGaze, RT-GENE). The documentation says so. Their use in N2uCore is decided apart from this rule.
