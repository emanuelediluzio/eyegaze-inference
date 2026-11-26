# Rule R1: who is distilled, who is tested, when R1 beats v3

Written 2026-11-26, committed before any R1 data is extracted or any training runs. After the commit this file does not change. A change is a new rule (R1b, R2, …), committed before the work it governs.

R1 is a research student (see `README.md`, Licenses). Whatever the result, it does not replace v3 in N2uCore without an explicit decision.

## Models compared

- **v3:** the product student, `../distill-v3`, best checkpoint, as in N2uCore_dev 48d5f63.
- **R1:** a ViT-S student of the same architecture, trained from v3's best checkpoint with the recipe of `distill_gaze_v3.py` and the extra face sources below.
- **Teacher ViT-B:** reported for reference on the same frames, not part of the criterion. It was trained on GazeGene, which is synthetic, so it has never been trained on any face of these datasets either.

## People

The split is by person, never by frame: every image of a person is in exactly one group.

| Dataset | Distillation (unlabelled faces, teacher labels) | Checkpoint selection | Test (true labels) |
|---|---|---|---|
| GazeGene | v3's subjects, as in v3 | — | — |
| MPIIFaceGaze | p00–p09 (crops already used by v2 and v3) | — | **p10–p14** |
| Recorded sessions | `gazefree-20261123-1615-emanuelediluzio` (as in v3) | — | — |
| AVCAffe | the 86 people of `info/train.txt` | the 20 people of `info/val.txt` | — |
| RT-GENE | the second half of `split_rtgene.py` | — | **the first half of `split_rtgene.py`** |
| Joel, `gazefree-20261123-1622-joelnovello` | — | — | control, reported apart |

- **RT-GENE test people** stay out of everything:
  - no image of them is cropped, labelled by the teacher, or shown to either student before the final test;
  - their only use is the final test, with their true labels.
  
  The groups come from `split_rtgene.py`, which reads only the folder names: subjects ordered by sha256 of a fixed salt plus the ID, first half (rounded up) to test.
  
  The script runs once, after the dataset's MD5 check is confirmed complete (35/35). Its output is committed into this repository, in `README.md`, before any RT-GENE image is read.
- **MPIIFaceGaze p10–p14** are never in any distillation pool. v2 and v3 used them to pick their checkpoint. R1 does not, so the comparison on them leans towards v3, not R1.
- **AVCAffe `val.txt`** is used only to pick R1's checkpoint, by the mean gap to the teacher over those 20 people, each person weighing the same. No AVCAffe person is in a test.
- **Joel** is not used for selection in R1. He is reported as a control but is not part of the criterion, because v3 used him for selection.
- **MPIIGaze** (eye images) is not used in R1.

## Faces as in production

Every face, for distillation or test, is cut as Iride cuts it:

- the tracking node's FaceLandmarker (`../tracking_landmarks.py`);
- the box of the 478 landmarks (`GazeDinoEngine._faceBox`), resized to 224×224.

Images without a face, with the eyes closed, or with the face box at the image edge are dropped. Their count is reported per dataset.

For RT-GENE, the face images with the eye-tracking glasses inpainted are used when the dataset provides them, for distillation and test alike. Faces in Iride wear no tracking glasses.

Every model is evaluated on exactly the same test frames.

## Error

**Angular error in degrees, calibrated per person**, as in `../model-size/eval_mpii.py`:

- a straight line per axis (yaw, pitch) is fitted on 16 frames of the person;
- the error is measured on the person's other frames;
- the result is the mean of 20 random draws of the 16 frames.

This is what the screen calibration does in Iride. It removes each dataset's own camera and axis conventions.

**Test frames:**

- MPIIFaceGaze: 300 frames per person, as in `eval_mpii.py`;
- RT-GENE: at most 1,000 frames per person, evenly spaced in recording order, chosen once.

**Statistics:**

- a person's error is the mean over the 20 draws;
- the 95% interval of a person's error is a percentile bootstrap over that person's test frames: 10,000 resamples, seed 0;
- the difference between two models is taken per person (R1 − v3), and its mean over people has a percentile bootstrap interval over people: 10,000 resamples, seed 0.

## Criterion: R1 is better than v3 only if all four hold

1. **Main:** over the test people of MPIIFaceGaze and RT-GENE together, each person weighing the same, the 95% bootstrap interval of the mean per-person difference R1 − v3 is entirely below 0.
2. **No dataset worse:** on MPIIFaceGaze p10–p14 alone and on the RT-GENE test people alone, the mean difference R1 − v3 is at most +0.1°.
3. **No regression on the recorded sessions:** in screen px, on sessions 2 and 3 as in `../distill-v3`, no column (cv, still, back, left, closer, down) is more than 10% worse than v3.
4. **Speed:** on the Mac mini M1 CPU, the time per face of R1 is at most 10% above v3's. Both are measured as the CPU device runs them:
   - Core ML int8 with `CPU_ONLY`;
   - the int8 joblib;
   - for each, the median of 500 calls at batch 1 after 50 warm-up calls.
   
   R1 has v3's architecture, so this checks only that nothing in the export changed.

If any fails, R1 is reported as not better than v3. The numbers are written in `README.md` either way.

## Also reported, not part of the criterion

- **The teacher** on the same test frames.
- **Error per head pose band** on the RT-GENE test people, from the dataset's head pose labels:
  - |yaw| < 10°, 10–20°, 20–30°, ≥ 30°;
  - pitch < −10°, −10…10°, ≥ 10°;
  - only people with at least 20 frames in a band count.
- **Distance on RT-GENE:** by the face size, the longer box side over the image width.
- **Gap to the teacher** on AVCAffe `val.txt`: per person, and per band of the head pose estimated by FaceLandmarker.
- **Joel, as control:** the gap to the teacher, overall and per recording phase.
- **The uncalibrated error on MPIIFaceGaze**, as in `eval_mpii.py`.

## Where things stay

- AVCAffe frames, crops and anything derived from them stay on the Mac mini's HARD DISK, under `/Volumes/HARD DISK/datasets/AVCAffe/`. They are not copied anywhere else, the FILESERVER included.
- RT-GENE derived files stay on the Mac mini, under `/Volumes/RT-GENE-data/`.
- R1 checkpoints and logs stay on the Mac mini's HARD DISK.
- This repository gets code and numbers only.

## Clarifications (2026-11-26, second commit, before any extraction)

These make the rule above precise. They do not change the criterion. Where they are more precise than a sentence above, they govern.

### What the teacher was trained on

The teacher is `best_run3_3.24deg.pt`. Its sha256 begins with `b68131063b6475f8`, and it is the same file on the MacBook, on the Mac mini (`~/training_mini/gazedistill/teacher/`) and in this repository's `checkpoints/`.

- **Training repository:** [eye-gaze](https://github.com/emanuelediluzio/eye-gaze), `train_gaze.py`, which loads data only through `data/gazegene_loader.py`. No real-face dataset appears in its code.
- **Run 3:** phase 2 only, resumed from Run 2, which comes from Run 1. All three are GazeGene (`results.md` there). Run 3 used 50 subjects × 9 cameras × 200 frames, validation error 3.24°, epoch 49.
- **GazeGene** is synthetic: rendered faces, no real person.
- **Backbone:** the DINOv2 ViT-B/14 weights of `facebookresearch/dinov2`, self-supervised on LVD-142M (web images, no labels). We cannot rule out that a web photo of someone in these datasets is among those images, but no gaze label comes with it.

So the teacher has never been trained on MPIIFaceGaze, RT-GENE or AVCAffe. Its predictions carry no label of the test people, and **both tests stand**: MPIIFaceGaze p10–p14 and the RT-GENE test people.

The earlier students:

- **v1:** distilled on GazeGene only.
- **v2 and v3:** added the teacher-labelled crops of MPIIFaceGaze p00–p09 and, for v3, the session `gazefree-20261123-1615-emanuelediluzio`.
- **Checkpoint selection:** v2 and v3 picked their checkpoint on the gap to the teacher over p10–p14 crops, without labels, and v3 also over Joel's session.
- **RT-GENE:** no student has ever seen it.

If any of this proves wrong for a test set, only the test sets the teacher has never seen are used. That would be written here in a new rule before any result is reported.

### Test frames, the same for every model

Frames are chosen from the file lists and labels only, before any model runs. The frames v3, R1 and the teacher are scored on are the same set.

- **MPIIFaceGaze:** per person, 300 lines of `pXX.txt`, evenly spaced (`np.linspace(0, n − 1, 300)` rounded down), as in `eval_mpii.py`.
- **RT-GENE:** per person, the frames that have both a label and an inpainted face image, in frame-number order. If there are more than 1,000, 1,000 of them are taken, evenly spaced in the same way.
- **Landmark failures:** frames where the landmarks find no face, or that fail the face checks, are dropped. The landmarks do not depend on the gaze model, so the dropped frames are the same for every model. The scoring uses the frames valid for all models compared. Counts are reported.

### Calibration frames are not test frames

For a person with N valid test frames, the 16 calibration frames of draw k (k = 0…19) are drawn without replacement from 0…N−1 by `numpy.random.default_rng(seed)`.

- **Seed:** the first 8 bytes of sha256(`"eyegaze-r1-cal:" + dataset + ":" + person + ":" + k`), as an unsigned integer.
- **Same for every model:** the draws depend only on the person and N, never on the model or on other people, so v3, R1 and the teacher get the same 16 frames in every draw.
- **Exclusion:** the error of draw k is measured only on the N − 16 frames not drawn. The 16 calibration frames never count in the error.
- **A person's error:** the mean over the 20 draws.
- **Bootstrap over a person's frames:** each frame's error is its mean over the draws in which it was not a calibration frame. Those per-frame errors are resampled.

The R1 evaluation script implements exactly this, and v3 and the teacher are re-measured with it. No earlier number is reused in the comparison.

### Checkpoint selection on AVCAffe `val.txt`

There are no labels. The measure is the **distance from the teacher's predictions**:

- per crop, the angle between the gaze directions predicted by the student and by the teacher;
- per person, the mean over that person's selection crops;
- then the mean over the 20 people, each weighing the same.

**Selection crops:** per person, at most 500 of their AVCAffe crops, evenly spaced in time over all their tasks. They are fixed once, after extraction and before training, and are the same at every epoch.

The checkpoint with the lowest value is kept. A tie goes to the earlier epoch. MPIIFaceGaze p10–p14, the RT-GENE test people and Joel are never part of this measure.

### RT-GENE without inpainted faces

Only the images with the eye-tracking glasses inpainted are used. There is never a fallback to the images with the glasses.

- A frame without an inpainted image is not used.
- A person with no inpainted images is excluded from both groups (distillation and test), and the exclusion is reported.
- A person stays in the group `split_rtgene.py` gives them: an excluded person is not replaced.

## Clarification: RT-GENE people and images (2026-11-26, third commit, before the split is run)

The split still goes by person. This only makes "person" exact for RT-GENE, from its `README_Dataset.md`:

- 15 participants were recorded in 17 sessions, s000–s016;
- **session s014 is participant s002 again**, and **s015 is participant s005 again**, on different days with different camera poses.

`split_rtgene.py` therefore splits the 15 people, not the 17 sessions. s014 goes with s002 and s015 with s005, so both sessions of a person are always in the same group.

**Images used** for distillation and test alike: those in `inpainted/face_after_inpainting` of the `sNNN_glasses` sessions. They are the face with a wide margin and the glasses inpainted, so Iride's landmark box fits inside them.

**Not used in R1:**

- the MTCNN crops (`inpainted/face`): their tight margin would cut the landmark box;
- the images with the glasses (`original/`);
- the `sNNN_noglasses` sessions, even for distillation people.

**Labels:** the test labels are the gaze columns of `label_combined.txt`, matched to the images by sequence number. The head pose bands use its head pose columns.

**Data on the volume:** the data are tar archives (inside `RT-GENE dataset.zip`, and three loose copies). They are extracted on the Mac mini into `/Volumes/RT-GENE-data`, only for the distillation people's sessions until the baseline step. The test people's archives are opened only for the test.
