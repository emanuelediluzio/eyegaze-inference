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
