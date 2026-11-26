# Distillation research line: more people from public datasets

Status 2026-11-26: the rule for R1 is written (`RULE-R1.md`). Nothing has been extracted or trained yet.

This is a research line, separate from the product student:

- The product student is distillation v3 (`../distill-v3`), on the CPU device in N2uCore_dev since 48d5f63. Nothing here changes it.
- The students trained here are called **R1, R2, …** (research), never v4. **They are research models**: every dataset they learn from is non-commercial only. They do not replace v3 in N2uCore without an explicit decision and a license check.
- Every stage has its rule, committed before its data is used: who is distilled, who is tested, and when the student beats v3.
- Code, results and documentation stay in this repository, never in N2uCore_dev.

## Licenses

Every dataset used here allows academic, non-commercial research only (MPIIFaceGaze, which v3 used, already did).

| Dataset | Where | Use here |
|---|---|---|
| AVCAffe (Sarkar, Posen, Etemad, AAAI 2023, arXiv:2205.06887): academic research only, no redistribution, no re-identification | Mac mini, `/Volumes/HARD DISK/datasets/AVCAffe/` | R1: distillation, checkpoint selection |
| RT-GENE | Mac mini, `/Volumes/RT-GENE-data` (APFS, because some files exceed FAT32's 4 GB) | R1: distillation and external test, split by person |
| MPIIFaceGaze | Mac mini, `/Volumes/HARD DISK/datasets/MPIIFaceGaze/` | as in v3: distillation p00–p09, test p10–p14 |
| MPIIGaze (normalized eye images) | Mac mini, `/Volumes/HARD DISK/datasets/MPIIGaze/` | not in R1; for checking the eye crop |
| ETH-XGaze | not yet available | R2, with its own rule |
| GazeCapture | not yet available | later, with its own rule |

For all of them:

- data and everything derived from them (frames, crops, shards, checkpoints, logs) stay on the Mac mini;
- AVCAffe data is not copied anywhere else, the FILESERVER included;
- this repository holds only code and numbers: no frames, no crops, no face images, no weights.

## R1: AVCAffe and RT-GENE

The rule is in `RULE-R1.md`. In short:

- **Distillation faces:**
  - AVCAffe frames of 86 people;
  - half of the RT-GENE people;
  - v3's sources (GazeGene, MPIIFaceGaze p00–p09, one recorded session).
- **Checkpoint selection:** the 20 AVCAffe people of `val.txt`, by the gap to the teacher.
- **Test with true labels:**
  - MPIIFaceGaze p10–p14;
  - the other half of the RT-GENE people, never seen by anything before the test.
- **Criterion:** R1 beats v3 only if all of these hold:
  - the bootstrap interval of the per-person difference in calibrated angular error is below 0;
  - neither test set is worse by more than 0.1°;
  - the recorded sessions do not regress by more than 10%;
  - the speed on the Mac mini CPU stays within 10%.

**What each dataset adds:**

- **AVCAffe:** many people (106), but almost all frontal faces at a desk, in webcam video calls at 640×360.
- **RT-GENE:** distances from 0.5 to 3 m and varied head poses, with true labels.
- **ETH-XGaze (R2):** the extreme poses.

### RT-GENE split

The split was made by `split_rtgene.py` on 2026-11-26, after the volume's MD5 check passed 35/35. It read names only: the volume's entries and the member names of `RT-GENE dataset.zip`. It was committed before any RT-GENE image was read.

There are 17 sessions of 15 people: s014 is s002 recorded again, and s015 is s005.

| Group | People | Sessions |
|---|---|---|
| **Test** (true labels, never seen before the test) | 8: s003, s004, s005, s006, s009, s011, s012, s016 | the same, plus s015 (s005) |
| **Distillation** (unlabelled, teacher labels) | 7: s000, s001, s002, s007, s008, s010, s013 | the same, plus s014 (s002) |

A person with no inpainted faces is excluded from their group, without replacement. Exclusions will be listed here.

### AVCAffe crops

- **Sampling:** one frame every 2 s, about 230 per video and about 220k in all. Skip the macOS `._*` files next to each video.
- **Script:** `export_crops_video.py` (to write). It reads each video once, cuts the face as Iride does, and stores per crop:
  - the person and the task;
  - the frame index;
  - the face size;
  - the head yaw and pitch from FaceLandmarker's transformation matrix.
- **Storage:** FAT32 holds few entries per directory, and small files read slowly from the USB disk. So crops go in one `.npz` per person: the JPEG bytes (about 10–13 KB a crop, 13–23 MB a person in the pilot) and the per-crop fields. Training reads them once, into memory.
- **Pilot first:** 2 training people. Check the crop yield and the face sizes, and look at a few crops on the Mac mini screen. Only then extract all.

### Steps

1. `export_crops_video.py` and the evaluation script (calibrated error per person, bootstrap, paired difference, gap per band), tested on a recorded session of ours.
2. AVCAffe pilot, then all 106 people, one job at a time on the Mac mini.
3. Once RT-GENE is confirmed: commit the split, then crop the distillation half.
4. Measure v3 and the teacher on the test sets: the baseline.
5. Train R1. Checkpoints and logs go on the HARD DISK.
6. Measure R1 and apply the criterion. Write the results here.

## R2: ETH-XGaze (when the link arrives)

With a new rule, committed before use.

- The 224 version gives normalized face patches. The tracking landmarks run on each patch and Iride's box is cut from it.
- The official test people are the external test.

GazeCapture is not awaited. If it arrives, it gets its own rule.
