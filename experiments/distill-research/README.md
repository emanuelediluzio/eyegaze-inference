# Distillation research line: more people from public datasets

Status 2026-11-26: the rule for R1 is written (`RULE-R1.md`). Nothing has been extracted or trained yet.

This is a research line, separate from the product student:

- The product student is distillation v3 (`../distill-v3`), on the CPU device in N2uCore_dev since 48d5f63. Nothing here changes it.
- The students trained here are called **R1, R2, …**, never v4; a teacher fine-tuned on real labelled faces is **T1** (`RULE-T1.md`). Every real dataset they learn from is non-commercial only (table below), as MPIIFaceGaze in v3 already was. Their use in N2uCore is a separate decision.
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

A person with no inpainted faces is excluded from their group, without replacement.

**Distillation sessions extracted** on 2026-11-26. Every one has inpainted faces, so no person is excluded. All `inpainted/face_after_inpainting` images went through `export_crops_video.py --frames` (every image, no sampling). The test people's archives are still closed.

| Session | Person | Inpainted faces | Kept | No face | Edge | Eyes closed |
|---|---|---|---|---|---|---|
| s000 | s000 | 5,620 | 5,142 | 106 | 24 | 348 |
| s001 | s001 | 11,947 | 8,533 | 9 | 23 | 3,382 |
| s002 | s002 | 5,869 | 5,761 | 10 | 56 | 42 |
| s014 | s002 | 6,442 | 5,780 | 8 | 9 | 645 |
| s007 | s007 | 5,702 | 5,560 | 11 | 35 | 96 |
| s008 | s008 | 5,284 | 4,391 | 685 | 207 | 1 |
| s010 | s010 | 11,069 | 10,862 | 8 | 8 | 191 |
| s013 | s013 | 12,201 | 9,659 | 170 | 95 | 2,277 |
| **All** | 7 people | 64,134 | 55,688 | | | |

### AVCAffe crops

- **Sampling:** one frame every 2 s, about 180 per video and 169k in all. Skip the macOS `._*` files next to each video.
- **Script:** `export_crops_video.py`. It reads each video once, cuts the face as Iride does, and stores per crop:
  - the person and the task;
  - the frame index;
  - the face size;
  - the head yaw and pitch from FaceLandmarker's transformation matrix.
- **Storage:** FAT32 holds few entries per directory, and small files read slowly from the USB disk. So crops go in one `.npz` per person: the JPEG bytes (about 10–13 KB a crop, 13–23 MB a person in the pilot) and the per-crop fields. Training reads them once, into memory.
- **Pilot:** 2 training people (aiim005, aiim006), 1,744 and 1,276 crops, about 37 s per person. The chain worked and the crops looked right. The pilot's numbers do not count for the criterion.

#### Extraction

Extracted 2026-11-26 (`export_crops_video.py`, one frame every 2 s): 135,464 crops kept of 169,473 frames sampled from 950 videos. Dropped: 18,763 eyes closed, 13,144 face at the frame edge, 2,102 no face.

**Exclusions:** none. No person is under the 300-crop threshold: the minimum is 503 crops, the median 1,333 and the maximum 1,767.

**Tasks 4 and 5:** few crops because the videos are short (about 3 minutes instead of 7–8), not because more frames are dropped. The kept share is 84–85% there and 76–84% in the other tasks.

| Task | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Kept | 19,203 | 17,645 | 21,453 | 6,640 | 6,490 | 19,890 | 11,277 | 9,674 | 23,192 |
| Sampled | 23,925 | 23,171 | 28,214 | 7,921 | 7,639 | 24,092 | 13,461 | 11,937 | 29,113 |

<details><summary>Per person: kept / sampled crops, per task kept / sampled</summary>

| Person | Group | Kept | Sampled | No face | Edge | Closed | Per task (kept/sampled) |
|---|---|---|---|---|---|---|---|
| aiim001 | selection | 1,590 | 1,829 | 1 | 12 | 226 | 1:209/231 2:189/226 3:260/300 4:77/91 5:64/77 6:275/301 7:131/151 8:137/151 9:248/301 |
| aiim002 | selection | 1,767 | 1,829 | 2 | 15 | 45 | 1:223/231 2:215/226 3:292/301 4:90/91 5:77/77 6:298/301 7:144/150 8:138/151 9:290/301 |
| aiim005 | distillation | 1,744 | 1,857 | 1 | 18 | 94 | 1:244/257 2:194/238 3:276/301 4:75/76 5:76/76 6:292/301 7:151/154 8:151/153 9:285/301 |
| aiim006 | distillation | 1,276 | 1,855 | 77 | 425 | 77 | 1:229/256 2:217/238 3:132/301 4:17/76 5:28/76 6:220/301 7:87/154 8:127/152 9:219/301 |
| aiim007 | distillation | 1,677 | 1,726 | 2 | 0 | 47 | 1:216/226 2:218/226 3:294/301 4:74/76 5:75/76 6:289/293 7:86/90 8:130/137 9:295/301 |
| aiim008 | distillation | 1,407 | 1,725 | 4 | 7 | 307 | 1:162/226 2:142/226 3:261/301 4:67/76 5:65/76 6:267/293 7:87/90 8:109/137 9:247/300 |
| aiim009 | distillation | 829 | 1,973 | 256 | 17 | 871 | 1:161/408 2:77/226 3:141/302 4:38/76 5:28/64 6:122/301 7:76/147 8:67/148 9:119/301 |
| aiim010 | distillation | 813 | 1,971 | 142 | 848 | 168 | 1:372/408 2:101/226 3:14/301 4:2/76 5:63/63 6:83/301 7:142/147 8:30/148 9:6/301 |
| aiim011 | distillation | 1,605 | 1,758 | 27 | 57 | 69 | 1:201/225 2:208/230 3:232/268 4:73/75 5:76/76 6:215/267 7:147/151 8:144/146 9:309/320 |
| aiim012 | distillation | 1,532 | 1,738 | 2 | 3 | 201 | 1:175/225 2:176/230 3:250/268 4:70/76 5:73/76 6:253/267 7:144/151 8:144/146 9:247/299 |
| aiim013 | selection | 1,196 | 1,477 | 24 | 20 | 237 | 1:136/202 2:122/145 3:223/278 4:64/76 5:65/76 6:175/212 7:103/148 8:86/97 9:222/243 |
| aiim014 | selection | 1,073 | 1,476 | 1 | 312 | 90 | 1:179/202 2:121/145 3:253/278 4:73/76 5:65/76 6:143/213 7:138/147 8:41/96 9:60/243 |
| aiim015 | distillation | 1,364 | 1,642 | 4 | 180 | 94 | 1:191/226 2:199/226 3:264/288 4:75/76 5:76/76 6:254/257 7:102/115 8:88/91 9:115/287 |
| aiim016 | distillation | 1,400 | 1,645 | 15 | 124 | 106 | 1:204/226 2:161/226 3:174/289 4:75/76 5:74/76 6:244/258 7:114/115 8:90/92 9:264/287 |
| aiim017 | distillation | 1,679 | 1,779 | 10 | 18 | 72 | 1:221/231 2:218/239 3:256/281 4:69/76 5:69/76 6:228/236 7:143/150 8:121/124 9:354/366 |
| aiim018 | distillation | 813 | 1,799 | 31 | 795 | 160 | 1:152/231 2:168/239 3:36/301 4:29/76 5:60/76 6:62/236 7:58/150 8:41/124 9:207/366 |
| aiim019 | distillation | 1,246 | 1,563 | 16 | 65 | 236 | 1:176/226 2:178/226 3:229/297 4:66/76 5:62/76 6:162/197 7:91/97 8:92/121 9:190/247 |
| aiim020 | distillation | 1,423 | 1,562 | 4 | 16 | 119 | 1:204/226 2:177/226 3:274/297 4:75/76 5:72/76 6:191/197 7:93/97 8:114/121 9:223/246 |
| aiim021 | distillation | 1,284 | 1,511 | 16 | 46 | 165 | 1:180/237 2:185/226 3:114/172 4:74/76 5:72/76 6:221/232 7:118/127 8:113/130 9:207/235 |
| aiim022 | distillation | 1,396 | 1,511 | 0 | 3 | 112 | 1:208/237 2:200/226 3:163/172 4:70/76 5:73/76 6:221/231 7:119/127 8:123/130 9:219/236 |
| aiim023 | distillation | 1,443 | 1,634 | 5 | 22 | 164 | 1:205/226 2:195/226 3:278/301 4:40/45 5:57/74 6:202/218 7:130/150 8:105/126 9:231/268 |
| aiim024 | distillation | 1,319 | 1,634 | 10 | 3 | 302 | 1:170/226 2:116/226 3:275/301 4:39/45 5:60/74 6:187/218 7:136/150 8:118/126 9:218/268 |
| aiim025 | distillation | 1,432 | 1,471 | 0 | 18 | 21 | 1:229/236 2:154/169 3:290/301 4:75/76 5:60/61 6:217/219 7:125/125 8:145/147 9:137/137 |
| aiim026 | distillation | 1,232 | 1,470 | 1 | 2 | 235 | 1:190/236 2:143/169 3:255/301 4:69/76 5:49/60 6:189/218 7:104/126 8:118/147 9:115/137 |
| aiim027 | distillation | 1,517 | 1,545 | 0 | 3 | 25 | 1:221/226 2:221/226 3:259/262 4:72/74 5:77/77 6:149/151 7:133/134 8:87/90 9:298/305 |
| aiim028 | distillation | 1,402 | 1,545 | 2 | 55 | 86 | 1:201/226 2:199/226 3:195/262 4:72/74 5:77/77 6:146/151 7:130/133 8:84/91 9:298/305 |
| aiim029 | distillation | 1,309 | 1,327 | 0 | 1 | 17 | 1:226/226 2:222/226 3:252/255 4:74/76 5:75/76 6:106/108 7:113/114 8:98/101 9:143/145 |
| aiim030 | distillation | 1,150 | 1,327 | 0 | 0 | 177 | 1:188/226 2:165/226 3:233/255 4:69/76 5:68/76 6:104/108 7:108/114 8:84/102 9:131/144 |
| aiim031 | distillation | 1,446 | 1,742 | 60 | 122 | 114 | 1:135/226 2:145/226 3:253/301 4:73/76 5:75/76 6:233/244 7:145/151 8:107/151 9:280/291 |
| aiim032 | distillation | 1,547 | 1,742 | 2 | 0 | 193 | 1:201/226 2:139/225 3:287/301 4:62/76 5:72/76 6:231/245 7:145/151 8:149/151 9:261/291 |
| aiim033 | distillation | 1,401 | 1,475 | 1 | 4 | 69 | 1:193/226 2:163/173 3:267/281 4:74/76 5:60/61 6:141/143 7:117/119 8:98/101 9:288/295 |
| aiim034 | distillation | 970 | 1,491 | 21 | 443 | 57 | 1:189/226 2:76/173 3:168/280 4:47/76 5:66/76 6:46/143 7:80/121 8:39/101 9:259/295 |
| aiim035 | distillation | 944 | 1,288 | 255 | 0 | 89 | 1:110/156 2:106/220 3:277/301 4:70/76 5:50/53 6:78/79 7:76/91 8:76/88 9:101/224 |
| aiim036 | distillation | 1,046 | 1,289 | 23 | 12 | 208 | 1:91/156 2:131/219 3:279/301 4:67/76 5:50/54 6:77/80 7:83/91 8:78/88 9:190/224 |
| aiim037 | distillation | 1,335 | 1,679 | 29 | 251 | 64 | 1:210/226 2:214/242 3:226/302 4:56/78 5:64/76 6:80/178 7:86/112 8:110/129 9:289/336 |
| aiim038 | distillation | 1,542 | 1,679 | 4 | 3 | 130 | 1:199/226 2:202/242 3:289/302 4:74/78 5:72/76 6:165/178 7:108/112 8:126/129 9:307/336 |
| aiim039 | distillation | 1,151 | 1,514 | 3 | 25 | 335 | 1:127/226 2:142/226 3:180/191 4:67/76 5:73/74 6:206/226 7:103/131 8:71/83 9:182/281 |
| aiim040 | distillation | 1,413 | 1,514 | 0 | 0 | 101 | 1:199/226 2:197/226 3:188/191 4:75/76 5:72/74 6:223/226 7:124/131 8:80/83 9:255/281 |
| aiim041 | distillation | 791 | 1,661 | 10 | 205 | 655 | 1:97/226 2:63/226 3:93/301 4:41/76 5:48/76 6:150/228 7:62/126 8:68/99 9:169/303 |
| aiim042 | distillation | 1,448 | 1,661 | 45 | 111 | 57 | 1:208/226 2:189/226 3:187/301 4:68/76 5:75/76 6:213/228 7:116/126 8:96/99 9:296/303 |
| aiim043 | selection | 807 | 1,609 | 17 | 661 | 124 | 1:186/225 2:160/226 3:61/244 4:19/76 5:23/75 6:159/300 7:61/80 8:45/82 9:93/301 |
| aiim044 | selection | 1,397 | 1,614 | 1 | 3 | 213 | 1:179/225 2:187/226 3:218/244 4:68/76 5:69/75 6:262/300 7:82/85 8:77/82 9:255/301 |
| aiim045 | distillation | 1,442 | 1,535 | 4 | 1 | 88 | 1:212/226 2:206/225 3:203/207 4:71/75 5:73/76 6:214/231 7:89/93 8:90/100 9:284/302 |
| aiim046 | distillation | 1,350 | 1,535 | 10 | 7 | 168 | 1:199/226 2:163/225 3:199/207 4:73/75 5:68/76 6:217/231 7:89/93 8:92/100 9:250/302 |
| aiim047 | distillation | 1,536 | 1,598 | 2 | 22 | 38 | 1:210/226 2:211/226 3:274/301 4:76/76 5:58/58 6:207/207 7:85/85 8:116/116 9:299/303 |
| aiim048 | distillation | 1,506 | 1,598 | 0 | 3 | 89 | 1:206/226 2:192/226 3:298/301 4:75/76 5:56/58 6:196/207 7:81/85 8:116/116 9:286/303 |
| aiim049 | distillation | 1,117 | 1,554 | 5 | 329 | 103 | 1:156/226 2:188/227 3:184/301 4:58/76 5:68/76 6:170/219 7:102/108 8:81/87 9:110/234 |
| aiim050 | distillation | 1,443 | 1,554 | 4 | 3 | 104 | 1:200/226 2:200/227 3:285/301 4:65/76 5:72/76 6:212/219 7:102/108 8:83/87 9:224/234 |
| aiim051 | distillation | 1,280 | 1,469 | 24 | 14 | 151 | 1:144/163 2:205/223 3:251/295 4:70/76 5:46/53 6:153/158 7:126/143 8:81/111 9:204/247 |
| aiim052 | distillation | 1,366 | 1,469 | 13 | 4 | 86 | 1:144/163 2:197/223 3:293/295 4:73/76 5:52/53 6:145/158 7:131/143 8:92/111 9:239/247 |
| aiim053 | selection | 671 | 1,399 | 1 | 0 | 727 | 1:94/226 2:67/226 3:184/299 4:41/76 5:36/76 6:99/163 7:53/96 8:38/70 9:59/167 |
| aiim054 | selection | 1,136 | 1,399 | 1 | 174 | 88 | 1:197/226 2:203/226 3:214/299 4:74/76 5:70/76 6:76/163 7:81/96 8:57/70 9:164/167 |
| aiim055 | distillation | 1,459 | 1,680 | 17 | 56 | 148 | 1:169/226 2:210/228 3:272/301 4:72/76 5:67/76 6:162/191 7:134/151 8:124/130 9:249/301 |
| aiim056 | distillation | 1,417 | 1,680 | 21 | 17 | 225 | 1:168/226 2:156/228 3:255/301 4:67/76 5:69/76 6:181/191 7:149/151 8:121/130 9:251/301 |
| aiim057 | selection | 1,598 | 1,692 | 4 | 10 | 80 | 1:208/228 2:212/226 3:230/242 4:70/75 5:70/75 6:252/259 7:127/145 8:141/142 9:288/300 |
| aiim058 | selection | 1,615 | 1,693 | 1 | 1 | 76 | 1:214/228 2:205/226 3:240/242 4:72/76 5:74/75 6:249/259 7:141/145 8:138/142 9:282/300 |
| aiim059 | distillation | 1,605 | 1,792 | 3 | 12 | 172 | 1:195/225 2:192/225 3:277/300 4:74/75 5:64/69 6:287/301 7:142/150 8:126/146 9:248/301 |
| aiim060 | distillation | 1,370 | 1,792 | 136 | 229 | 57 | 1:196/225 2:203/225 3:274/300 4:73/75 5:65/69 6:248/301 7:147/150 8:135/146 9:29/301 |
| aiim061 | distillation | 928 | 1,604 | 7 | 120 | 549 | 1:114/226 2:86/225 3:160/242 4:52/75 5:51/75 6:111/179 7:105/147 8:41/135 9:208/300 |
| aiim062 | distillation | 1,412 | 1,604 | 32 | 40 | 120 | 1:191/226 2:187/225 3:223/242 4:72/75 5:75/75 6:165/179 7:109/147 8:122/135 9:268/300 |
| aiim063 | selection | 667 | 1,621 | 18 | 861 | 75 | 1:204/226 2:166/226 3:118/301 4:12/70 5:8/76 6:38/213 7:75/134 8:30/151 9:16/224 |
| aiim064 | selection | 855 | 1,622 | 5 | 24 | 738 | 1:88/226 2:47/226 3:183/301 4:43/71 5:36/76 6:134/213 7:80/134 8:108/151 9:136/224 |
| aiim065 | distillation | 607 | 1,653 | 74 | 886 | 86 | 1:137/219 2:78/226 3:72/258 4:51/75 5:8/76 6:55/255 7:13/150 8:12/93 9:181/301 |
| aiim066 | distillation | 1,270 | 1,653 | 21 | 130 | 232 | 1:129/219 2:135/226 3:164/258 4:61/75 5:67/76 6:233/255 7:132/150 8:87/93 9:262/301 |
| aiim067 | selection | 785 | 1,306 | 6 | 269 | 246 | 1:154/225 2:134/225 3:59/218 4:33/75 5:26/75 6:116/166 7:96/133 8:59/69 9:108/120 |
| aiim068 | selection | 641 | 1,306 | 11 | 308 | 346 | 1:99/225 2:73/225 3:64/218 4:51/75 5:37/75 6:94/166 7:75/133 8:48/69 9:100/120 |
| aiim069 | distillation | 734 | 954 | 34 | 141 | 45 | 1:150/165 2:59/68 3:152/165 4:52/53 5:25/36 6:104/128 7:5/74 8:54/100 9:133/165 |
| aiim070 | distillation | 798 | 953 | 5 | 4 | 146 | 1:109/165 2:50/68 3:149/164 4:50/53 5:36/36 6:111/128 7:73/74 8:92/100 9:128/165 |
| aiim071 | distillation | 1,626 | 1,706 | 5 | 0 | 75 | 1:214/226 2:197/226 3:299/300 4:75/75 5:72/75 6:266/269 7:149/151 8:86/108 9:268/276 |
| aiim072 | distillation | 1,534 | 1,708 | 6 | 19 | 149 | 1:193/226 2:198/226 3:270/301 4:76/76 5:72/75 6:252/269 7:146/151 8:95/108 9:232/276 |
| aiim073 | distillation | 1,281 | 1,715 | 10 | 140 | 284 | 1:179/230 2:172/226 3:204/301 4:58/76 5:44/63 6:195/250 7:130/151 8:59/102 9:240/316 |
| aiim074 | distillation | 1,584 | 1,715 | 9 | 1 | 121 | 1:214/230 2:211/226 3:297/301 4:73/76 5:60/63 6:237/250 7:148/151 8:77/102 9:267/316 |
| aiim075 | distillation | 1,580 | 1,665 | 3 | 1 | 81 | 1:200/226 2:207/227 3:294/301 4:76/77 5:64/65 6:245/255 7:114/119 8:75/78 9:305/317 |
| aiim076 | distillation | 1,185 | 1,648 | 15 | 359 | 89 | 1:207/226 2:199/227 3:118/301 4:67/77 5:24/65 6:185/255 7:57/119 8:51/78 9:277/300 |
| aiim077 | selection | 1,080 | 1,505 | 33 | 124 | 268 | 1:115/159 2:120/227 3:190/281 4:65/76 5:64/76 6:250/301 7:48/105 8:98/112 9:130/168 |
| aiim078 | selection | 1,275 | 1,494 | 7 | 28 | 184 | 1:144/160 2:175/227 3:221/281 4:70/76 5:61/76 6:273/301 7:95/105 8:95/112 9:141/156 |
| aiim079 | distillation | 1,023 | 1,535 | 10 | 1 | 501 | 1:147/229 2:140/229 3:241/301 4:59/76 5:59/77 6:140/205 7:37/74 8:55/75 9:145/269 |
| aiim080 | distillation | 1,278 | 1,532 | 0 | 78 | 176 | 1:186/228 2:144/229 3:253/301 4:70/76 5:67/76 6:172/204 7:68/74 8:75/75 9:243/269 |
| aiim081 | distillation | 1,413 | 1,667 | 7 | 0 | 247 | 1:172/227 2:168/226 3:280/301 4:61/76 5:66/73 6:213/240 7:128/145 8:66/78 9:259/301 |
| aiim082 | distillation | 1,110 | 1,668 | 74 | 333 | 151 | 1:171/227 2:162/226 3:174/301 4:63/76 5:58/73 6:82/240 7:112/145 8:23/79 9:265/301 |
| aiim083 | distillation | 1,391 | 1,639 | 2 | 0 | 246 | 1:188/228 2:161/226 3:263/265 4:74/76 5:72/75 6:213/224 7:144/151 8:70/125 9:206/269 |
| aiim084 | distillation | 1,568 | 1,641 | 1 | 0 | 72 | 1:218/228 2:206/226 3:255/265 4:74/76 5:74/76 6:216/224 7:146/151 8:123/125 9:256/270 |
| aiim085 | distillation | 1,014 | 1,576 | 17 | 341 | 204 | 1:170/229 2:163/227 3:17/215 4:36/74 5:43/69 6:130/237 7:95/126 8:93/98 9:267/301 |
| aiim086 | distillation | 1,049 | 1,575 | 8 | 422 | 96 | 1:200/229 2:190/227 3:34/215 4:20/74 5:48/69 6:173/237 7:96/125 8:21/98 9:267/301 |
| aiim087 | distillation | 1,433 | 1,526 | 2 | 1 | 90 | 1:197/226 2:206/226 3:283/297 4:72/77 5:75/76 6:125/131 7:79/80 8:107/112 9:289/301 |
| aiim088 | distillation | 1,203 | 1,531 | 31 | 170 | 127 | 1:193/226 2:189/226 3:108/297 4:70/77 5:69/76 6:111/131 7:82/85 8:99/112 9:282/301 |
| aiim089 | distillation | 1,274 | 1,362 | 2 | 1 | 85 | 1:208/232 2:213/231 4:71/76 5:75/76 6:203/211 7:118/126 8:99/109 9:287/301 |
| aiim090 | distillation | 1,174 | 1,362 | 3 | 0 | 185 | 1:175/232 2:186/231 4:70/76 5:69/76 6:189/211 7:120/126 8:92/109 9:273/301 |
| aiim091 | distillation | 1,380 | 1,688 | 8 | 1 | 299 | 1:192/227 2:167/226 3:246/297 4:67/76 5:65/76 6:256/301 7:129/149 8:62/89 9:196/247 |
| aiim092 | distillation | 1,382 | 1,690 | 12 | 35 | 261 | 1:177/228 2:126/226 3:227/297 4:69/76 5:69/76 6:266/302 7:136/149 8:84/89 9:228/247 |
| aiim093 | distillation | 1,325 | 1,570 | 55 | 160 | 30 | 1:214/226 2:220/227 3:73/193 4:76/76 5:66/69 6:236/241 7:130/134 8:92/102 9:218/302 |
| aiim094 | distillation | 1,331 | 1,570 | 44 | 51 | 144 | 1:200/226 2:191/227 3:168/193 4:73/77 5:61/69 6:203/241 7:119/134 8:62/102 9:254/301 |
| aiim095 | distillation | 1,664 | 1,807 | 0 | 111 | 32 | 1:221/226 2:219/227 3:286/301 4:74/76 5:75/76 6:212/258 7:99/151 8:141/150 9:337/342 |
| aiim096 | distillation | 1,722 | 1,806 | 0 | 1 | 83 | 1:218/226 2:196/227 3:294/301 4:71/76 5:73/76 6:247/258 7:148/151 8:146/149 9:329/342 |
| aiim097 | distillation | 1,295 | 1,764 | 5 | 301 | 163 | 1:193/226 2:196/226 3:47/301 4:70/76 5:66/76 6:256/293 7:116/145 8:83/121 9:268/300 |
| aiim098 | distillation | 1,619 | 1,766 | 0 | 1 | 146 | 1:211/226 2:168/226 3:279/301 4:73/76 5:70/76 6:278/293 7:143/146 8:117/121 9:280/301 |
| aiim099 | selection | 1,213 | 1,783 | 12 | 279 | 279 | 1:172/240 2:146/226 3:190/303 4:28/75 5:43/76 6:154/274 7:99/124 8:79/120 9:302/345 |
| aiim100 | selection | 1,562 | 1,752 | 3 | 5 | 182 | 1:211/240 2:206/226 3:288/302 4:66/76 5:71/76 6:248/273 7:110/123 8:103/119 9:259/317 |
| aiim101 | distillation | 1,407 | 1,452 | 14 | 22 | 9 | 1:222/227 2:240/240 4:75/77 5:76/76 6:283/287 7:137/152 8:91/92 9:283/301 |
| aiim102 | distillation | 983 | 1,450 | 12 | 235 | 220 | 1:179/226 2:132/241 4:74/76 5:74/76 6:267/288 7:80/151 8:83/91 9:94/301 |
| aiim103 | distillation | 1,371 | 1,757 | 7 | 0 | 379 | 1:185/226 2:154/226 3:243/301 4:59/76 5:63/76 6:193/235 7:93/129 8:110/142 9:271/346 |
| aiim104 | distillation | 503 | 1,757 | 5 | 1163 | 86 | 1:13/226 2:123/226 3:42/301 4:38/76 5:66/76 6:54/235 7:22/129 8:89/142 9:56/346 |
| aiim105 | distillation | 1,331 | 1,617 | 22 | 124 | 140 | 1:158/226 2:169/226 3:155/281 4:57/59 5:59/62 6:291/301 7:121/122 8:89/101 9:232/239 |
| aiim106 | distillation | 1,249 | 1,614 | 19 | 46 | 300 | 1:161/226 2:176/226 3:242/280 4:51/59 5:54/61 6:273/301 7:97/121 8:75/101 9:120/239 |
| aiim107 | selection | 1,319 | 1,504 | 56 | 0 | 129 | 1:196/226 2:106/121 3:247/301 4:74/76 5:75/76 6:171/171 7:119/121 8:94/110 9:237/302 |
| aiim108 | selection | 969 | 1,504 | 0 | 0 | 535 | 1:105/226 2:83/121 3:200/301 4:40/76 5:46/76 6:124/171 7:84/121 8:86/110 9:201/302 |

</details>

### Steps

1. `export_crops_video.py` and the evaluation script (calibrated error per person, bootstrap, paired difference, gap per band), tested on a recorded session of ours.
2. AVCAffe pilot, then all 106 people, one job at a time on the Mac mini.
3. Once RT-GENE is confirmed: commit the split, then crop the distillation half.
4. Measure v3 and the teacher on the test sets: the baseline.
5. Train R1 (`distill_gaze_r1.py`, started 2026-11-26 20:52 on the Mac mini: from v3's best checkpoint, 30 epochs at most, patience 8, 500 batches an epoch). Checkpoints and logs go on the HARD DISK.
   - **Stop on 2026-11-27 04:20, during epoch 5 (exit 137, SIGKILL).** Cause, established: a memory leak in the trainer. `itertools.cycle` over the two real-crop loaders keeps every batch it yields, so the process grew by each batch. The Mac mini's own monitor (`~/jobqueue/monitor.log`, one line a minute) shows swap rising steadily from the start, from 2.2 GB at 20:00 to 42.1 GB at 04:00, about 5 GB an hour. At 04:20 swap fell back to 4.3 GB and free memory rose from 27% to 79%: the system killed the process when memory ran out.
   - Fabio's `token_lm` run started at 04:05 and used about 0.7 GB; the growth had begun seven hours before it. The system log has no jetsam or memorystatus entry between 04:00 and 04:30.
   - Fix: commit `efa378a`, which re-iterates the loaders instead of caching them. The 4 finished epochs are unaffected: the leak only held memory and changed no computation.
   - R1 resumed on 2026-11-27 at 16:08 from `last.pt` after epoch 4, alone on the Mac mini, once Fabio's run had logged "ALL DONE". `last.pt` is written after every epoch, so a stop costs at most one epoch.
   - Selection gap to the teacher, mean over the 20 AVCAffe selection people: v3 3.93°, then 3.34° (epoch 1), 3.22°, 3.19°, 3.16° (epoch 4). This is the selection measure, not a test result.
6. Measure R1 and apply the criterion. Write the results here.

## R2: ETH-XGaze (when the link arrives)

With a new rule, committed before use.

- The 224 version gives normalized face patches. The tracking landmarks run on each patch and Iride's box is cut from it.
- The official test people are the external test.

GazeCapture is not awaited. If it arrives, it gets its own rule.
