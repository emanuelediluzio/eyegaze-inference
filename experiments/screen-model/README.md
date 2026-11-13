# Screen model for the gaze calibration: tested, not adopted

**Status (2026-11-13): not adopted.** The screen model is correct, but on our recordings it does not beat the polynomial calibration where the problems are (centre, top corners). The script stays here so it can be reused once GazeDINO's vertical angles are corrected.

## Question

After calibration the top corners of the screen were hard to reach: the calibration put them toward the centre. We asked whether a physical model of the screen would do better than the polynomial calibration of `ScreenCalibration` (`N2uCore/inference/screengaze.py`).

The model reconstructs where the eyes are in front of the screen and how far away. It then intersects each gaze direction with the screen plane.

## Model

The screen is a plane with x to the right and y down, in px. The eye is at distance `D` in front of the point `(ex, ey)`. The GazeDINO angles become true angles through:

- a horizontal angle `a = sx (yaw - cx)`, with gain 1, so `D` is set by the horizontal axis;
- a vertical angle `b = gy (pitch - cy) + k (pitch - cy)^2`, where the vertical gain and a squeeze term are free.

The gaze hits the screen at `x = ex + D tan(a)`, `y = ey + D tan(b) / cos(a)`. The linear iris terms are added as in the polynomial fit.

The parameters are bounded to the limits of a real desk:

| Parameter | Range |
|---|---|
| Eye, horizontally (`ex`) | middle 60% of the width |
| Eye, vertically (`ey`) | from 0.3 H above the screen to 0.8 H down |
| Distance `D` | 0.6–4 W |
| Pitch offset `cy` | within 0.6 rad of the mean gaze |
| Vertical gain `gy` | 0.3–3 |

A first version without these bounds traded parameters against each other. It produced eyes 1500 px below the screen and a pitch offset of 10 rad, so its results were meaningless.

## Check on a perfect geometry

`python screen_model.py synthetic` builds 16 calibration targets from a known geometry. In this test GazeDINO's angles are taken as the true angles. The model recovers `D` and the eye position, with errors of 1–2 px when each target is left out of the fit.

The straight-line calibration errs by 3–38 px on the same targets, up to 93 px at the corners when the eyes are close (D = 0.8 W). The geometric effect is therefore real, and it grows as the eyes get closer to the screen.

## Recorded sessions

We used three sessions recorded on 2026-11-08: one person, an iPhone webcam, a MacBook at 1440×900 and an external screen at 1920×1080. Each session has 16 calibration targets and 9 verification targets that are never used in the fit.

Run them with `python screen_model.py sessions <dir> ...`. The files with the iris features were lost, so all models below are compared without the iris (the angles only). Errors are in px. "LOO" is the error on a calibration target left out of the fit. "check" is the error on the verification targets. Values are the mean over the three sessions.

| model | LOO all | inner | top corners | bottom corners | bottom centre | check all | centre | bottom centre | top centre |
|---|---|---|---|---|---|---|---|---|---|
| linear (before 2026-11-12) | 87 | 91 | 163 | 64 | 69 | 87 | 128 | 91 | 100 |
| x² + x²·pitch on the vertical (candidate) | 89 | **74** | **132** | 88 | 71 | **74** | **87** | 79 | **74** |
| screen model | 92 | 85 | 144 | **62** | **68** | 88 | 105 | **77** | 92 |

On the real sessions the fit always ends on a desk limit:

| Session | Eyes | Vertical gain | Distance |
|---|---|---|---|
| 1 | at the corner of the allowed area | 2.4 | 1.9 W |
| 2 | at the bottom edge of the screen (`ey = 0.8 H`) | 1.2 | 1.1 W |
| 3 | at the bottom edge of the screen (`ey = 0.8 H`) | 1.2 | 1.0 W |

The distance is plausible in all three sessions.

## Conclusion

- **The curvature is not geometric.** The curvature in the data cannot be produced by the geometry of a person sitting in front of the screen. To fit it, the model has to move the eyes to the edge of what is physically allowed.
- **It comes from GazeDINO.** GazeDINO's vertical angles are wrong. They are 1.2–2.4 times smaller than the real movement, and more so looking up. A geometric model cannot correct this.
- **The screen model is not adopted.** It wins only at the bottom (bottom corners, bottom centre). At the centre and at the top corners the polynomial candidate is better.
- **It may still help later.** Once GazeDINO's vertical angles are corrected for the person (adapting the network head on the calibration images, under evaluation), the geometric error becomes what is left. It matters most when the eyes are close to the screen, and the screen model can then be tried again.

## Files

- `screen_model.py` contains the screen model, the polynomial fit as in `ScreenCalibration`, the check on a perfect geometry and the comparison on recorded sessions. It needs only numpy and scipy.
- The sessions are not in the repo: they are personal recordings, kept locally in `~/n2u_training_data/gazecal-20261108/`.
