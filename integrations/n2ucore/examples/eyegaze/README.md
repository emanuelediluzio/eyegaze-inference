# Gaze Direction → Screen Point of Gaze

Two separate nodes connected through Flow:

```text
CvCamera.VideoStream → GazeDirection.VideoStream
GazeDirection.GazeDirection → ScreenPointOfGaze.GazeDirection
```

| Node | Class | Input | Output |
|---|---|---|---|
| Gaze Direction | `N2uCore.dyn.eyegaze.GazeDirection` (`MlDevice`) | RGB face video | `GazeDirection`: model yaw/pitch, image-plane head roll, unit 3D gaze vector, status and timestamp |
| Screen Point of Gaze | `N2uCore.dyn.screenpointofgaze.ScreenPointOfGaze` (`SwDevice`) | Gaze direction | `ScreenPoint`: X/Y in pixels, calibrated screen dimensions and status |

The model runs **only in the first node**. The second loads neither PyTorch nor MediaPipe and does not open a camera: it applies a lightweight polynomial transformation and manages calibration. Gaze Direction can run on its own. Screen Point of Gaze requires the first node’s output, following the `gaze-direction.v1` contract.

Both channels use TEXT with UTF-8 JSON and are readable through Iride’s text views. They do not use the Tobii GAZE type: that format represents pupil measurements in millimetres and scene-video coordinates, rather than webcam gaze angles or screen pixels.

## Target computer configuration

Install the integration using the [integration README](https://github.com/emanuelediluzio/eyegaze-inference/blob/main/integrations/n2ucore/README.md).
Assign the three nodes to a SysInit with the required ML dependencies. Import
`gazedino.iride.json` and choose the correct SysInit; the example uses `SysInit`.
Webcam ID `0` selects the first camera on the computer executing the node;
`1`, `2`, etc. select other cameras. Device ordering can differ between machines.
Windows uses DirectShow and Linux uses V4L2; check camera permissions.

The node always loads `N2uCore/res/models/gazedino/model.joblib` in the target framework. Device selection is automatic; inference is capped at 15 FPS. DINOv2 source must be provisioned in the local Torch Hub cache (`facebookresearch_dinov2_main/hubconf.py`); runtime never downloads missing source. The model and inference settings are fixed in code, not user parameters.

Screen Point of Gaze always starts without calibration and publishes `uncalibrated` with `screen_xy_px: null`. Calibration belongs to the current node process; there is no calibration-path setting or file import/export.

After installation, restart services using the device’s normal procedure and
reload Iride without cache. Do not interrupt active pipelines.

## Guided calibration in Iride

Open the **Screen Point of Gaze** card while the pipeline is running:

- **Calibration points** selects either a 16-point 4×4 grid or a 32-point 8×4 grid across the Iride browser area. Follow the targets while keeping your head still. Collection lasts 2.8 s per target and discards the first 0.98 s. The panel advances every 3 s.
- The backend requires at least three fresh predictions per target, filters outliers and computes a cubic fit using at least ten targets and a full-rank matrix. The guided procedure collects all selected targets, with a maximum of 32. Missing faces or a degenerate fit cause the command to fail without replacing the previous calibration.
- **Show point on page** enables the dot inside the Iride page after calibration.
- **Try fullscreen** expands Iride across the monitor; calibrate after the viewport changes.
- **Esc / Cancel**, window resizing, a hidden page or loss of the node cancels the procedure. Loss of the gaze-direction stream also cancels collection on the backend.

Control names in this guide describe the corresponding panel actions in English. This documentation update does not change the installed UI language.

Guided targets use **CSS pixels within the Iride area**. Output coordinates use the same reference; they are not global desktop coordinates. To apply the output across an entire monitor, use a fullscreen viewport that remains stable throughout calibration and operation, or provide an explicit origin/scale transformation. Do not move the camera or change your seating position after calibration.

The Gaze Direction card displays video, the face box, eye centres and gaze arrows. The Screen Point of Gaze card manages calibration and draws the dot within Iride, including fullscreen mode. Technical data remains available in a collapsible section and through JSON streams. The dot cannot overlay other desktop applications.

### Session-only calibration

The guided procedure fits coefficients in memory for the current process. Restarting the pipeline requires a new calibration. Closing or reopening a card keeps that same session and does not restart the node. Cancelling an incomplete replacement preserves the current session’s previous calibration.

## Data contracts

Direction (`GazeDirection`, TEXT):

```json
{"schema":"gaze-direction.v1","status":"ok","yaw_rad":0.2,"pitch_rad":-0.1,"yaw_deg":11.459,"pitch_deg":-5.730,"roll_rad":0.05,"roll_deg":2.865,"roll_source":"head_landmarks","direction_xyz":[0.197677,0.099833,-0.975170],"coordinate_space":"camera","bbox_xyxy_px":[100,80,300,320],"confidence":null,"source_ts_us":1792762000000000,"dropped_frames":12}
```

Head roll is computed from MediaPipe Face Mesh landmarks 33 and 263 as `atan2(dy_pixels, dx_pixels)`, with clockwise-positive image coordinates. It is separate from the two learned GazeDINO outputs, and null if a valid eye axis is unavailable. No neural-network retraining is involved.

Vector: `[sin(yaw)·cos(pitch), -sin(pitch), -cos(yaw)·cos(pitch)]`, using the trained model’s convention. No probability is fabricated: this is a regression model rather than a classifier.

Screen point (`ScreenPoint`, TEXT):

```json
{"schema":"screen-point-of-gaze.v1","coordinate_space":"screen_pixels","source_ts_us":1792762000000000,"status":"ok","screen_xy_px":[520,290],"screen_size_px":[1000,600]}
```

Both outputs preserve the capture timestamp. The first node performs inference on an ML thread with one frame in flight; it drops excess frames and the frame that triggered weight loading. The second rejects repeated, out-of-order or stale timestamps (fixed 2-second maximum age), nonfinite values and incompatible formats.

The statuses `no_face`, `model_ready`, `model_unavailable`, `error`, `input_disconnected`, `stale_input`, `invalid_input` and `uncalibrated` do not carry usable coordinates. Consumers must not repeat the last point after the stream stops; they must check freshness themselves.

## Verification

```bash
python -m pytest tests/dyn/test_eyegaze.py tests/dyn/test_screenpointofgaze.py tests/dyn/wrk/test_mldevice.py -q
node --test tests/web/screen-gaze-panel.test.mjs
python tools/verify_eyegaze_iride.py --fixture /path/to/face.png
```

The integration test starts isolated services and a test video. It checks the chain `Iride WS → uinode → sysinit → camera → direction → projection → Iride WS`, timestamps, uncalibrated status, fresh guided target collection and cancellation without importing calibration files. It then stops the pipeline and waits for IDLE. It does not use a physical webcam or modify existing pipelines. The replay image cannot establish pointing accuracy or fit a meaningful multi-target calibration. Python tests separately verify fitting and projection on synthetic data; real-subject calibration must still be tested interactively.

Earlier local tests also verified transport through uinode and sysinit using a test video and synthetic calibration. Repeat the integration test on the target device; it does not replace accuracy validation with real people.

## Remaining validation and review

The checkpoint’s 3.24° figure is the **frame-level** validation error reported by the training repository, not an independent measurement on unseen subjects or in pixels on the current monitor. The calibration fit residual is not a test on new targets. Pointing accuracy still needs to be checked with real people and targets excluded from fitting, and latency needs to be measured on the final device.

Review with Daniele: contracts and integration into the company runtime, including WebSocket keep-alive while Flow connections start; some earlier tests required a reconnection. Review with Sergio: application goals, hardware, framing and accuracy requirements. No messages have been sent to either person.

The `eye-gaze` training repository and the `eyegaze-inference` desktop demo/integration repository use the same model. The two new nodes split **two stages of the same inference pipeline**. `StressGazeInference`, Tobii and BioNormWear concern instrumented acquisition or physiological estimation; they are separate from this pair of nodes.

The **Show point on page** checkbox belongs to the pipeline session. Closing, hiding or minimizing the Screen Point of Gaze card leaves the dot and subscription active when the checkbox is selected. Reopening the card restores the selection; unchecking it hides the dot. Stopping the pipeline clears the session. Stale data, a missing face, disconnection or invalid geometry temporarily hides the dot without changing the preference. Geometry changes require recalibration.

## Held-out accuracy comparison

After each 16- or 32-point calibration, use **Verify accuracy (9 points)**. Its targets differ from both calibration grids; no fit commands are sent during verification. The card retains the latest report for each mode in the same viewport and displays mean error, RMSE and P95 in CSS pixels. Use **Save results** to export JSON before stopping the pipeline. Each verification takes about 27 s.

Keep the setup unchanged, compare both modes, and repeat in reverse order before concluding that 32 targets improve precision. See the [comparison procedure](https://github.com/emanuelediluzio/eyegaze-inference/blob/main/integrations/n2ucore/README.md#comparing-16-and-32-calibration-points) for sampling rules and report details.
