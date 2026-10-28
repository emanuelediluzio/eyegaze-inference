# Gaze Direction → Screen Point of Gaze

Two separate nodes connected through Flow:

```text
CvCamera.VideoStream → GazeDirection.VideoStream
GazeDirection.GazeDirection → ScreenPointOfGaze.GazeDirection
```

| Node | Class | Input | Output |
|---|---|---|---|
| Gaze Direction | `N2uCore.dyn.eyegaze.GazeDirection` (`MlDevice`) | RGB face video | `GazeDirection`: yaw/pitch in radians and degrees, unit 3D vector, status and timestamp |
| Screen Point of Gaze | `N2uCore.dyn.screenpointofgaze.ScreenPointOfGaze` (`SwDevice`) | Gaze direction | `ScreenPoint`: X/Y in pixels, calibrated screen dimensions and status |

The model runs **only in the first node**. The second loads neither PyTorch nor MediaPipe and does not open a camera: it applies a lightweight polynomial transformation and manages calibration. Gaze Direction can run on its own. Screen Point of Gaze requires the first node’s output, following the `gaze-direction.v1` contract.

Both channels use the existing `FlowChannelType.GAZE` with UTF-8 JSON, following `TobiiGlassesTracking.GazeData`. The channel type identifies the gaze data family; the `schema` and coordinate-space metadata identify the payload. GazeDINO does not fabricate Tobii pupil measurements or `gaze2d` scene coordinates. Consumers must recognize `gaze-direction.v1` or `screen-point-of-gaze.v1`; sharing the GAZE channel type does not make these payloads interchangeable with Tobii data. Iride forwards raw GAZE JSON to the existing card sessions and decodes each schema separately.

## Target computer configuration

Install the integration using the [integration README](https://github.com/emanuelediluzio/eyegaze-inference/blob/main/integrations/n2ucore/README.md).
Assign the three nodes to a SysInit with the required ML dependencies. Import
`gazedino.iride.json` and choose the correct SysInit; the example uses `SysInit`.
Webcam ID `0` selects the first camera on the computer executing the node;
`1`, `2`, etc. select other cameras. Device ordering can differ between machines.
Windows uses DirectShow and Linux uses V4L2; check camera permissions.

The node always loads `N2uCore/res/models/gazedino/model.joblib` in the target framework. Device selection defaults to automatic; choose `cpu` in **Compute device** under Iride's Expert settings to force CPU inference. This uses the standard `device` ENUM (`auto` or `cpu`, default `auto`) and is saved with the pipeline. **Maximum inference FPS** (`maxFps`, standard `INT`, 1–120, default 15) limits frame dispatch. Change it in node settings before starting the pipeline. This is a ceiling: camera rate and model speed can lower the actual output rate. The node processes only the latest available frame with one inference in flight; it does not queue old frames or run on a fixed 60 ms timer. DINOv2 source must be provisioned in the local Torch Hub cache (`facebookresearch_dinov2_main/hubconf.py`); runtime never downloads missing source. The model path, smoothing and idle timeout remain fixed in code.

Screen Point of Gaze always starts without calibration and publishes `uncalibrated` with `screen_xy_px: null`. Calibration belongs to the current node process; there is no calibration-path setting or file import/export.

After installation, restart services using the device’s normal procedure and
reload Iride without cache. Do not interrupt active pipelines.

## Guided calibration in Iride

Open the **Screen Point of Gaze** card while the pipeline is running:

- **Calibrate 16 points** runs the single standard 4×4 grid across the Iride browser area. Follow the targets while keeping your head still. Collection lasts 2.8 s per target and discards the first 0.98 s. The panel advances every 3 s.
- The backend requires at least three fresh predictions per target, filters outliers and computes a cubic fit only after all 16 targets are complete and the matrix has full rank. The backend rejects additional targets and incomplete fits. Missing faces or a degenerate fit cause the command to fail without replacing the previous calibration.
- Very low inference rates can fail the three-sample requirement. For guided calibration, start with a ceiling of at least 5 FPS and check that the actual model throughput is sufficient. A 1 FPS ceiling is useful for standalone direction monitoring, but cannot fill the current calibration sampling window.
- **Show point on page** enables the dot inside the Iride page after calibration.
- **Try fullscreen** expands Iride across the monitor; calibrate after the viewport changes.
- **Esc / Cancel**, window resizing, a hidden page or loss of the node cancels the procedure. Loss of the gaze-direction stream also cancels collection on the backend.

Control names in this guide describe the corresponding panel actions in English. Panel labels and messages follow Iride’s language through the shared EN/IT `Texts` tables.

Guided targets use **CSS pixels within the Iride area**. Output coordinates use the same reference; they are not global desktop coordinates. To apply the output across an entire monitor, use a fullscreen viewport that remains stable throughout calibration and operation, or provide an explicit origin/scale transformation. Do not move the camera or change your seating position after calibration.

The Gaze Direction card displays video, the face box, eye centres and gaze arrows. The Screen Point of Gaze card manages calibration and draws the dot within Iride, including fullscreen mode. Technical data remains available in a collapsible section and through JSON streams. The dot cannot overlay other desktop applications.

### Session-only calibration

The guided procedure fits coefficients in memory for the current process. Restarting the pipeline requires a new calibration. Closing or reopening a card keeps that same session and does not restart the node. Cancelling an incomplete replacement preserves the current session’s previous calibration.

## Data contracts

Direction (`GazeDirection`, GAZE):

```json
{"schema":"gaze-direction.v1","status":"ok","yaw_rad":0.2,"pitch_rad":-0.1,"yaw_deg":11.459,"pitch_deg":-5.730,"roll_rad":0.05,"roll_deg":2.865,"roll_source":"head_landmarks","direction_xyz":[0.197677,0.099833,-0.975170],"coordinate_space":"camera","bbox_xyxy_px":[100,80,300,320],"confidence":null,"source_ts_us":1792762000000000,"dropped_frames":12}
```

Head roll is computed from MediaPipe Face Mesh landmarks 33 and 263 as `atan2(dy_pixels, dx_pixels)`, with clockwise-positive image coordinates. It is separate from the two learned GazeDINO outputs, and null if a valid eye axis is unavailable. No neural-network retraining is involved.

Vector: `[sin(yaw)·cos(pitch), -sin(pitch), -cos(yaw)·cos(pitch)]`, using the trained model’s convention. No probability is fabricated: this is a regression model rather than a classifier.

Screen point (`ScreenPoint`, GAZE):

```json
{"schema":"screen-point-of-gaze.v1","coordinate_space":"screen_pixels","source_ts_us":1792762000000000,"status":"ok","screen_xy_px":[520,290],"screen_size_px":[1000,600]}
```

Both outputs preserve the capture timestamp. The first node performs inference on an ML thread with one frame in flight; it drops excess frames and the frame that triggered weight loading. The second rejects duplicate/out-of-order timestamps, processing age or increasing transport backlog above 2 seconds, nonfinite values and incompatible formats. Its clock-offset estimate uses local monotonic arrival time; hosts do not need identical wall clocks. See [runtime and clock contracts](../../docs/eyegaze-node-review.md#runtime-failure-and-clock-contracts) for first-packet and camera-transport limits.

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

## Held-out accuracy verification

After the standard 16-point calibration, use **Verify accuracy (9 points)**. This is an independent accuracy check, not a second calibration mode. Its targets differ from the calibration grid; no fit commands are sent during verification. The card displays the latest mean error, RMSE and P95 for the current calibration and viewport in CSS pixels. Each verification takes about 27 s.

Results stay in memory for the current pipeline session and are cleared when a new calibration is applied. There are no file upload or download controls for calibration or validation results. Keep the setup unchanged and repeat the verification to check consistency. See the [verification procedure](https://github.com/emanuelediluzio/eyegaze-inference/blob/main/integrations/n2ucore/README.md#checking-calibration-accuracy) for sampling rules and report details.

## Local performance and framework review

See [the local audit](../../docs/eyegaze-node-review.md) for the CPU/MPS measurements, channel compatibility checks and the frontend handoff for Marco. `tools/benchmark_eyegaze.py` measures the engine with an existing RGB image, without opening a webcam. Automatic device selection happens before model loading; live GPU-to-CPU recovery is not implemented.


Screen Point also emits standard GAZE JSON: gaze2d contains normalized viewport X/Y (pixelX / width, pixelY / height), with (0, 0) at top-left. Pupil diameters are null because this model does not measure them; unavailable points are null. The page UI retains screen_xy_px and screen_size_px. Existing GAZE consumers can read gaze2d without a new converter. The coordinate reference is the calibrated browser viewport, not the scene-camera image.
