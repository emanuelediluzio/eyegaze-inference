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

The node always loads `N2uCore/res/models/gazedino/model.joblib` in the target framework. Computation Device is the standard `device` ENUM (`CPU` or `GPU`, default `GPU`), saved with the pipeline; `GPU` picks CUDA, XPU or MPS in half precision and falls back to CPU. There is no FPS setting: the node sends one frame at a time to the model and drops the frames that arrive meanwhile, so the model sets the rate. DINOv2 source must be provisioned in the local Torch Hub cache (`facebookresearch_dinov2_main/hubconf.py`); runtime never downloads missing source. The model path, smoothing and idle timeout remain fixed in code.

Screen Point of Gaze always starts without calibration and publishes `uncalibrated` with `screen_xy_px: null`. Calibration belongs to the current node process; there is no calibration-path setting or file import/export.

After installation, restart services using the device’s normal procedure and
reload Iride without cache. Do not interrupt active pipelines.

## Guided calibration in Iride

Open the **Screen Point of Gaze** card while the pipeline is running:

- **Calibrate 16 points** runs the single standard 4×4 grid across the Iride browser area. Follow the targets while keeping your head still. Collection lasts 2.8 s per target and discards the first 0.98 s. The panel advances every 3 s.
- The backend requires at least three fresh predictions per target, filters outliers and fits only after all 16 targets are complete: the polynomial degree (1–3) with the lowest leave-one-out error wins, on the GazeDINO angles plus a light iris correction (iris position and eye openness, weight 0.25, ridge on the iris terms only). The backend rejects additional targets and incomplete fits. Missing faces or a degenerate fit cause the command to fail without replacing the previous calibration.
- A very slow model (for example on CPU) can fail the three-sample requirement per target.
- **Show point on page** enables the dot inside the Iride page after calibration.
- **Try fullscreen** expands Iride across the monitor; calibrate after the viewport changes.
- **Esc / Cancel**, window resizing, a hidden page or loss of the node cancels the procedure. Loss of the gaze-direction stream also cancels collection on the backend.

Control names in this guide describe the corresponding panel actions in English. Panel labels and messages follow Iride’s language through the shared EN/IT `Texts` tables.

Guided targets use **CSS pixels within the Iride area**. Output coordinates use the same reference; they are not global desktop coordinates. To apply the output across an entire monitor, use a fullscreen viewport that remains stable throughout calibration and operation, or provide an explicit origin/scale transformation. Do not move the camera or change your seating position after calibration.

The Gaze Direction card displays the video with the gaze points, the estimate quality and the head position hints. The Screen Point of Gaze card manages calibration and draws the dot within Iride, including fullscreen mode. Technical data remains available in a collapsible section and through JSON streams. The dot cannot overlay other desktop applications.

### Session-only calibration

The guided procedure fits coefficients in memory for the current process. Restarting the pipeline requires a new calibration. Closing or reopening a card keeps that same session and does not restart the node. Cancelling an incomplete replacement preserves the current session’s previous calibration.

## Data contracts

Direction (`GazeDirection`, GAZE, schema `gaze-direction.v1`), a real sample from the integration test:

```json
{"status":"ok","yaw_rad":-0.0003,"pitch_rad":-0.0635,"bbox_xyxy_px":[225,64,337,166],"eyes_px":[[254.5,94.9],[308.2,96.9]],"eye_center_px":[281.6,96.9],"eye_span_px":75.0,"iris":[-0.011,-0.084,0.255],"quality":{"score":null,"hint":"move_closer","face_size_ratio":0.175,"jitter_deg":null}}
```

`iris` is the iris position between the inner and outer eye corner (x, y, in eye widths, mean of both eyes) and the eye openness. `GazePoints` (`POINTS_2D`) carries the eye midpoint and the gaze direction projected on the image, drawn by the card.

Screen point (`ScreenPoint`, GAZE, schema `screen-point-of-gaze.v1`):

```json
{"status":"ok","quality":{"score":82,"hint":"ok","face_size_ratio":0.31,"jitter_deg":0.4},"screen_xy_px":[520,290],"screen_size_px":[1000,600],"gaze2d":[0.52,0.483],"calibration_drift":{"drifted":false,"hint":"ok","dx":0.02,"dy":-0.01,"scale":1.03,"reference":[640,300,180],"current":[643.6,298.2,185.4]},"frame_aspect":0.5625}
```

Both outputs keep the timestamp and packID of the camera frame. The statuses `no_face`, `model_unavailable`, `error` and `uncalibrated` carry no coordinates (`screen_xy_px` and `gaze2d` are null). Consumers must not repeat the last point after the stream stops.

## Verification

```bash
python -m pytest tests/dyn/test_eyegaze.py tests/dyn/test_gaze_quality.py tests/dyn/test_screen_calibration.py tests/dyn/test_screenpointofgaze.py -q
python tools/verify_eyegaze_iride.py --fixture /path/to/face.png --address <LAN address>
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
