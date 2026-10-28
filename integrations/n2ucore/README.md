# Eye Gaze — Next2U Core / Iride integration

This guide is for developers integrating the two nodes into an existing **N2uCore** checkout and using them from **Iride**. The [main README](../../README.md) also covers the standalone Python desktop application; this guide covers the framework package.

## Start here: joblib and node source files

| What you need | Location | Destination / purpose |
|---|---|---|
| **Minimal `gazedino_inference.joblib` model** | **[Direct GitHub download](https://github.com/emanuelediluzio/eyegaze-inference/releases/download/inference-v1/gazedino_inference.joblib)** · **[Direct Gitea download](http://192.168.70.200:3000/attachments/8a538e7d-0727-4b6b-81d3-77c27c6deb06)** · [release details](http://192.168.70.200:3000/ediluzio/eyegaze-inference/releases/tag/inference-v1) | Copy and rename to **`N2uCore/res/models/gazedino/model.joblib`** in the target framework |
| **Gaze Direction node** | [N2uCore/dyn/eyegaze.py](N2uCore/dyn/eyegaze.py) | `GazeDirection(MlDevice)` and `EyeGazeTicker(MlTicker)`: video to yaw/pitch, MediaPipe head roll and a 3D vector |
| **Screen Point of Gaze node** | [N2uCore/dyn/screenpointofgaze.py](N2uCore/dyn/screenpointofgaze.py) | `ScreenPointOfGaze(SwDevice)`: gaze direction to a screen point after calibration |
| **Complete installation** | [install.py](install.py) · [framework-hooks.json](framework-hooks.json) | Installs engines, catalogue entries, Iride panels and the required framework hooks |
| **Example pipeline** | [examples/eyegaze/gazedino.iride.json](examples/eyegaze/gazedino.iride.json) | Import into Iride and select the target device’s SysInit and webcam |

**The joblib is a release attachment and is not included in a Git clone.** Its size is **324,174,954 bytes** (about 324 MB / 309 MiB), approximately **69% smaller** than the 1,044,671,152-byte training checkpoint.

SHA-256 of the joblib in release `inference-v1`:

```text
d265ced7c0840c5f8f7d8880b2876d52b27313f337651e575f3975f8cd319cf4
```

For a working integration, use **the complete contents of this directory**, not just the two `dyn/*.py` files: the nodes depend on the inference engines, catalogue entries and Iride panels listed below. The installer places these components in the framework’s existing directory hierarchy.

## 1. Install into the existing framework

Prerequisites: an N2uCore checkout with `MlDevice`, `MlTicker`, `WorkerFlowInput`, the dynamic node catalogue and Iride. Its existing `memprovider.py` must already expose `resolveComputeResources()` and `releaseComputeCache()`. Framework ML changes are a separate prerequisite and are not included or installed by this Eye Gaze package. Use the Python environment configured in the **SysInit that will execute the nodes**.

From the root of the `eyegaze-inference` repository, with the pipeline stopped:

```sh
python -m pip install -r integrations/n2ucore/requirements.txt
python integrations/n2ucore/install.py "/path/to/n2ucore"
python integrations/n2ucore/install.py "/path/to/n2ucore" --apply
```

On Windows, a target path could be `C:/repos/n2ucore`. Running without `--apply` previews the changes. Running with `--apply` checks all integration points before writing and backs up existing files under `.gaze-integration-backups/`. An upgrade also backs up and removes the previous integration engine `gazedino.py`; the implementation now lives in `gazedinoengine.py`.

The installer merges the two nodes into the existing catalogues while preserving other entries. The JSON files under `N2uCore/res/` are **fragments to merge**, not complete replacement catalogues. Shared Card/CardBoard hooks, panel registration are declared in `framework-hooks.json` and `install.py`. If the framework is incompatible, preflight stops without writing. Repeating the installation against the same code produces no further changes.

The installer does not start services, install weights or change the webcam configuration. This directory is a framework overlay: do not add it to `PYTHONPATH` as an alternative framework.

## 2. Place the joblib

Download the model using the links above. Create the target directory in the **destination N2uCore checkout** and place the file at:

```text
N2uCore/res/models/gazedino/model.joblib
```

The model path is fixed at `models/gazedino/model.joblib`, resolved through the framework resource manager. There is no model selector, model-path parameter or per-node tuning panel. Old saved settings do not override these fixed values.

DINOv2 source must already be available in the local Torch Hub cache at `facebookresearch_dinov2_main`, containing `hubconf.py`. Runtime loading is local-only and fails with a clear error if the cache is missing. `pretrained=False` is used: backbone weights are already included in the joblib.

## 3. Run in Iride

The pipeline is:

```text
WebCam.VideoStream → GazeDirection.VideoStream
GazeDirection.GazeDirection → ScreenPointOfGaze.GazeDirection
```

Import the [portable pipeline](examples/eyegaze/gazedino.iride.json), select the correct SysInit and verify the webcam ID **on the computer executing WebCam**. The ID depends on device enumeration on that machine.

The direction node uses the fixed minimal joblib above, EMA alpha 0.25 and a 600-second idle unload. Its two standard settings are the maximum inference FPS (1–120, default 15) and Expert compute device (`auto` or `cpu`). Automatic selection uses the engine’s declared CUDA/XPU/MPS/CPU capability order. The screen node accepts input captured within 2 seconds, always starts uncalibrated and has no configuration fields.

After installation, restart services using the device’s normal procedure, with the pipeline stopped, and reload Iride without cache. Then start the pipeline.

Open Gaze Direction to see the video, face box, eye markers and gaze arrows; technical details are collapsible. Open Screen Point of Gaze and choose **Calibrate 16 points** to run the single standard calibration. To use the whole monitor, enter fullscreen first and calibrate in that mode.

The **Show point on page** checkbox belongs to the pipeline session: closing, hiding or minimizing the card leaves the dot active; reopening it restores the selected checkbox. Unchecking it hides the dot. Stopping the pipeline clears the session. Missing faces, stale data, disconnections and a hidden page temporarily hide the dot without changing the preference. Changes to window size, window position or display density require recalibration.

Calibration is held only for the current node process. There are no saved calibration files or import/export controls. Calibrate after starting a new pipeline; cancelling a replacement calibration keeps the current session’s previous fit. The dot is drawn inside the Iride page and cannot remain above other applications or after the browser closes.

Both cards reuse Iride’s shared command buttons (`command-run`, also used by TextInput, TextViewer and Swarm), form controls and card palette. The base font is inherited from Iride’s user preferences; internal headings use the standard card-title size. Larger numeric direction readings are reserved for the live measurements. Card colours follow the framework in both light and dark canvas themes. The layout adapts to the card width, with direction metrics moving below the video in narrow cards. No separate button theme is bundled.

Control names in this guide describe the corresponding panel actions in English. This documentation update does not change the installed UI language.

For channel contracts and session calibration details, see [examples/eyegaze/README.md](examples/eyegaze/README.md).

## Integration files and directory hierarchy

These paths are relative to this directory and match their destinations in the N2uCore checkout:

```text
N2uCore/
  dyn/eyegaze.py                         ML node and dedicated ticker
  dyn/screenpointofgaze.py               Projection and calibration node
  inference/gazedinoengine.py            Object-based engine, composed network and face detection
  inference/gazeartifact.py              Joblib contract, loader and exporter
  inference/screengaze.py                Calibration fitting and projection
  res/dyndefs.json                       Catalogue fragment for the two nodes
  res/web/iride/schema-ui.json           Iride settings fragment
  res/web/iride/lib/ui/forms/
    gaze-direction-panel.js             Video and direction card
    screen-gaze-panel.js                 Calibration card and controls
    screen-gaze-session.js               Dot independent of card visibility
    panel-sessions.js                    Pipeline-owned panel state
    gaze-ui.js                          Shared DataSocket views and helpers
  res/web/iride/css/app/gaze-panels.css   Card and dot styles
examples/eyegaze/                        Pipelines, sketches and documentation
tests/dyn/                              Python tests
tests/web/                              UI tests
tools/export_eyegaze.py                  Original checkpoint exporter
tools/verify_eyegaze_iride.py            Full transport integration test
```

**Compute device** is the existing framework `device` property, shown at Iride's Expert level as a standard string `ENUM`: `auto` (default) or `cpu`. Select `cpu` to force this model onto CPU even when a GPU is available; the value is saved in the pipeline and applies at the next Start. Normal schema defaults keep older pipelines working. Eye Gaze uses no custom `optional` metadata or validation bypass. The installer removes only the custom text-field and UI extensions introduced by earlier gaze integration work, while restoring or preserving the framework's original optional-list validation and empty-subnodes rule for standalone agents. Screen Point of Gaze is already CPU-only and has no redundant device setting.

Memory handling follows the framework provider contract used by PhysioDOM ML. The shared `resolveComputeResources()` function selects a compute device and matching memory provider. Gaze Direction and the standalone GazeDINO engine declare CUDA/XPU/MPS/CPU capabilities. The node passes the selected provider and device, fixed model resource and idle timeout into `EyeGazeTicker`; model loading remains on the ML thread. CUDA and AMD ROCm use `CudaMemoryProvider`, Intel XPU uses `AcceleratorMemoryProvider`, and CPU/Apple MPS use `SystemMemoryProvider`. Cache release uses the common `releaseComputeCache()` helper. Physical CPU/MPS inference is verified; other accelerators have simulated coverage and need hardware validation.

The shared memory implementation belongs to the destination framework at `N2uCore/dyn/wrk/memprovider.py`. This installer checks that the required resource functions exist and never replaces that file. The schema-aware `lib/render/decoders/gaze-decoder.js` source is also guarded by a known-source fingerprint; the worker receives a small hook for raw GAZE card sessions. Before replacing it, the installer checks the target against known source fingerprints or the exact installed version. Iride hooks target the current Card/CardBoard lifecycle. Unknown source changes stop preflight without writes. Shared resource handling uses native or general PyTorch memory APIs; unavailable backends or unreadable memory allow automatic CPU fallback, while genuine low memory still blocks model loading through the unchanged `MlDevice`/`MlTicker` guard. `SystemMemoryProvider` reads psutil directly across operating systems. CPU-only callers do not import Torch through the resource helpers. Model weights and physiological buffers are unchanged. Runtime failures return logged neutral results; the node and ticker check these outcomes. Screen Point uses one inherited start and estimates clock offset from monotonic arrivals for calibration and freshness. See [runtime and clock contracts](docs/eyegaze-node-review.md#runtime-failure-and-clock-contracts) for details and limits.

Gaze Direction follows the **`MlDevice` / `MlTicker`** pattern: decoding, face detection and PyTorch inference run on the ML thread, with one frame in flight and obsolete frames dropped. The joblib format does not turn the network into a tabular model. Screen Point of Gaze uses **`SwDevice`** and NumPy; it does not load the neural network. Iride communicates only with **uinode**, using the framework’s WebSocket channels; nodes communicate through Flow.

The engine is `GazeDinoEngine(object)` in `gazedinoengine.py`. A standard PyTorch `Sequential` composes the existing backbone and head; there is no nested network subclass. Checkpoint keys remain `backbone.*` and `head.*`. Internal instance state uses underscore-prefixed camelCase names, dependencies are declared explicitly at module scope, as in the framework’s voice node. Node and ticker construction do not load weights or create a face tracker; both are created by the ML ticker’s model-load hook. Existing snake_case JSON field names are retained as the channel protocol, not Python variable names.

Yaw and pitch are neural-network predictions. `roll_rad` and `roll_deg` describe head tilt measured from MediaPipe Face Mesh outer-eye landmarks 33 and 263, using pixel coordinates and clockwise-positive image angles. `roll_source` is `head_landmarks`. This is an image-plane tilt estimate, not a full 3D head-pose Euler angle, a third learned network output or ocular torsion; it does not change the yaw/pitch gaze vector or screen calibration. Missing or degenerate eye landmarks yield null roll values. No model retraining or new joblib is needed. For full 3D head pose, a separate geometry-based pose estimate is required; MediaPipe documents a [face geometry transformation pipeline](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md#face-transform-module).

## Joblib contents and export

The file is a dictionary using schema `n2u.gazedino.inference.v1`:

| Field | Contents |
|---|---|
| `format` | Artifact contract version |
| `architecture` | GazeDINO, the DINOv2 ViT-B/14 backbone and head dimensions |
| `preprocessing` | RGB, 224×224 resize and ImageNet normalization |
| `output` | Yaw/pitch order, in radians |
| `state_dict` | 183 tensors stored as NumPy float32 arrays |
| `metadata` | Source checkpoint SHA-256 and the validation error reported by that checkpoint |

It contains no optimizer, dataset, training history or serialized model-class instance. The loader reconstructs the PyTorch network and loads weights with `strict=True`. Use artifacts from trusted sources.

To create a new minimal copy, run from the **N2uCore checkout root after installation**:

```sh
python tools/export_eyegaze.py --source /path/to/best_run3_3.24deg.pt --output /path/to/model.joblib
```

The exporter reads the generated file back and compares every weight against the source; it does not overwrite an existing destination. The original `.pt` remains supported by the node for compatibility and is the format used by the desktop GUI.

## Verification and measured limits

- **183 float32 tensors identical** to the original checkpoint, with no quantization or weight changes.
- **Identical predictions on five crops** of a public image, with a maximum difference of 0 radians on MPS, verified locally and on the Mac mini. [Parity and checksum report](examples/eyegaze/inference-export-verification.json).
- **Historical full Mac mini integration test (before the local session-only calibration revision)** using a test video: WebSocket → uinode → sysinit → WebCam → Gaze Direction/MlTicker → Screen Point of Gaze → WebSocket, including projection checks and shutdown to IDLE. [Test report](examples/eyegaze/minimal-node-integration-test.json).
- Verified environment: Python 3.11, MediaPipe 0.10.21, NumPy 1.26.4 and MPS. Select torch/torchvision builds suited to the target hardware. Windows and Linux have not yet been certified through equivalent tests.

The file reduction removes training state; the network remains DINOv2 ViT-B/14. It is not evidence of a lower computational cost. A Mac mini M1 benchmark measured approximately 100–150 ms per full analysis; the configurable maximum inference FPS setting (default 15) is a ceiling. Export parity is not a new accuracy validation: pointing accuracy must be checked with the actual person and monitor, using targets excluded from calibration.

To repeat the checks, run from the **N2uCore checkout root** in its Python environment:

```sh
python -m pip install -r /path/to/eyegaze-inference/integrations/n2ucore/requirements-dev.txt
python -m pytest tests/dyn/test_gazeartifact.py tests/dyn/test_eyegaze.py tests/dyn/test_screenpointofgaze.py -q
python -m pytest tests/dyn/wrk/test_accelerator_memory.py tests/dyn/wrk/test_compute_resources.py tests/dyn/wrk/test_memprovider_portability.py tests/dyn/wrk/test_memprovider.py tests/dyn/wrk/test_mldevice.py tests/dyn/test_eyegaze.py -q
node --test tests/web/gaze-direction-panel.test.mjs tests/web/gaze-calibration.test.mjs tests/web/screen-gaze-panel.test.mjs tests/web/optional-gaze-fields.test.mjs
python tools/verify_eyegaze_iride.py --fixture /path/to/face.png
```

The full test repeats a public image in a test video and does not require a physical webcam. Test dependencies are separate from runtime dependencies.

## Checking calibration accuracy

1. Keep the same camera, monitor, browser viewport and seating position. Enter fullscreen before starting if you intend to use fullscreen.
2. Use **Calibrate 16 points** (about 48 s). This is the only calibration mode: a fixed 4×4 grid. The backend requires all 16 completed targets and rejects additional targets.
3. Run **Verify accuracy (9 points)** (about 27 s). Follow each new target without moving your head. This checks the existing calibration without fitting another one.
4. Read mean error, RMSE and P95 in the card. Lower values indicate better pointing on these held-out targets. Repeat verification with the same setup to check consistency.

The latest verification stays in memory for the current calibration and viewport, including while the card is closed. A new calibration or stopping the pipeline clears the report. Calibration coefficients are held only in the node process. There are no import/export controls or downloads for calibrations or validation reports.

Verification uses a separate 3×3 grid that does not overlap the calibration grid. The first second at each verification target is discarded; only unique, valid, timestamped samples captured after settling count, with at least three observations per target and a bound of 128.

The report records viewport geometry, calibration point count, date, per-target sample counts and errors, overall mean Euclidean error, RMSE, median and P95. Units are **CSS pixels within the browser viewport**, not physical display pixels or degrees. Mean and RMSE weight the nine targets equally; median and P95 use all accepted samples. The page dot is hidden during verification to avoid visual feedback affecting fixation and returns to its previous enabled state afterward. Cancelling, losing fresh face data or changing geometry aborts the test without saving a partial result or changing the calibration. This check does not establish accuracy on unseen users.

## Inference rate and gaze channels

**Maximum inference FPS** (`maxFps`) is a standard integer setting from 1 to 120, defaulting to 15. It limits dispatch of fresh frames, keeps one inference in flight and drops excess input. It is not a 60 ms timer. Actual throughput also depends on the camera and model. The model resource remains fixed. For guided calibration, start with at least 5 FPS; the current three-sample collection window cannot be filled at 1 FPS.

Both `GazeDirection` and `ScreenPoint` use the existing `GAZE` channel type, following Tobii Glasses. Their distinct JSON schemas retain angle versus browser-pixel semantics and do not fabricate Tobii pupil data or normalized scene gaze. Install the node, catalogue, decoder and worker changes together, restart the affected pipeline and reload Iride. Existing pipeline edges retain the same channel names; missing `maxFps` values receive the normal schema default. External consumers must dispatch by schema instead of assuming every GAZE payload is Tobii data.

See [the node review and Marco frontend handoff](docs/eyegaze-node-review.md) for source paths, compatibility checks and local CPU/MPS measurements. Benchmark tool: `tools/benchmark_eyegaze.py`; raw reports: `examples/eyegaze/benchmarks/`. This work is local only. Live GPU-to-CPU recovery and multi-model scheduling are deferred.

The gaze panels use the framework’s existing button, input and toggle classes, inherited fonts, central `Texts` EN/IT tables and theme tokens. The installer adds gaze translation entries to the existing app language tables; it does not replace those tables. The user will coordinate frontend review with Marco later.


Screen Point emits the existing GAZE JSON fields: normalized viewport gaze2d plus eyeleft/eyeright pupildiameter set to null. The existing framework gaze converter reads these coordinates without changes. Screen pixel fields and the session/page UI remain available. The normalized coordinates refer to the calibrated browser viewport; pupil measurements and scene-camera coordinates are not inferred. Gaze Direction keeps its angular schema. See docs/eyegaze-node-review.md for compatibility tests, checklist status and the latest Mac measurement.
