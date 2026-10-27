# Eye Gaze — Next2U Core / Iride integration

This guide is for developers integrating the two nodes into an existing **N2uCore** checkout and using them from **Iride**. The [main README](../../README.md) also covers the standalone Python desktop application; this guide covers the framework package.

## Start here: joblib and node source files

| What you need | Location | Destination / purpose |
|---|---|---|
| **Minimal `gazedino_inference.joblib` model** | **[Direct GitHub download](https://github.com/emanuelediluzio/eyegaze-inference/releases/download/inference-v1/gazedino_inference.joblib)** · **[Direct Gitea download](http://192.168.70.200:3000/attachments/8a538e7d-0727-4b6b-81d3-77c27c6deb06)** · [release details](http://192.168.70.200:3000/ediluzio/eyegaze-inference/releases/tag/inference-v1) | Copy and rename to **`N2uCore/res/models/gazedino/model.joblib`** in the target framework |
| **Gaze Direction node** | [N2uCore/dyn/eyegaze.py](N2uCore/dyn/eyegaze.py) | `GazeDirection(MlDevice)` and `EyeGazeTicker(MlTicker)`: video to yaw/pitch and a 3D vector |
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

Prerequisites: an N2uCore checkout with `MlDevice`, `MlTicker`, `WorkerFlowInput`, the dynamic node catalogue and Iride. Use the Python environment configured in the **SysInit that will execute the nodes**.

From the root of the `eyegaze-inference` repository, with the pipeline stopped:

```sh
python -m pip install -r integrations/n2ucore/requirements.txt
python integrations/n2ucore/install.py "/path/to/n2ucore"
python integrations/n2ucore/install.py "/path/to/n2ucore" --apply
```

On Windows, a target path could be `C:/repos/n2ucore`. Running without `--apply` previews the changes. Running with `--apply` checks all integration points before writing and backs up existing files under `.gaze-integration-backups/`.

The installer merges the two nodes into the existing catalogues while preserving other entries. The JSON files under `N2uCore/res/` are **fragments to merge**, not complete replacement catalogues. Shared Card/CardBoard hooks, panel registration and optional-field support are declared in `framework-hooks.json` and `install.py`. If the framework is incompatible, preflight stops without writing. Repeating the installation against the same code produces no further changes.

The installer does not start services, install weights or change the webcam configuration. This directory is a framework overlay: do not add it to `PYTHONPATH` as an alternative framework.

## 2. Place the joblib

Download the model using the links above. Create the target directory in the **destination N2uCore checkout** and place the file at:

```text
N2uCore/res/models/gazedino/model.joblib
```

Set Gaze Direction’s **Checkpoint path** parameter to:

```text
models/gazedino/model.joblib
```

This is the default for new nodes. Existing saved Iride projects retain their explicit settings: update this parameter if they still reference the original `.pt` file. An absolute path is also supported; it must refer to the computer executing the node.

The **DINOv2 source code** is also required. When **Local DINOv2 source directory (offline)** is empty, the engine uses the Torch cache `facebookresearch_dinov2_main`, or downloads the source if it is missing. For an offline device, prepare the cache or explicitly provide a local directory containing `hubconf.py`. Loading uses `pretrained=False`: all backbone weights are already included in the joblib.

## 3. Run in Iride

The pipeline is:

```text
WebCam.VideoStream → GazeDirection.VideoStream
GazeDirection.GazeDirection → ScreenPointOfGaze.GazeDirection
```

Import the [portable pipeline](examples/eyegaze/gazedino.iride.json), select the correct SysInit and verify the webcam ID **on the computer executing WebCam**. The ID depends on device enumeration on that machine.

| Setting | Value / meaning |
|---|---|
| Gaze Direction — Checkpoint path | `models/gazedino/model.joblib` |
| Compute device | `auto`: CUDA, MPS or CPU, depending on availability |
| Maximum inference FPS | `15`: an upper limit, not a guaranteed rate |
| Local DINOv2 source directory (offline) | Optional; cached or local source as described above |
| Screen Point of Gaze — Screen calibration JSON | Optional; without calibration, the node emits `uncalibrated` |

After installation, restart services using the device’s normal procedure, with the pipeline stopped, and reload Iride without cache. Then start the pipeline.

Open Gaze Direction to see the video, face box, eye markers and gaze arrows; technical details are collapsible. Open Screen Point of Gaze and select **16 or 32 points**, then start calibration. To use the whole monitor, enter fullscreen first and calibrate in that mode.

The **Show point on page** checkbox belongs to the pipeline session: closing, hiding or minimizing the card leaves the dot active; reopening it restores the selected checkbox. Unchecking it hides the dot. Stopping the pipeline clears the session. Missing faces, stale data, disconnections and a hidden page temporarily hide the dot without changing the preference. Changes to window size, window position or display density require recalibration.

**Load/Save calibration** are under Advanced options. The calibration JSON contains coefficients and calibrated area dimensions; it is separate from the model’s joblib. Cancelling a new calibration preserves the previous fit. The dot is drawn inside the Iride page and cannot remain above other applications or after the browser closes.

Both cards reuse Iride’s shared command buttons (`command-run`, also used by TextInput, TextViewer and Swarm), form controls and card palette. The base font is inherited from Iride’s user preferences; internal headings use the standard card-title size. Larger numeric direction readings are reserved for the live measurements. Card colours follow the framework in both light and dark canvas themes. The layout adapts to the card width, with direction metrics moving below the video in narrow cards. No separate button theme is bundled.

Control names in this guide describe the corresponding panel actions in English. This documentation update does not change the installed UI language.

For channel contracts, calibration details and JSON compatibility, see [examples/eyegaze/README.md](examples/eyegaze/README.md).

## Integration files and directory hierarchy

These paths are relative to this directory and match their destinations in the N2uCore checkout:

```text
N2uCore/
  dyn/eyegaze.py                         ML node and dedicated ticker
  dyn/screenpointofgaze.py               Projection and calibration node
  inference/gazedino.py                  PyTorch network and face detection
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

Gaze Direction follows the **`MlDevice` / `MlTicker`** pattern: decoding, face detection and PyTorch inference run on the ML thread, with one frame in flight and obsolete frames dropped. The joblib format does not turn the network into a tabular model. Screen Point of Gaze uses **`SwDevice`** and NumPy; it does not load the neural network. Iride communicates only with **uinode**, using the framework’s WebSocket channels; nodes communicate through Flow.

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
- **Full Mac mini integration test** using a test video: WebSocket → uinode → sysinit → WebCam → Gaze Direction/MlTicker → Screen Point of Gaze → WebSocket, including projection checks and shutdown to IDLE. [Test report](examples/eyegaze/minimal-node-integration-test.json).
- Verified environment: Python 3.11, MediaPipe 0.10.21, NumPy 1.26.4 and MPS. Select torch/torchvision builds suited to the target hardware. Windows and Linux have not yet been certified through equivalent tests.

The file reduction removes training state; the network remains DINOv2 ViT-B/14. It is not evidence of a lower computational cost. A Mac mini M1 benchmark measured approximately 100–150 ms per full analysis; `maxFps=15` is a ceiling. Export parity is not a new accuracy validation: pointing accuracy must be checked with the actual person and monitor, using targets excluded from calibration.

To repeat the checks, run from the **N2uCore checkout root** in its Python environment:

```sh
python -m pip install -r /path/to/eyegaze-inference/integrations/n2ucore/requirements-dev.txt
python -m pytest tests/dyn/test_gazeartifact.py tests/dyn/test_eyegaze.py tests/dyn/test_screenpointofgaze.py -q
node --test tests/web/gaze-direction-panel.test.mjs tests/web/gaze-calibration.test.mjs tests/web/screen-gaze-panel.test.mjs tests/web/optional-gaze-fields.test.mjs
python tools/verify_eyegaze_iride.py --fixture /path/to/face.png --checkpoint /path/to/model.joblib --hub-dir /path/to/dinov2
```

The full test repeats a public image in a test video and does not require a physical webcam. Test dependencies are separate from runtime dependencies.

## Comparing 16 and 32 calibration points

1. Keep the same camera, monitor, browser viewport and seating position. Enter fullscreen before starting if you intend to use fullscreen.
2. Choose **16 points**, calibrate (about 48 s), then run **Verify accuracy (9 points)** (about 27 s). Follow each new target without moving your head.
3. Choose **32 points**, calibrate (about 96 s), then repeat the same nine-point verification. The successful second fit replaces the first; save each calibration beforehand if you need both files.
4. Compare the two rows in the card. A lower mean error and RMSE indicate better pointing on these held-out targets. Repeat both modes, preferably reversing their order, to check that any improvement is consistent.
5. Use **Save results** to export the validation report as JSON. Results are kept for the pipeline session, including while the card is closed; stopping the pipeline clears them. Imported calibrations have an unknown target count and are labelled accordingly.

Calibration covers a 4×4 or 8×4 grid over the same area. Verification uses a separate 3×3 grid that does not overlap either calibration grid. It never refits the calibration. The first second at each verification target is discarded; only unique, valid, timestamped samples captured after settling count, with at least three observations per target and a bound of 128.

The report records viewport geometry, calibration mode, date, per-target sample counts and errors, overall mean Euclidean error, RMSE, median and P95. Units are **CSS pixels within the browser viewport**, not physical display pixels or degrees. Mean and RMSE weight the nine targets equally; median and P95 use all accepted samples. The page dot is hidden during verification to avoid visual feedback affecting fixation and returns to its previous enabled state afterward. Cancelling, losing fresh face data or changing geometry aborts the test without saving a partial result or changing the calibration.

This is a practical within-session comparison. It does not guarantee that 32 points will outperform 16 or establish accuracy on unseen users.
