# Eye Gaze node review

Local changes only, 28 October 2026. Automatic GPU-to-CPU recovery during inference is deferred. No model subprocess, process supervisor or live device migration has been added. `mldevice.py` and `memprovider.py` are unchanged by this review.

## Node settings and execution

Gaze Direction retains `MlDevice` / `MlTicker` and a dedicated inference thread. `maxFps` is a standard integer setting, 1–120, default 15, displayed as **Maximum inference FPS** at Beginner level. The Expert-only `device` enum remains `auto` / `cpu`. Model path, smoothing and idle timeout remain fixed. Changes apply at the next Start.

Setup checks that the fixed model path resolves to a file and reports a missing model before creating the input connection. Weight deserialization and model allocation remain lazy on the ticker thread; the startup check does not validate the artifact contents.

The limit gates dispatch of incoming camera frames; it is not a periodic 60 ms timer. Only one inference can run at a time. Excess frames are discarded, the newest available input is used, and source timestamps and generation checks are preserved. A ceiling above the camera or model throughput does not increase measured FPS. A lower ceiling reduces inference frequency, not the network's parameter count or resident weight memory.

Screen Point of Gaze remains a lightweight CPU `SwDevice`: no additional neural model and no redundant compute selector. It exposes no settings and supports one guided, session-only calibration with exactly 16 targets. There are no calibration import/export, accuracy-report download or clear-calibration controls. The independent nine-target accuracy check measures the current fit without replacing it. The calibration needs at least three fresh predictions per target in a 1.82-second sampling window; a 1 FPS ceiling cannot satisfy it. Use at least 5 FPS as the initial ceiling for calibration and verify actual throughput.

## Existing gaze channel

`TobiiGlassesTracking` registers `GazeData` using `FlowChannelType.GAZE`, declares its scene-video dependency, and publishes UTF-8 JSON. `TobiiSparkTracking` is a separate older implementation using `DATA`; it is not the reference for the dedicated GAZE type.

Both Eye Gaze outputs now use the existing **GAZE** type, and the Screen Point input declaration uses GAZE. No channel enum or binary structure has been added. Names and timestamp semantics are unchanged.

| Channel | Schema | Coordinate space | Meaning |
| --- | --- | --- | --- |
| `GazeDirection` | `gaze-direction.v1` | Camera | Model yaw/pitch, unit gaze vector; head roll from face landmarks |
| `ScreenPoint` | Existing GAZE JSON + screen-point-of-gaze.v1 fields | Normalized viewport; CSS pixels retained | gaze2d, missing pupil diameters and calibrated pixels |
| Tobii Glasses `GazeData` | Existing Tobii JSON | Normalized scene video | Scene gaze and actual pupil measurements |

Screen Point emits the existing flat GAZE JSON contract: `gaze2d: [u, v]`, `eyeleft: {pupildiameter: null}` and `eyeright: {pupildiameter: null}`. Here `u = pixelX / width` and `v = pixelY / height`, with the origin at top-left. `gaze2d_coordinate_space` identifies the calibrated browser viewport. The original CSS-pixel fields remain available to the page card. Uncalibrated, invalid, stale or disconnected samples have `gaze2d: null`. The existing `modalinference.DataConverter` consumes the output unchanged, producing the same normalized X/Y columns as a Tobii-shaped sample and NaN for unavailable pupils.

This provides positional wire compatibility with existing GAZE JSON consumers. It does not turn viewport coordinates into scene-camera coordinates or provide pupil-based physiology features. The older Tobii Spark binary DATA layout is a separate transport contract. Gaze Direction intentionally keeps its angular payload, and Screen Point still requires angular input before session calibration.

New Screen Point packets use the existing normalized GAZE decoding path; old screen recordings without `gaze2d` retain their pixel-chart fallback. Tobii pupil data and scene overlays are unchanged. The dependency remains the angular input, so the viewport point is not assigned to the webcam image. Closing a card keeps the page-point session active until its checkbox is disabled or the pipeline stops.

## Style and roll

The two nodes, `gazedinoengine.py`, `screengaze.py` and `gazeartifact.py` have module-level imports, camelCase internal names and underscore-prefixed private instance state. `estimate` is a public framework signal, following other tickers. Public JSON keys and third-party API keyword names retain their established contracts.

`GazeDinoEngine(object)` is in `N2uCore/inference/gazedinoengine.py`. It builds the existing network from standard `nn.Sequential` layers; no extra model subclass is introduced. The network still predicts only yaw and pitch. Head roll comes from MediaPipe landmarks 33 and 263, using pixel aspect ratio and the clockwise-positive image convention. Missing or degenerate landmarks produce null roll; retraining is unnecessary for this head-pose measurement. Roll is not a third learned gaze angle.

## Compute resources and unchanged consumers

`resolveComputeResources()` returns **both the selected device and its memory provider**. Gaze Direction and Voice Analyzer ML pass the same device to their engine and the returned provider to their ticker. The resolver delegates to `SystemMemoryProvider`, `CudaMemoryProvider` or the supported accelerator provider; it does not replace those classes or automatically move a loaded model after failure.

PhysioDOM ML calls `resolveComputeResources("cpu", supportedDevices=("cpu",))`, which returns a CPU device and `SystemMemoryProvider`, because its classifier is CPU-only. Its NumPy input buffer, one-packet receive policy, time windows, stride, feature history and predictions are compared against `physiosympdom.py`. No buffering or signal-processing optimization is part of this review. Voice ML, including the inherited four- and five-class variants, retains VAD, RMS gating and smoothing; device routing and regression behavior are tested without downloading audio models.

The memory provider reads available memory; it does not reserve it. Fair admission, priorities and scheduling between multiple simultaneously loading ML devices remain future framework work. This review does not claim to solve concurrent allocation races.

## Earlier engine-only measurements

Apple M1, 16 GiB unified memory, macOS 26.5.1, PyTorch 2.14.1. Each run uses a separate Python process, 5 warmup frames and 40 measured analyses of the same existing 512×512 face image. Both runs detected a face in all 40 measured frames.

| Device | Throughput | Mean analysis | P95 analysis | Peak process RSS | Peak RSS above imported runtime baseline |
| --- | --- | --- | --- | --- | --- |
| MPS | 8.62 FPS | 116 ms | 142 ms | 931 MiB | 562 MiB |
| CPU | 4.62 FPS | 216 ms | 260 ms | 1016 MiB | 705 MiB |

MPS sampled peak counters: 336 MiB of live tensors and 362 MiB driver allocation. These use unified RAM and must not be added to RSS. Average process CPU use during measured inference was about 10% of one core for MPS and 132% for CPU; 100% means one fully occupied logical core. This is not total system or GPU utilization, and no electrical power measurement was made.

The measurement includes face landmarks, crop transformation, neural inference and host result transfer. It excludes camera capture, video encoding, Flow transport, browser rendering and initial loading. Repeated-image throughput does not establish live-user latency or pointing accuracy. RSS includes the imported runtime and transient loading allocations; exact memory use varies with OS accounting, runtime and other workloads. Raw reports are in `examples/eyegaze/benchmarks/`.

```sh
PYTHONPATH=. python tools/benchmark_eyegaze.py /path/to/model.joblib /path/to/face.png --device mps --frames 40 --warmup 5 --output /tmp/gaze-mps.json
PYTHONPATH=. python tools/benchmark_eyegaze.py /path/to/model.joblib /path/to/face.png --device cpu --frames 40 --warmup 5 --output /tmp/gaze-cpu.json
```

## Frontend alignment and files for Marco

The frontend follows the existing TextInputPanel and SwarmPanel patterns: `command-run` buttons, `command-arg-input` selects, shared field toggles, inherited application font and card theme tokens. Remaining panel-specific colors were replaced with theme tokens. Labels and messages now use the framework's `Texts.get()` and the central EN/IT tables. There is no panel-local language or theme mechanism. At the default 16 px root size, buttons share the standard 13 px label, 28 px minimum height and 6 px radius. Labels use the same 0.8125 rem size as shared forms; larger numeric readings distinguish live data from controls.

Review these integration points:

- `lib/render/data-worker.js`: raw GAZE JSON to existing text-kind sessions; retain the worker's single shared DataCollector socket.
- `lib/render/decoders/gaze-decoder.js`: Tobii data, gaze angles and viewport pixels remain separate schemas.
- `lib/ui/forms/gaze-direction-panel.js`: existing shared camera stream, face/eye overlays, gaze arrows and collapsible technical details.
- `lib/ui/forms/screen-gaze-panel.js` and `screen-gaze-session.js`: guided calibration and page-point lifetime independent of card lifetime.
- `schema-ui.json`: standard FPS setting; no custom optional metadata or calibration files.

Paths above are relative to `N2uCore/res/web/iride/`. The real panel modules were visually checked in a local fixture preview in light/dark themes, EN/IT and desktop/narrow layouts. The reference TextInputPanel used the same computed button font, size, height and radius. No horizontal overflow was found at a 430 px viewport. The preview uses fixture data and does not replace testing with the live user and camera. The user will coordinate the later handoff to Marco; no message has been sent.

## Verification for this review

The isolated end-to-end test passed through `uinode`, `sysinit`, test video, both GAZE nodes and Iride data WebSocket. It checked channel type/schema metadata, three valid predictions, source timestamps, initially uncalibrated output, fresh guided-target collection, cancellation and shutdown to IDLE. The test now waits for actual node-alive facts before subscribing; a RUNNING pipeline status alone can precede node connection. This change is confined to the test harness. Real-person screen-point accuracy still requires an interactive test.


## Runtime failure and clock contracts

`resolveComputeResources()` returns a `(deviceName, provider)` pair on success and `None` on failure, with a framework log. Auto keeps CPU available; requesting an unavailable accelerator explicitly fails setup. The memory providers keep their `(availableBytes, totalBytes)` contract and report `(0, 0)` for an unknown budget. Zero available bytes with a positive total remains a valid exhausted device. There is no live GPU-to-CPU migration.

GazeDINO construction only stores settings. `load()` performs validation and allocation and returns `True`/`False`; `analyze()` returns a sample or `None`. Invalid artifacts, exports and calibration operations log a diagnostic and return a neutral result. The ticker and node check these results before publishing data or acknowledging a calibration command. The ML voice wrapper likewise defers its legacy analyzer initialization to `init()` on the ticker thread. PhysioDOM ML keeps the original `Dominance` channel and its original buffering and prediction algorithm.

Screen Point uses the inherited `SwDevice` start exactly once. Its freshness check preserves producer timestamps and rejects duplicates, out-of-order samples, excessive processing age and increasing transport backlog. It estimates the producer-to-local clock offset from monotonic arrival observations instead of comparing wall clocks across computers. Guided targets are timed in the producer's timestamp domain, with local settling time retained. Gaze Direction includes `source_age_secs`, measured between receipt of its camera frame and completion of its ticker work.

A constant offset between hosts is supported. The minimum observed offset is updated before checking transport backlog, so a delayed first packet followed by ordinary traffic, or a forward source-clock jump, cannot permanently latch `stale_input`. Duplicate or decreasing timestamps, excessive processing age and increasing backlog are still rejected. An arbitrarily delayed first packet cannot be distinguished from a clock offset without a synchronized clock or a transport time bound. Camera-to-direction transport before the frame reaches Gaze Direction is outside `source_age_secs`. Backward clock jumps and slow clock drift are not resynchronized by this bounded correction; reconnecting resets the input reference. This change does not implement distributed clock synchronization. Browser page freshness and viewport checks still apply, and a camera/display/window change still requires recalibration.


## Checklist reconciliation and real Mac mini pipeline measurements

Internal Python node state uses underscore-prefixed camelCase; framework signals remain public. Imports are at module level. Optional Torch and FunASR dependencies load when needed through importlib, preserving CPU-only resource queries without Torch. GazeDINO remains an object engine with a standard sequential network and landmark head roll. Model selection and idle behavior are fixed for Eye Gaze; maxFps is configurable at the user's later explicit request. Calibration is guided and session-only.

The frontend uses Card/CardBoard/DataSocket and shared control/theme conventions. Its files and contracts are in the integration overlay for a later handoff to Marco; no external message was sent. PhysioDOM buffering and prediction policy are unchanged. Multi-device ML scheduling is future work; live GPU-to-CPU recovery remains excluded.

The real Iride pipeline was measured on the Mac mini M1 (16 GiB unified memory, eight logical CPUs, macOS 26.3, PyTorch 2.14.1) on 28 October 2026. Iride and the physical webcam were running. The user started the pipeline once with `auto` and once with `cpu`; the loaded model parameters were verified on `mps:0` and `cpu`, respectively. The FPS ceiling remained 15. Each process was observed for about 60 seconds after a 20-second warmup. The counters below belong only to the Gaze Direction process, excluding the webcam, browser and service processes.

| Metric | Normal startup (`auto` → Metal) | Forced CPU (`cpu`) |
| --- | ---: | ---: |
| Mean CPU, share of whole eight-core machine | 1.99% | 14.44% |
| Peak CPU, share of whole eight-core machine | 2.44% | 16.00% |
| Mean CPU, 100% per logical core | 15.90% | 115.48% |
| Mean process physical footprint | 835.9 MiB | 774.2 MiB |
| Peak process physical footprint | 1118.3 MiB | 996.6 MiB |
| GPU queue time / observation interval | 32.27% | 0.00% |
| Peak Metal driver allocation | 362.4 MiB | 0.0 MiB |
| Mean analysis time, frames with a face | 66.79 ms | 109.94 ms |
| P95 analysis time, frames with a face | 78.78 ms | 120.14 ms |
| Frames with a detected face | 424 / 661 (64.15%) | 439 / 442 (99.32%) |
| Valid predictions per second | 7.05 | 7.31 |

These are observed operating windows, not an equal-load comparison: missed faces skip neural inference and change average resource use. GPU queue time is a per-process timing counter, not hardware-wide shader occupancy. Metal allocations overlap unified RAM and must not be added to the process footprint. Analysis timing includes face landmarks and neural analysis, excluding JPEG decode, Flow and browser latency. Temporary timing wrappers and two allocator reads per second add a small profiling overhead included in CPU use. No electrical power, camera images or gaze coordinates were recorded. The temporary production profiler was removed after measurement.

The portable results and measurement scope are in [mac-mini-m1-live-node-20261028.json](../examples/eyegaze/benchmarks/mac-mini-m1-live-node-20261028.json). The earlier [mac-m1-mps-compatibility.json](../examples/eyegaze/benchmarks/mac-m1-mps-compatibility.json) report used a repeated image outside the full pipeline and is not interchangeable with these measurements.
