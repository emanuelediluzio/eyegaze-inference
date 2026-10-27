import copy
import json
import time

import psutil
import torch

from N2uCore.dyn.wrk.mldevice import MlDevice, MlTicker, MemoryProvider
from N2uCore.dyn.wrk.input import WorkerFlowInput
from N2uCore.miniflow.channel import FlowChannelType
from N2uCore.miniflow.proto import FlowEvent, FlowResource
from N2uCore.signal import Signal, AttachMode
from N2uCore.resman import ResourceManager
from N2uCore.multimedia.imgdec import ImageDecoder
from N2uCore.inference.gazedinoengine import GazeDinoEngine


class GazeMemoryProvider(MemoryProvider):
    def available(self):
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return int(free), int(total)
        vm = psutil.virtual_memory()
        return int(vm.available), int(vm.total)


class EyeGazeTicker(MlTicker):
    def __init__(self):
        super().__init__(GazeMemoryProvider())
        self._engine = None
        self.estimate = Signal(self, "estimate")
        self._retryAt = 0
        self._generation = None

    def _estimateFootprint(self):
        return 700 * 1024**2

    def _loadModel(self):
        modelPath = ResourceManager().absolutePath("models/gazedino/model.joblib")
        self._engine = GazeDinoEngine(modelPath)
        try:
            self._engine.load()
        except Exception as exc:
            self._engine.close()
            self._engine = None
            self.err("Cannot load GazeDINO: {}".format(exc))
            return False
        self._modelLabel = "GazeDINO ViT-B/14"
        return True

    def _unloadModel(self):
        if self._engine is not None:
            self._engine.close()
        self._engine = None

    def infer(self, data, props, ts, generation):
        outcome = {"schema": "gaze-direction.v1", "status": "error"}
        wasLoaded = self._loaded
        try:
            if time.monotonic() < self._retryAt:
                outcome["status"] = "model_unavailable"
            elif not self._ensureLoaded():
                self._retryAt = time.monotonic() + 30
                outcome["status"] = "model_unavailable"
            elif not wasLoaded:
                # Wait for a fresh frame after loading.
                outcome["status"] = "model_ready"
            else:
                if generation != self._generation:
                    self._engine.reset()
                    self._generation = generation
                decoder = ImageDecoder()
                if props.get("pxFmt") != "UINT8_T" or props.get("depth") != 3:
                    raise ValueError("Input must be visible RGB UINT8, depth 3")
                if not decoder.setup(props["size"], 3, "UINT8_T", props["encoding"]["codec"]):
                    raise ValueError("Unsupported input codec")
                outcome = self._engine.analyze(decoder.decode(data))
        except Exception as exc:
            self.err("EyeGaze inference: {}".format(exc))
            outcome["error"] = str(exc)
            if not self._loaded:
                self._retryAt = time.monotonic() + 30
        finally:
            self.estimate.trigger([{"sample": outcome, "ts": ts, "generation": generation}])


class GazeDirection(MlDevice):
    def __init__(self):
        super().__init__()
        self._flow = self._receiver = self._output = None
        self._source = None
        self._inFlight = False
        self._generation = 0
        self._lastDispatch = 0
        self._dropped = 0

    def _onSetup(self):
        if not super()._onSetup():
            return False
        self._source = self.service().flowPack().get("VideoStream")
        if not self._source:
            self.err("EyeGaze requires a VideoStream input")
            return False
        self._flow = WorkerFlowInput(self.service())
        self._flow.setCredentials("guest", "password")
        return True

    def _tickerClass(self):
        return EyeGazeTicker

    def _tickerArgs(self):
        return {}

    def _onInit(self):
        channel = self.service().addChannel(
            "GazeDirection",
            FlowChannelType.TEXT,
            {
                "format": "json",
                "schema": "gaze-direction.v1",
                "angleUnit": "radians",
                "dependsOn": ["{}.{}".format(self._source["node"], self._source["channel"])],
                "coordinateSpace": "camera",
                "rollSource": "head_landmarks",
                "rollConvention": "atan2(eye263.y-eye33.y, eye263.x-eye33.x) in image pixels; clockwise positive",
                "vectorConvention": "[sin(yaw)*cos(pitch), -sin(pitch), -cos(yaw)*cos(pitch)]",
            },
        )
        if not channel:
            return False
        self._output = channel
        if not super()._onInit():
            return False
        self._flow.setTargetByAnnounce(self._source["node"])
        self._flow.setAutoSubscribeChannels([self._source["channel"]])
        self._flow.subscribed.attach(self._onSubscribe)
        self._flow.disabled.attach(self._onDisable)
        self._flow.eventGrabbed.attach(self._onProperties)
        self.spot().announced.attach(self._flow.onAnnounce)
        return self.spot().startGrabbingAnnounce()

    def _onThreadEnabled(self, thread):
        super()._onThreadEnabled(thread)
        self._ticker.estimate.attach(self._onEstimate, AttachMode.Queued)

    def _onSubscribe(self, flow, chanID):
        if self._receiver is not None:
            self._receiver.hasData.detach(self._onData)
        self._receiver = flow.buildReceiver(chanID)
        self._receiver.clear()
        self._generation += 1
        self._receiver.hasData.attach(self._onData)

    def _onProperties(self, flow, evt, owner, name, resType, resID, ts, payload):
        if (
            evt == FlowEvent.RESOURCE_CHANGED
            and resType == FlowResource.CHANNEL
            and self._receiver is not None
            and resID == self._receiver.chanID()
        ):
            self._generation += 1
            self._receiver.clear()

    def _onDisable(self, flow):
        self._generation += 1
        if self._receiver is not None:
            self._receiver.hasData.detach(self._onData)
            self._receiver.clear()
        self._receiver = None
        self.service().emitEvent(FlowEvent.EXTERNAL, "INPUT_DISCONNECTION", self._source)
        if self.isRunning():
            self.service().publishString(
                self._output,
                json.dumps({"schema": "gaze-direction.v1", "status": "input_disconnected"}),
                time.time_ns() // 1000,
            )

    def _onData(self, receiver):
        latest = receiver.next()
        while not receiver.isEmpty():
            latest = receiver.next()
            self._dropped += 1
        if latest is None:
            return
        now = time.monotonic()
        if (
            not self.isRunning()
            or self._ticker is None
            or self._inFlight
            or now - self._lastDispatch < 1 / 15.0
        ):
            self._dropped += 1
            return
        self._inFlight = True
        self._lastDispatch = now
        self._ticker.loop().invokeSlot(
            self._ticker.infer,
            {
                "data": bytes(latest.data),
                "props": copy.deepcopy(receiver.props()),
                "ts": latest.ts,
                "generation": self._generation,
            },
        )

    def _onEstimate(self, pack):
        self._inFlight = False
        if not self.isRunning() or pack["generation"] != self._generation:
            return
        sample = {**pack["sample"], "source_ts_us": pack["ts"], "dropped_frames": self._dropped}
        self.service().publishString(self._output, json.dumps(sample, allow_nan=False), pack["ts"])

    def _onStop(self):
        self._generation += 1

    def _onQuit(self):
        self._generation += 1
        if self._receiver is not None:
            self._receiver.hasData.detach(self._onData)
        if self._flow is not None:
            self.spot().announced.detach(self._flow.onAnnounce)
            self._flow.close()
        super()._onQuit()


EyeGaze = GazeDirection
