from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

###############################################################################

from N2uCore.dyn.wrk.mldevice           import MlDevice, MlTicker
from N2uCore.dyn.wrk.memprovider        import resolveComputeResources
from N2uCore.dyn.wrk.input              import WorkerFlowInput
from N2uCore.miniflow.channel           import FlowChannelType
from N2uCore.miniflow.proto             import FlowEvent, FlowResource
from N2uCore.miniflow.flowasync         import FlowChannelDataAggregator
from N2uCore.signal                     import Signal, AttachMode
from N2uCore.resman                     import ResourceManager
from N2uCore.multimedia.imgdec          import ImageDecoder
from N2uCore.inference.gazedinoengine   import GazeDinoEngine
from N2uCore.loop                       import LoopTimerMode

###############################################################################
###############################################################################

class GazeDirectionTicker(MlTicker):

    MODEL_PATH = "models/gazedino/model.joblib"
    COREML_PATH = "models/gazedino/model.mlpackage"   # macOS: the int8 Core ML export, used when present
    # device CPU: the ViT-S student distilled from the model (eyegaze-inference experiments/distill-v3), 4x smaller
    # and ~2.5x faster, same accuracy on MPIIFaceGaze (6.25 vs 6.31 deg) and on the recorded sessions
    STUDENT_PATH = "models/gazedino/student.joblib"
    STUDENT_COREML_PATH = "models/gazedino/student.mlpackage"
    FOOTPRINT = 700 * 1024**2     # ViT-B/14 + head + face mesh

    RETRY_SECS = 30

    def __init__(self, memProvider, modelPath, deviceName, idleTimeoutSecs):
        super().__init__(memProvider, idleTimeoutSecs=idleTimeoutSecs)

        self._modelPath = modelPath
        self._deviceName = deviceName
        self._engine = None
        self._retryAt = 0

        # pack: {"sample": dict, "ts": int, "packID": int}
        self.estimate = Signal(self, "estimate")

    def _estimateFootprint(self) -> int:
        return GazeDirectionTicker.FOOTPRINT

    def _loadModel(self) -> bool:
        self._engine = GazeDinoEngine(self._modelPath, deviceName=self._deviceName)
        student = Path(self._modelPath).stem == "student"
        self._modelLabel = "GazeDINO " + ("ViT-S/14 student" if student else "ViT-B/14") + \
            (" Core ML" if self._modelPath.endswith(".mlpackage") else "")
        return self._engine.load()

    def _unloadModel(self) -> None:
        self._engine.close()
        self._engine = None

    def infer(self, image, mesh, ts, packID=-1) -> None:
        # one estimate per call: the node sends the next frame only after it
        if time.time() < self._retryAt:
            sample = {"status": "model_unavailable"}
        elif not self._ensureLoaded():
            self._retryAt = time.time() + self.RETRY_SECS
            self.err("Model not loaded: retrying in {} s".format(self.RETRY_SECS))
            sample = {"status": "model_unavailable"}
        else:
            sample = self._engine.analyze(image, mesh, ts) or {"status": "error"}

        self.estimate.trigger([{"sample": sample, "ts": ts, "packID": packID}])

###############################################################################
###############################################################################

class GazeDirection(MlDevice):

    GAZE_LENGTH_PX = 200      # eye dot -> direction dot, in image pixels

    def __init__(self):
        super().__init__()

        self._imgFlow = None
        self._ptsFlow = None
        self._imgReceiver = None
        self._ptsReceiver = None
        self._imgChanID = None
        self._ptsChanID = None
        self._imgPackID = None
        self._ptsPackID = None
        self._aggregator = None
        self._decoder = None
        self._imgSource = None
        self._meshSource = None

        self._output = None           # GazeDirection (JSON), read by ScreenPointOfGaze
        self._points = None           # GazePoints (POINTS_2D), drawn on the card

        self._device = None
        self._memProvider = None
        self._inFlight = False

    def _onSetup(self) -> bool:
        if not super()._onSetup():
            return False

        self.setRunningTimerMode(LoopTimerMode.TIMEDLOOP_RT)
        self.disableAutoAnnounce()

        resources = resolveComputeResources(
            deviceName="cpu" if self._props["device"] == "CPU" else "auto",
            supportedDevices=GazeDinoEngine.SUPPORTED_DEVICES
        )
        if resources is None:
            return False
        self._device, self._memProvider = resources

        self._imgSource = self.service().flowPack()["VideoStream"]
        self._meshSource = self.service().flowPack()["Landmarks"]

        self._imgFlow = WorkerFlowInput(self.service())
        self._imgFlow.setCredentials("guest", "password")
        self._ptsFlow = WorkerFlowInput(self.service())
        self._ptsFlow.setCredentials("guest", "password")

        self._decoder = ImageDecoder()
        return True

    def _tickerClass(self) -> type:
        return GazeDirectionTicker

    def _tickerArgs(self) -> dict:
        return {
            "memProvider": self._memProvider,
            "modelPath": self._modelFile(self._props["device"] == "CPU"),
            "deviceName": self._device,
            "idleTimeoutSecs": 600,
        }

    @staticmethod
    def _modelFile(student: bool) -> str:
        # device CPU: the distilled student; GPU: the model. On macOS the Core ML export of either (same angles,
        # ~10x faster model) when it and coremltools are there
        weights, coreml = (GazeDirectionTicker.STUDENT_PATH, GazeDirectionTicker.STUDENT_COREML_PATH) if student \
            else (GazeDirectionTicker.MODEL_PATH, GazeDirectionTicker.COREML_PATH)
        if sys.platform == "darwin" and importlib.util.find_spec("coremltools") is not None:
            path = ResourceManager().absolutePath(coreml)
            if path and Path(path).exists():
                return path
        return ResourceManager().absolutePath(weights)

    def _onInit(self) -> bool:
        source = "{}.{}".format(self._imgSource["node"], self._imgSource["channel"])
        meshSource = "{}.{}".format(self._meshSource["node"], self._meshSource["channel"])

        self._output = self.service().addChannel(
            "GazeDirection",
            FlowChannelType.GAZE,
            {
                "format": "json",
                "schema": "gaze-direction.v1",
                "imageSize": None,              # set on subscription, like the rate
                "angleUnit": "radians",
                "dependsOn": [source, meshSource],
                "coordinateSpace": "camera",
                "vectorConvention": "[sin(yaw)*cos(pitch), -sin(pitch), -cos(yaw)*cos(pitch)]",
                "pupilDiameterAvailable": False,
            },
        )

        # N x 3 float32 [x_px, y_px, confidence]: row 0 = eye, row 1 = eye + direction
        self._points = self.service().addChannel(
            "GazePoints",
            FlowChannelType.POINTS_2D,
            {"dependsOn": [source, meshSource]},
        )

        if not super()._onInit():
            return False

        self._imgFlow.setTargetByAnnounce(self._imgSource["node"])
        self._imgFlow.setAutoSubscribeChannels([self._imgSource["channel"]])
        self._imgFlow.subscribed.attach(self._onSubscribe)
        self._imgFlow.disabled.attach(self._onDisable)
        self._imgFlow.eventGrabbed.attach(self._onInputProperties)

        self._ptsFlow.setTargetByAnnounce(self._meshSource["node"])
        self._ptsFlow.setAutoSubscribeChannels([self._meshSource["channel"]])
        self._ptsFlow.subscribed.attach(self._onSubscribe)
        self._ptsFlow.disabled.attach(self._onDisable)

        self.spot().announced.attach(self._imgFlow.onAnnounce)
        self.spot().announced.attach(self._ptsFlow.onAnnounce)
        return self.spot().startGrabbingAnnounce()

    def _onThreadEnabled(self, thread) -> None:
        super()._onThreadEnabled(thread)
        self._ticker.estimate.attach(self._onEstimate, AttachMode.Queued)

###############################################################################

    def _onDisable(self, flow) -> None:
        if flow is self._imgFlow:
            self._imgReceiver = None
            lostSrc = self._imgSource
        else:
            self._ptsReceiver = None
            lostSrc = self._meshSource

        self.service().emitEvent(FlowEvent.EXTERNAL, "INPUT_DISCONNECTION", lostSrc)

        if self.spot().isSendingEnabled():
            self.quitSpotAnnouncing()
        self.spot().announced.attach(flow.onAnnounce)
        if not self.spot().isReceivingEnabled():
            self.spot().startGrabbingAnnounce()

    def _onSubscribe(self, flow, chanID) -> None:
        receiver = flow.buildReceiver(chanID)
        if flow is self._imgFlow and receiver.props()["type"] != "VIDEO_VISIBLE":
            # refused before it is kept: no frame reaches a decoder never set up
            self.err("Invalid device type: {}".format(receiver.props()["type"]))
            return
        receiver.hasData.attach(self._onDataCome)
        self.spot().announced.detach(flow.onAnnounce)

        if flow is self._imgFlow:
            if self._imgReceiver is not None:
                self.service().emitEvent(FlowEvent.EXTERNAL, "INPUT_RECONNECTION", self._imgSource)
            self._imgReceiver = receiver
            self._imgChanID = chanID
            props = receiver.props()
            self._setupDecoder(props)
            self._updateOutputs(props)
        else:
            if self._ptsReceiver is not None:
                self.service().emitEvent(FlowEvent.EXTERNAL, "INPUT_RECONNECTION", self._meshSource)
            self._ptsReceiver = receiver
            self._ptsChanID = chanID

        if self._imgReceiver is not None and self._ptsReceiver is not None:
            self.spot().stopGrabbingAnnounce()
            if self._aggregator is None:
                self._aggregator = FlowChannelDataAggregator()
                self._aggregator.addChannel(self._imgReceiver.props())
                self._aggregator.addChannel(self._ptsReceiver.props())
                self._imgPackID = "{}.{}".format(self._imgSource["node"], self._imgChanID)
                self._ptsPackID = "{}.{}".format(self._meshSource["node"], self._ptsChanID)
            else:
                self._aggregator.reset()
            self._imgReceiver.clear()
            self._ptsReceiver.clear()
            self.initSpotAnnouncing()

    def _setupDecoder(self, props) -> bool:
        return self._decoder.setup(props["size"], props["depth"], props["pxFmt"], props["encoding"]["codec"])

    def _updateOutputs(self, inputProps) -> None:
        self.service().channelByID(self._output).setProperties({"rate": inputProps["rate"], "imageSize": inputProps["size"]})
        self.service().channelByID(self._points).setProperties({"rate": inputProps["rate"]})

    def _onDataCome(self, receiver) -> None:
        if self._imgReceiver is None or self._ptsReceiver is None:
            receiver.clear()
            return

        pack = receiver.next()
        packs = self._aggregator.check(pack)
        if packs is None:
            return
        if len(packs) != 2:
            self.err("Aggregated pack list is NOT valid")
            return

        if self.service().channelByID(self._output).isEmpty() and self.service().channelByID(self._points).isEmpty():
            return

        # one frame at a time, the model sets the pace: the frames that come meanwhile are dropped
        if self._inFlight:
            return

        imgPack = packs[self._imgPackID]
        image = self._decoder.decode(imgPack.data)
        ptsPack = packs[self._ptsPackID]
        mesh = np.frombuffer(ptsPack.data, np.float32).reshape(-1, 3)

        self._inFlight = True
        self._ticker.loop().invokeSlot(self._ticker.infer, {"image": image, "mesh": mesh, "ts": imgPack.ts, "packID": imgPack.packID})

    def _onEstimate(self, pack) -> None:
        self._inFlight = False

        sample = pack["sample"]
        self.service().publishString(self._output, json.dumps(sample, allow_nan=False), pack["ts"], pack["packID"])

        if sample["status"] == "ok":
            self.service().publish(self._points, self._pointsFrom(sample).tobytes(), pack["ts"], pack["packID"])

    def _pointsFrom(self, sample):
        # eye midpoint and the same point moved along the direction projected on the image
        # (vectorConvention); confidence 1, the card draws only points above 0.7
        yaw, pitch = sample["yaw_rad"], sample["pitch_rad"]
        (lx, ly), (rx, ry) = sample["eyes_px"]
        ex, ey = (lx + rx) / 2, (ly + ry) / 2
        dx = math.sin(yaw) * math.cos(pitch)
        dy = -math.sin(pitch)
        return np.array([[ex, ey, 1.0],
                         [ex + dx * self.GAZE_LENGTH_PX, ey + dy * self.GAZE_LENGTH_PX, 1.0]],
                        dtype=np.float32)

###############################################################################

    def _onInputProperties(self, flow, evt, owner, name, resType, resID, ts, payload) -> None:
        """Slot of ``eventGrabbed``: applies the new input channel properties to the decoder and the outputs."""
        if evt != FlowEvent.RESOURCE_CHANGED:
            return
        if resType != FlowResource.CHANNEL:
            return
        if flow is not self._imgFlow:
            return
        if self._imgReceiver is None:
            return
        if resID != self._imgChanID:
            return

        props = flow.channelProperties(resID)
        if props is None:
            return

        self._setupDecoder(props)
        self._updateOutputs(props)

###############################################################################
###############################################################################
