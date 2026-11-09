from __future__ import annotations

import json
import math
import time

import numpy as np

###############################################################################

from N2uCore.dyn.wrk.mldevice           import MlDevice, MlTicker
from N2uCore.dyn.wrk.memprovider        import resolveComputeResources
from N2uCore.dyn.wrk.input              import WorkerFlowInput
from N2uCore.miniflow.channel           import FlowChannelType
from N2uCore.miniflow.proto             import FlowEvent, FlowResource
from N2uCore.signal                     import Signal, AttachMode
from N2uCore.resman                     import ResourceManager
from N2uCore.multimedia.imgdec          import ImageDecoder
from N2uCore.inference.gazedinoengine   import GazeDinoEngine

###############################################################################
###############################################################################

class GazeDirectionTicker(MlTicker):

    MODEL_PATH = "models/gazedino/model.joblib"
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
        self._modelLabel = "GazeDINO ViT-B/14"
        return self._engine.load()

    def _unloadModel(self) -> None:
        self._engine.close()
        self._engine = None

    def infer(self, image, ts, packID=-1) -> None:
        # one estimate per call: the node sends the next frame only after it
        if time.time() < self._retryAt:
            sample = {"status": "model_unavailable"}
        elif not self._ensureLoaded():
            self._retryAt = time.time() + self.RETRY_SECS
            self.err("Model not loaded: retrying in {} s".format(self.RETRY_SECS))
            sample = {"status": "model_unavailable"}
        else:
            sample = self._engine.analyze(image, ts) or {"status": "error"}

        self.estimate.trigger([{"sample": sample, "ts": ts, "packID": packID}])

###############################################################################
###############################################################################

class GazeDirection(MlDevice):

    GAZE_LENGTH_PX = 200      # eye dot -> direction dot, in image pixels

    def __init__(self):
        super().__init__()

        self._flow = None
        self._receiver = None
        self._decoder = None
        self._source = None

        self._output = None           # GazeDirection (JSON), read by ScreenPointOfGaze
        self._points = None           # GazePoints (POINTS_2D), drawn on the card

        self._memProvider = None
        self._inFlight = False

    def _onSetup(self) -> bool:
        if not super()._onSetup():
            return False

        self.disableAutoAnnounce()

        resources = resolveComputeResources(
            deviceName="cpu" if self._props["device"] == "CPU" else "auto",
            supportedDevices=GazeDinoEngine.SUPPORTED_DEVICES
        )
        if resources is None:
            return False
        self._device, self._memProvider = resources

        self._source = self.service().flowPack()["VideoStream"]

        self._flow = WorkerFlowInput(self.service())
        self._flow.setCredentials("guest", "password")

        self._decoder = ImageDecoder()
        return True

    def _tickerClass(self) -> type:
        return GazeDirectionTicker

    def _tickerArgs(self) -> dict:
        return {
            "memProvider": self._memProvider,
            "modelPath": ResourceManager().absolutePath(GazeDirectionTicker.MODEL_PATH),
            "deviceName": self._device,
            "idleTimeoutSecs": 600,
        }

    def _onInit(self) -> bool:
        source = "{}.{}".format(self._source["node"], self._source["channel"])

        self._output = self.service().addChannel(
            "GazeDirection",
            FlowChannelType.GAZE,
            {
                "format": "json",
                "schema": "gaze-direction.v1",
                "imageSize": None,              # set on subscription, like the rate
                "angleUnit": "radians",
                "dependsOn": [source],
                "coordinateSpace": "camera",
                "vectorConvention": "[sin(yaw)*cos(pitch), -sin(pitch), -cos(yaw)*cos(pitch)]",
                "pupilDiameterAvailable": False,
            },
        )

        # N x 3 float32 [x_px, y_px, confidence]: row 0 = eye, row 1 = eye + direction
        self._points = self.service().addChannel(
            "GazePoints",
            FlowChannelType.POINTS_2D,
            {"dependsOn": [source]},
        )

        if not super()._onInit():
            return False

        self._flow.setTargetByAnnounce(self._source["node"])
        self._flow.setAutoSubscribeChannels([self._source["channel"]])
        self._flow.subscribed.attach(self._onSubscribe)
        self._flow.disabled.attach(self._onDisable)
        self._flow.eventGrabbed.attach(self._onInputProperties)

        self.spot().announced.attach(self._flow.onAnnounce)
        return self.spot().startGrabbingAnnounce()

    def _onThreadEnabled(self, thread) -> None:
        super()._onThreadEnabled(thread)
        self._ticker.estimate.attach(self._onEstimate, AttachMode.Queued)

###############################################################################

    def _onDisable(self, flow) -> None:
        self.service().emitEvent(FlowEvent.EXTERNAL, "INPUT_DISCONNECTION", self._source)

        self.spot().announced.attach(flow.onAnnounce)
        self.spot().startGrabbingAnnounce()
        self.quitSpotAnnouncing()

    def _onSubscribe(self, flow, chanID) -> None:
        self.spot().stopGrabbingAnnounce()
        self.spot().announced.detach(flow.onAnnounce)

        if self._receiver is not None:
            self.service().emitEvent(FlowEvent.EXTERNAL, "INPUT_RECONNECTION", self._source)

        self._receiver = flow.buildReceiver(chanID)
        props = self._receiver.props()

        if props["type"] != "VIDEO_VISIBLE":
            self.err("Invalid device type: {}".format(props["type"]))
            return

        self._setupDecoder(props)
        self._updateOutputs(props)

        self._receiver.clear()
        self._receiver.hasData.attach(self._onDataCome)
        self.initSpotAnnouncing()

    def _setupDecoder(self, props) -> bool:
        return self._decoder.setup(props["size"], props["depth"], props["pxFmt"], props["encoding"]["codec"])

    def _updateOutputs(self, inputProps) -> None:
        self.service().channelByID(self._output).setProperties({"rate": inputProps["rate"], "imageSize": inputProps["size"]})
        self.service().channelByID(self._points).setProperties({"rate": inputProps["rate"]})

    def _onDataCome(self, receiver) -> None:
        pack = receiver.next()

        if self.service().channelByID(self._output).isEmpty() and self.service().channelByID(self._points).isEmpty():
            return

        # one frame at a time, the model sets the pace: the frames that come meanwhile are dropped
        if self._inFlight:
            return

        image = self._decoder.decode(bytes(pack.data))
        self._inFlight = True
        self._ticker.loop().invokeSlot(self._ticker.infer, {"image": image, "ts": pack.ts, "packID": pack.packID})

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
        if self._receiver is None:
            return
        if resID != self._receiver.chanID():
            return

        props = flow.channelProperties(resID)
        if props is None:
            return

        self._setupDecoder(props)
        self._updateOutputs(props)

###############################################################################
###############################################################################
