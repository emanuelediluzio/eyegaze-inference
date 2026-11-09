from __future__ import annotations

import json
import time

###############################################################################

from N2uCore.dyn.wrk.device             import SwDevice
from N2uCore.dyn.wrk.service            import tool
from N2uCore.dyn.wrk.input              import WorkerFlowInput
from N2uCore.miniflow.channel           import FlowChannelType
from N2uCore.miniflow.proto             import FlowEvent, FlowResource
from N2uCore.inference.screengaze       import ScreenCalibration, headPose, screenPoint

###############################################################################
###############################################################################

class ScreenPointOfGaze(SwDevice):

    def __init__(self):
        super().__init__()

        # session only: nothing is saved, the browser drives it with the commands below
        self._cal = ScreenCalibration()

        self._flow = None
        self._receiver = None
        self._source = None

        self._output = None
        self._imageSize = None        # camera frame size of the GazeDirection input

    def _onSetup(self) -> bool:
        self.disableAutoAnnounce()

        self._source = self.service().flowPack()["GazeDirection"]

        self._flow = WorkerFlowInput(self.service())
        self._flow.setCredentials("guest", "password")
        return True

    def _onInit(self) -> bool:
        self._output = self.service().addChannel(
            "ScreenPoint",
            FlowChannelType.GAZE,
            {
                "format": "json",
                "schema": "screen-point-of-gaze.v1",
                "coordinateSpace": "screen_pixels",
                "gaze2dCoordinateSpace": "viewport_normalized",
                "pupilDiameterAvailable": False,
                "dependsOn": ["{}.{}".format(self._source["node"], self._source["channel"])],
            },
        )

        self._flow.setTargetByAnnounce(self._source["node"])
        self._flow.setAutoSubscribeChannels([self._source["channel"]])
        self._flow.subscribed.attach(self._onSubscribe)
        self._flow.disabled.attach(self._onDisable)
        self._flow.eventGrabbed.attach(self._onInputProperties)

        self.spot().announced.attach(self._flow.onAnnounce)
        return self.spot().startGrabbingAnnounce()

    def _onStart(self) -> bool:
        return True

    def _onStop(self) -> None:
        self._cal.cancel()

    def _onQuit(self) -> None:
        pass

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

        self._onInputChanged(self._receiver.props())

        self._receiver.clear()
        self._receiver.hasData.attach(self._onDataCome)
        self.initSpotAnnouncing()

    def _onInputChanged(self, inputProps) -> None:
        self._imageSize = inputProps["imageSize"]
        self.service().channelByID(self._output).setProperties({"rate": inputProps["rate"]})

    def _onDataCome(self, receiver) -> None:
        pack = receiver.next()

        if self.service().channelByID(self._output).isEmpty():
            return

        self._publish(self._process(json.loads(bytes(pack.data)), pack.ts), pack.ts, pack.packID)

    def _process(self, sample, ts) -> dict:
        output = {"status": sample["status"]}
        if sample["status"] != "ok":
            return output

        output["quality"] = sample["quality"]

        gaze = [sample["yaw_rad"], sample["pitch_rad"], *sample["iris"]]
        head = headPose(sample, self._imageSize)
        self._cal.feed(gaze, ts / 1e6, head)

        calibration = self._cal.calibration()
        if calibration is None:
            output["status"] = "uncalibrated"
            return output

        output["screen_xy_px"] = screenPoint(gaze, calibration)
        output["screen_size_px"] = [calibration["screen_w"], calibration["screen_h"]]
        output["gaze2d"] = [point / size for point, size in zip(output["screen_xy_px"], output["screen_size_px"])]
        # the mapping holds only near the head position of the calibration
        output["calibration_drift"] = self._cal.drift(head)
        output["frame_aspect"] = self._imageSize["h"] / self._imageSize["w"]
        return output

    def _publish(self, sample, ts, packID=-1) -> None:
        self.service().publishString(
            self._output,
            json.dumps({"screen_xy_px": None, "gaze2d": None, **sample}, allow_nan=False),
            ts,
            packID,
        )

###############################################################################

    def _onInputProperties(self, flow, evt, owner, name, resType, resID, ts, payload) -> None:
        """Slot of ``eventGrabbed``: follows image size and rate when the GazeDirection channel changes."""
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

        self._onInputChanged(props)

###############################################################################

    def _command(self, conn, fn) -> None:
        value = fn()
        if not value:
            self.service().sendResponse(conn, False, "Calibration command failed; check node log")
            return
        self.service().sendResponse(conn, True, value)

    @tool
    def calibrationStatus(self, conn, args):
        self.service().sendResponse(conn, True, self._cal.status())

    @tool
    def getCalibration(self, conn, args):
        self.service().sendResponse(conn, True, self._cal.calibration())

    @tool
    def beginCalibration(self, conn, args):
        self._command(conn, lambda: self._cal.begin(args["screen_w"], args["screen_h"]))

    @tool
    def calibrationTarget(self, conn, args):
        self._command(conn, lambda: self._cal.target(args["x"], args["y"], time.time()))

    @tool
    def finishTarget(self, conn, args):
        self._command(conn, self._cal.finishTarget)

    @tool
    def fitCalibration(self, conn, args):
        self._command(conn, self._cal.fit)

    @tool
    def cancelCalibration(self, conn, args):
        self._cal.cancel()
        self.service().sendResponse(conn, True, {"cancelled": True})

###############################################################################
###############################################################################
