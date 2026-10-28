import json
import math
import time

from N2uCore.dyn.wrk.device import SwDevice
from N2uCore.dyn.wrk.service import tool
from N2uCore.dyn.wrk.input import WorkerFlowInput
from N2uCore.miniflow.channel import FlowChannelType
from N2uCore.inference.screengaze import ScreenCalibration, screenPoint


class ScreenPointOfGaze(SwDevice):
    def __init__(self):
        super().__init__()
        self._cal = ScreenCalibration()
        self._source = self._flow = self._receiver = None
        self._output = None
        self._lastTs = None
        self._lastArrival = self._clockOffset = None
        self._lastSourceAge = 0

    def _onSetup(self):
        self._source = self.service().flowPack().get("GazeDirection")
        if not self._source:
            self.err("ScreenPointOfGaze requires a GazeDirection input")
            return False
        self._flow = WorkerFlowInput(self.service())
        self._flow.setCredentials("guest", "password")
        return True

    def _onInit(self):
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
        if not self._output:
            return False
        self._flow.setTargetByAnnounce(self._source["node"])
        self._flow.setAutoSubscribeChannels([self._source["channel"]])
        self._flow.subscribed.attach(self._onSubscribe)
        self._flow.disabled.attach(self._onDisable)
        self.spot().announced.attach(self._flow.onAnnounce)
        return self.spot().startGrabbingAnnounce()

    def _onSubscribe(self, flow, chanID):
        if self._receiver is not None:
            self._receiver.hasData.detach(self._onData)
        self._lastTs = None
        self._lastArrival = self._clockOffset = None
        self._lastSourceAge = 0
        self._cal.cancel()
        self._receiver = flow.buildReceiver(chanID)
        self._receiver.clear()
        self._receiver.hasData.attach(self._onData)

    def _onDisable(self, flow):
        if self._receiver is not None:
            self._receiver.hasData.detach(self._onData)
        self._receiver = None
        self._lastTs = None
        self._lastArrival = self._clockOffset = None
        self._lastSourceAge = 0
        self._cal.cancel()
        if self.isRunning():
            self._publish({"status": "input_disconnected", "screen_xy_px": None}, time.time_ns() // 1000)

    def _publish(self, sample, ts):
        self.service().publishString(
            self._output,
            json.dumps(
                {
                    "schema": "screen-point-of-gaze.v1",
                    "coordinate_space": "screen_pixels",
                    "source_ts_us": ts,
                    "gaze2d": None,
                    "gaze2d_coordinate_space": "viewport_normalized",
                    "eyeleft": {"pupildiameter": None},
                    "eyeright": {"pupildiameter": None},
                    **sample,
                },
                allow_nan=False,
            ),
            ts,
        )

    def _process(self, sample, ts):
        output = {"status": sample.get("status", "invalid_input"), "screen_xy_px": None, "gaze2d": None}
        if sample.get("schema") != "gaze-direction.v1":
            output["status"] = "invalid_input"
            return output
        now = time.monotonic()
        if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts):
            output["status"] = "invalid_input"
            return output
        if self._lastTs is not None and ts <= self._lastTs:
            output["status"] = "stale_input"
            return output
        sourceAge = sample.get("source_age_secs", 0)
        if (
            isinstance(sourceAge, bool)
            or not isinstance(sourceAge, (int, float))
            or not math.isfinite(sourceAge)
            or sourceAge < 0
        ):
            output["status"] = "invalid_input"
            return output
        if sourceAge > 2.0:
            output["status"] = "stale_input"
            return output
        offset = now - ts / 1e6 - sourceAge
        if self._clockOffset is None:
            self._clockOffset = offset
        self._clockOffset = min(self._clockOffset, offset)
        delay = offset - self._clockOffset
        if sourceAge + delay > 2.0:
            output["status"] = "stale_input"
            return output
        self._lastTs = ts
        self._lastArrival = now
        self._lastSourceAge = sourceAge
        if sample.get("status") != "ok":
            return output
        yaw, pitch = sample.get("yaw_rad"), sample.get("pitch_rad")
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in (yaw, pitch)
        ):
            output["status"] = "invalid_input"
            return output
        self._cal.feed(yaw, pitch, now, ts)
        if self._cal.calibration() is None:
            output["status"] = "uncalibrated"
        else:
            output["screen_xy_px"] = screenPoint(yaw, pitch, self._cal.calibration())
            if output["screen_xy_px"] is None:
                output["status"] = "invalid_input"
                return output
            output["screen_size_px"] = [self._cal.calibration()[k] for k in ("screen_w", "screen_h")]
            output["gaze2d"] = [point / size for point, size in zip(output["screen_xy_px"], output["screen_size_px"])]
        return output

    def _onData(self, receiver):
        pack = receiver.next()
        while not receiver.isEmpty():
            pack = receiver.next()
        if pack is None or not self.isRunning():
            return
        try:
            sample = json.loads(bytes(pack.data))
            if not isinstance(sample, dict):
                self.err("Expected a gaze direction object")
                result = {"status": "invalid_input", "screen_xy_px": None}
            else:
                result = self._process(sample, pack.ts)
        except Exception as exc:
            self.err("Invalid gaze input: {}".format(exc))
            result = {"status": "invalid_input", "screen_xy_px": None, "error": str(exc)}
        self._publish(result, pack.ts)

    def _onStart(self):
        return True

    def _onStop(self):
        self._lastTs = None
        self._lastArrival = self._clockOffset = None
        self._lastSourceAge = 0
        self._cal.cancel()

    def _onQuit(self):
        self._cal.cancel()
        if self._receiver is not None:
            self._receiver.hasData.detach(self._onData)
        if self._flow is not None:
            self.spot().announced.detach(self._flow.onAnnounce)
            self._flow.close()

    def _command(self, conn, fn):
        try:
            value = fn()
            if value is None or value is False:
                self.service().sendResponse(conn, False, "Calibration command failed; check node log")
                return
            self.service().sendResponse(conn, True, value)
        except Exception as exc:
            self.err("Calibration command: {}".format(exc))
            self.service().sendResponse(conn, False, str(exc))

    @tool
    def calibrationStatus(self, conn, args):
        """Return whether this session is calibrated and collecting a target."""
        self.service().sendResponse(conn, True, self._cal.status())

    @tool
    def getCalibration(self, conn, args):
        """Return the current session calibration, or null when uncalibrated."""
        self.service().sendResponse(conn, True, self._cal.calibration())

    @tool
    def beginCalibration(self, conn, args):
        """Begin a session fit for the given browser dimensions in CSS pixels."""
        self._command(conn, lambda: self._cal.begin(args["screen_w"], args["screen_h"]))

    @tool
    def calibrationTarget(self, conn, args):
        """Collect a target at browser coordinates x/y using fresh gaze samples."""
        now = time.monotonic()
        if self._lastTs is None or self._lastArrival is None or now - self._lastArrival > 2.0:
            self.service().sendResponse(conn, False, "Fresh gaze data is required before collecting a target")
            return
        sourceAfterUs = self._lastTs + int((now - self._lastArrival + self._lastSourceAge) * 1e6)
        self._command(conn, lambda: self._cal.target(args["x"], args["y"], now, sourceAfterUs))

    @tool
    def finishTarget(self, conn, args):
        """Finish the active target after settling; reject insufficient fresh samples."""
        self._command(conn, self._cal.finishTarget)

    @tool
    def fitCalibration(self, conn, args):
        """Fit completed targets and replace the session calibration only on success."""
        self._command(conn, self._cal.fit)

    @tool
    def cancelCalibration(self, conn, args):
        """Cancel target collection while preserving the previous session calibration."""
        self._cal.cancel()
        self.service().sendResponse(conn, True, {"cancelled": True})
