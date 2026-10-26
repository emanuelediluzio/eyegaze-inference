import json
import math
import time
from pathlib import Path

from N2uCore.dyn.wrk.device import SwDevice
from N2uCore.dyn.wrk.service import tool
from N2uCore.dyn.wrk.input import WorkerFlowInput
from N2uCore.miniflow.channel import FlowChannelType
from N2uCore.inference.screengaze import ScreenCalibration, screen_point, validate_calibration


class ScreenPointOfGaze(SwDevice):
    def __init__(self):
        super().__init__()
        self._cal = ScreenCalibration()
        self._source = self._flow = self._receiver = None
        self._output = None
        self._last_ts = None

    def _onSetup(self):
        self._source = self.service().flowPack().get("GazeDirection")
        if not self._source:
            self.err("ScreenPointOfGaze requires a GazeDirection input")
            return False
        path = self._props.get("calibrationPath", "")
        try:
            if path:
                self._cal.calibration = validate_calibration(json.loads(Path(path).expanduser().read_text()))
        except Exception as exc:
            self.err("Invalid calibration: {}".format(exc))
            return False
        self._flow = WorkerFlowInput(self.service())
        self._flow.setCredentials("guest", "password")
        # Calibration needs input even when the output card is closed.
        self.disableAutoStart()
        self.initialized.attach(self._startAfterInit)
        return True

    def _startAfterInit(self, *args):
        self.start()

    def _onInit(self):
        self._output = self.service().addChannel(
            "ScreenPoint",
            FlowChannelType.TEXT,
            {
                "format": "json",
                "schema": "screen-point-of-gaze.v1",
                "coordinateSpace": "screen_pixels",
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
        self._last_ts = None
        self._cal.cancel()
        self._receiver = flow.buildReceiver(chanID)
        self._receiver.clear()
        self._receiver.hasData.attach(self._onData)

    def _onDisable(self, flow):
        if self._receiver is not None:
            self._receiver.hasData.detach(self._onData)
        self._receiver = None
        self._last_ts = None
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
                    **sample,
                },
                allow_nan=False,
            ),
            ts,
        )

    def _process(self, sample, ts):
        output = {"status": sample.get("status", "invalid_input"), "screen_xy_px": None}
        if sample.get("schema") != "gaze-direction.v1":
            output["status"] = "invalid_input"
            return output
        age = (time.time_ns() // 1000 - ts) / 1e6
        if (self._last_ts is not None and ts <= self._last_ts) or not -0.5 <= age <= self._props[
            "maxSampleAgeSecs"
        ]:
            output["status"] = "stale_input"
            return output
        self._last_ts = ts
        if sample.get("status") != "ok":
            return output
        yaw, pitch = sample.get("yaw_rad"), sample.get("pitch_rad")
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in (yaw, pitch)
        ):
            output["status"] = "invalid_input"
            return output
        self._cal.feed(yaw, pitch, time.monotonic(), ts)
        if self._cal.calibration is None:
            output["status"] = "uncalibrated"
        else:
            output["screen_xy_px"] = screen_point(yaw, pitch, self._cal.calibration)
            output["screen_size_px"] = [self._cal.calibration[k] for k in ("screen_w", "screen_h")]
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
                raise ValueError("Expected a gaze direction object")
            result = self._process(sample, pack.ts)
        except Exception as exc:
            result = {"status": "invalid_input", "screen_xy_px": None, "error": str(exc)}
        self._publish(result, pack.ts)

    def _onStart(self):
        return True

    def _onStop(self):
        self._last_ts = None
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
            self.service().sendResponse(conn, True, value)
        except Exception as exc:
            self.service().sendResponse(conn, False, str(exc))

    @tool
    def calibrationStatus(self, conn, args):
        self.service().sendResponse(
            conn,
            True,
            {
                "calibrated": self._cal.calibration is not None,
                "collecting": self._cal.size is not None,
                "targets": len(self._cal.targets),
                "active_samples": len(self._cal.active["samples"]) if self._cal.active else 0,
            },
        )

    @tool
    def setCalibration(self, conn, args):
        def apply():
            value = validate_calibration(args["calibration"])
            self._cal.cancel()
            self._cal.calibration = value
            return value

        self._command(conn, apply)

    @tool
    def getCalibration(self, conn, args):
        self.service().sendResponse(conn, True, self._cal.calibration)

    @tool
    def clearCalibration(self, conn, args):
        self._cal.cancel()
        self._cal.calibration = None
        self.service().sendResponse(conn, True, {"calibrated": False})

    @tool
    def beginCalibration(self, conn, args):
        self._command(conn, lambda: self._cal.begin(args["screen_w"], args["screen_h"]))

    @tool
    def calibrationTarget(self, conn, args):
        self._command(
            conn, lambda: self._cal.target(args["x"], args["y"], time.monotonic(), time.time_ns() // 1000)
        )

    @tool
    def finishTarget(self, conn, args):
        self._command(conn, self._cal.finish_target)

    @tool
    def fitCalibration(self, conn, args):
        self._command(conn, self._cal.fit)

    @tool
    def cancelCalibration(self, conn, args):
        self._cal.cancel()
        self.service().sendResponse(conn, True, {"cancelled": True})
