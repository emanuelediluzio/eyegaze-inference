"""Tests for the ScreenPointOfGaze node (N2uCore/dyn/screenpointofgaze.py).

  a. ``_onSubscribe``: the second subscription emits INPUT_RECONNECTION;
  b. ``_onDisable``: INPUT_DISCONNECTION, waiting for the input again;
  c. ``_onInputChanged``: the output gets the input rate, the image size is kept;
  d. ``_onDataCome``: the output sample carries the input ts and packID, and
     only the per-sample fields (schema, coordinate spaces, pupils are channel
     properties);
  e. ``_process``: upstream status kept.

Built without an event loop (``__new__`` + attributes, MagicMock where
needed), imitating tests/dyn/test_roier_input_properties.py.

Run from project root:
    python3 -m unittest tests.dyn.test_screenpointofgaze -v
"""

import json
import time
import unittest
from unittest.mock import MagicMock

from N2uCore.dyn.screenpointofgaze import ScreenPointOfGaze
from N2uCore.inference.screengaze import ScreenCalibration
from N2uCore.miniflow.proto import FlowEvent


def _sample(**fields):
    """Build an "ok" gaze-direction sample, with ``fields`` overridden."""
    sample = {"status": "ok", "yaw_rad": 0.1, "pitch_rad": -0.1, "quality": None,
              "eye_center_px": [320.0, 200.0], "eye_span_px": 110.0, "iris": [0.0, 0.05, 0.3],
              "bbox_xyxy_px": [220, 100, 420, 340]}
    sample.update(fields)
    return sample


def _nowUs():
    """A timestamp in microseconds on the clock of the frames (time.time)."""
    return time.time_ns() // 1000


def _makeNode():
    """Build a bare ScreenPointOfGaze with only the attributes the tested methods read.

    Returns:
        ScreenPointOfGaze: the bare node under test (no __init__ run).
    """
    node = ScreenPointOfGaze.__new__(ScreenPointOfGaze)
    node._cal = ScreenCalibration()
    node._receiver = None
    node._source = {"node": "gaze", "channel": "GazeDirection"}
    node._output = 21
    node._imageSize = {"w": 640, "h": 480}
    node.err = MagicMock()
    node.isRunning = MagicMock(return_value=True)

    service = MagicMock()
    service.channelByID.return_value.isEmpty.return_value = False   # someone reads ScreenPoint
    node.service = MagicMock(return_value=service)
    spot = MagicMock()
    node.spot = MagicMock(return_value=spot)
    node.initSpotAnnouncing = MagicMock()
    node.quitSpotAnnouncing = MagicMock()
    return node


class _FakePack:
    """A fake FlowChannelData: data, ts and packID."""

    def __init__(self, data, ts, packID):
        self.data = data
        self.ts = ts
        self.packID = packID


class _FakeReceiver:
    """A fake FlowChannelReceiver holding one pack."""

    def __init__(self, pack):
        self._packs = [pack]

    def next(self):
        """Return the next pack, or None."""
        return self._packs.pop(0) if self._packs else None

    def isEmpty(self):
        """Return True when no pack is left."""
        return not self._packs


class TestSubscribe(unittest.TestCase):
    """(a) subscription of the GazeDirection input."""

    def test_second_subscription_is_a_reconnection(self):
        """(a) first subscription: no event; second one: INPUT_RECONNECTION."""
        node = _makeNode()
        flow = MagicMock()
        flow.buildReceiver.return_value.props.return_value = {"rate": 15, "imageSize": {"w": 640, "h": 480}}

        node._onSubscribe(flow, 5)
        node.service().emitEvent.assert_not_called()

        node._onSubscribe(flow, 5)
        node.service().emitEvent.assert_called_once_with(FlowEvent.EXTERNAL, "INPUT_RECONNECTION", node._source)


class TestDisable(unittest.TestCase):
    """(b) input lost."""

    def test_disconnection_event(self):
        """(b) INPUT_DISCONNECTION emitted, announcements grabbed again (as ImageGeometry)."""
        node = _makeNode()

        node._onDisable(MagicMock())

        node.service().emitEvent.assert_called_once_with(FlowEvent.EXTERNAL, "INPUT_DISCONNECTION", node._source)
        node.spot().startGrabbingAnnounce.assert_called_once()


class TestRate(unittest.TestCase):
    """(c) rate of the output."""

    def test_rate_follows_input(self):
        """(c) input at 15 Hz -> output at 15 Hz; the camera image size of the input is kept."""
        node = _makeNode()
        node._onInputChanged({"rate": 15, "imageSize": {"w": 1280, "h": 720}})
        node.service().channelByID.assert_called_with(21)
        node.service().channelByID().setProperties.assert_called_once_with({"rate": 15})
        self.assertEqual(node._imageSize, {"w": 1280, "h": 720})


class TestData(unittest.TestCase):
    """(d) one input sample, one output sample."""

    def test_output_carries_pack_id(self):
        """(d) the ScreenPoint sample carries the input ts and packID."""
        node = _makeNode()
        ts = _nowUs()
        node._onDataCome(_FakeReceiver(_FakePack(json.dumps(_sample()).encode(), ts, 7)))

        args = node.service().publishString.call_args[0]
        self.assertEqual(args[0], 21)
        self.assertEqual(args[2:], (ts, 7))
        published = json.loads(args[1])
        self.assertEqual(published["status"], "uncalibrated")
        for key in ("schema", "source_ts_us", "eyeleft", "eyeright", "coordinate_space"):
            self.assertNotIn(key, published)


    def test_nobody_reading_publishes_nothing(self):
        """(d) no subscriber on ScreenPoint -> nothing processed or published (as ImageGeometry)."""
        node = _makeNode()
        node.service().channelByID.return_value.isEmpty.return_value = True
        node._onDataCome(_FakeReceiver(_FakePack(json.dumps(_sample()).encode(), _nowUs(), 7)))
        node.service().publishString.assert_not_called()


class TestProcess(unittest.TestCase):
    """(e) one input sample."""

    def test_upstream_status_is_kept(self):
        """(e) upstream "no_face" -> "no_face", no screen point."""
        node = _makeNode()
        output = node._process(_sample(status="no_face", yaw_rad=None, pitch_rad=None), _nowUs())
        self.assertEqual(output["status"], "no_face")
        self.assertNotIn("screen_xy_px", output)   # published as null by _publish


if __name__ == "__main__":
    unittest.main()
