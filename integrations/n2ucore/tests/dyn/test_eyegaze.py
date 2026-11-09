"""Tests for the GazeDirection node (N2uCore/dyn/eyegaze.py).

  a. ``_pointsFrom``: eye dot at the midpoint of ``eyes_px``;
  b. ``_onSubscribe``: the second subscription emits INPUT_RECONNECTION, the
     decoder is set up for the input;
  c. ``_updateOutputs``: both outputs get the input rate, GazeDirection also
     the input image size;
  d. ``_onEstimate``: the sample is published as the engine gave it, with the
     frame ts and packID; dots only for an "ok" sample;
  e. ``_onInputProperties``: RESOURCE_CHANGED of the input channel sets the
     decoder up again (as ImageGeometry), other events are ignored;
  f. ``_onDataCome``: the frame is decoded in the node and sent to the
     ticker, one at a time (the model sets the pace).

Built without an event loop (``__new__`` + attributes, MagicMock where
needed), imitating tests/dyn/test_roier_input_properties.py.

Run from project root:
    python3 -m unittest tests.dyn.test_eyegaze -v
"""

import json
import unittest
from unittest.mock import MagicMock

from N2uCore.dyn.eyegaze import GazeDirection
from N2uCore.miniflow.proto import FlowEvent, FlowResource

PROPS = {"type": "VIDEO_VISIBLE", "size": {"w": 640, "h": 480}, "depth": 3, "pxFmt": "UINT8_T", "encoding": {"codec": "JPG"}, "rate": 30}


def _sample(**fields):
    """Build an "ok" gaze-direction sample looking straight ahead, with ``fields`` overridden."""
    sample = {"status": "ok", "yaw_rad": 0.0, "pitch_rad": 0.0, "eyes_px": [[100, 200], [140, 210]]}
    sample.update(fields)
    return sample


def _makeNode():
    """Build a bare GazeDirection with only the attributes the tested methods read.

    Returns:
        GazeDirection: the bare node under test (no __init__ run).
    """
    node = GazeDirection.__new__(GazeDirection)
    node._receiver = None
    node._decoder = MagicMock()
    node._ticker = MagicMock()
    node._source = {"node": "cam", "channel": "VideoStream"}
    node._output = 11
    node._points = 12
    node._inFlight = True
    node.err = MagicMock()
    node.isRunning = MagicMock(return_value=True)

    service = MagicMock()
    service.channelByID.return_value.isEmpty.return_value = False   # someone reads the outputs
    node.service = MagicMock(return_value=service)
    spot = MagicMock()
    node.spot = MagicMock(return_value=spot)
    node.initSpotAnnouncing = MagicMock()
    return node


def _pack(ts=1000, packID=7):
    """An input frame pack."""
    pack = MagicMock()
    pack.data, pack.ts, pack.packID = b"jpg", ts, packID
    return pack


class TestPointsFrom(unittest.TestCase):
    """(a) the dots drawn on the card."""

    def test_eye_dot_is_midpoint_of_eyes_px(self):
        """(a) eyes at (100, 200) and (140, 210) -> eye dot at (120, 205)."""
        points = _makeNode()._pointsFrom(_sample())
        self.assertEqual(points.shape, (2, 3))
        self.assertAlmostEqual(float(points[0][0]), 120.0)
        self.assertAlmostEqual(float(points[0][1]), 205.0)
        # straight ahead: the target dot is on the eye dot
        self.assertAlmostEqual(float(points[1][0]), 120.0)
        self.assertAlmostEqual(float(points[1][1]), 205.0)


class TestSubscribe(unittest.TestCase):
    """(b) subscription of the VideoStream input."""

    def test_second_subscription_is_a_reconnection(self):
        """(b) first subscription: no event; second one: INPUT_RECONNECTION."""
        node = _makeNode()
        flow = MagicMock()
        flow.buildReceiver.return_value.props.return_value = PROPS

        node._onSubscribe(flow, 5)
        node.service().emitEvent.assert_not_called()

        node._onSubscribe(flow, 5)
        node.service().emitEvent.assert_called_once_with(FlowEvent.EXTERNAL, "INPUT_RECONNECTION", node._source)

    def test_thermal_input_is_refused(self):
        """(b) an input that is not VIDEO_VISIBLE -> err, no decoder, no announce (as ImageGeometry)."""
        node = _makeNode()
        flow = MagicMock()
        flow.buildReceiver.return_value.props.return_value = {**PROPS, "type": "VIDEO_THERMAL"}
        node._onSubscribe(flow, 5)
        node.err.assert_called_once()
        node._decoder.setup.assert_not_called()
        node.initSpotAnnouncing.assert_not_called()

    def test_decoder_set_up_for_the_input(self):
        """(b) the decoder gets size, depth, pixel format and codec of the input."""
        node = _makeNode()
        flow = MagicMock()
        flow.buildReceiver.return_value.props.return_value = PROPS

        node._onSubscribe(flow, 5)

        node._decoder.setup.assert_called_once_with({"w": 640, "h": 480}, 3, "UINT8_T", "JPG")


class TestRate(unittest.TestCase):
    """(c) rate of the outputs."""

    def test_outputs_take_the_input_rate(self):
        """(c) input 90 Hz -> both outputs declare 90 Hz, as ImageGeometry (the model sets the real pace)."""
        node = _makeNode()
        node._updateOutputs({"rate": 90.0, "size": PROPS["size"]})
        node.service().channelByID().setProperties.assert_called_with({"rate": 90.0})
        self.assertEqual(node.service().channelByID().setProperties.call_count, 2)

    def test_image_size_is_a_channel_property(self):
        """(c) GazeDirection carries the input frame size as a property, not in every sample."""
        node = _makeNode()
        node._updateOutputs({"rate": 30.0, "size": {"w": 640, "h": 480}})
        node.service().channelByID().setProperties.assert_any_call({"rate": 30.0, "imageSize": {"w": 640, "h": 480}})

    def test_slower_input_keeps_its_rate(self):
        """(c) input 10 Hz -> 10."""
        node = _makeNode()
        node._updateOutputs({"rate": 10.0, "size": PROPS["size"]})
        node.service().channelByID().setProperties.assert_called_with({"rate": 10.0})


class TestEstimate(unittest.TestCase):
    """(d) publishing an estimate."""

    def test_sample_published_with_ts_and_pack_id(self):
        """(d) the GazeDirection JSON is the engine sample; JSON and dots carry the frame ts and packID."""
        node = _makeNode()
        sample = _sample()

        node._onEstimate({"sample": sample, "ts": 1000, "packID": 42})

        args = node.service().publishString.call_args[0]
        self.assertEqual(args[0], 11)
        self.assertEqual(args[2:], (1000, 42))
        self.assertEqual(json.loads(args[1]), sample)
        self.assertEqual(node.service().publish.call_args[0][2:], (1000, 42))
        self.assertFalse(node._inFlight)

    def test_no_face_publishes_no_dots(self):
        """(d) a "no_face" sample goes on GazeDirection, nothing on GazePoints; the ticker is free again."""
        node = _makeNode()
        node._onEstimate({"sample": {"status": "no_face"}, "ts": 1, "packID": 1})
        node.service().publishString.assert_called_once()
        node.service().publish.assert_not_called()
        self.assertFalse(node._inFlight)


class TestInputProperties(unittest.TestCase):
    """(e) input channel property changes."""

    def _node(self):
        node = _makeNode()
        node._receiver = MagicMock()
        node._receiver.chanID.return_value = 5
        return node

    def test_resource_changed_sets_the_decoder_up_again(self):
        """(e) RESOURCE_CHANGED of the input channel -> new decoder setup and output rate."""
        node = self._node()
        flow = MagicMock()
        flow.channelProperties.return_value = {**PROPS, "size": {"w": 1280, "h": 720}}

        node._onInputProperties(flow, FlowEvent.RESOURCE_CHANGED, "cam", "VideoStream", FlowResource.CHANNEL, 5, 0, None)

        node._decoder.setup.assert_called_once_with({"w": 1280, "h": 720}, 3, "UINT8_T", "JPG")
        node.service().channelByID().setProperties.assert_called_with({"rate": 30})

    def test_other_channel_is_ignored(self):
        """(e) a change of another channel -> nothing."""
        node = self._node()
        node._onInputProperties(MagicMock(), FlowEvent.RESOURCE_CHANGED, "cam", "x", FlowResource.CHANNEL, 9, 0, None)
        node._decoder.setup.assert_not_called()


class TestDataCome(unittest.TestCase):
    """(f) frames sent to the ticker."""

    def _receiver(self, *packs):
        receiver = MagicMock()
        queue = list(packs)
        receiver.next.side_effect = lambda: queue.pop(0) if queue else None
        receiver.isEmpty.side_effect = lambda: not queue
        return receiver

    def test_frame_is_decoded_and_sent(self):
        """(f) a free ticker: the frame is decoded in the node and goes to the ticker."""
        node = _makeNode()
        node._inFlight = False
        node._decoder.decode.return_value = "image"

        node._onDataCome(self._receiver(_pack(ts=2, packID=2)))

        node._ticker.loop().invokeSlot.assert_called_once_with(node._ticker.infer, {"image": "image", "ts": 2, "packID": 2})
        self.assertTrue(node._inFlight)

    def test_nobody_reading_skips_the_frame(self):
        """(f) no subscriber on either output -> no decoding, no inference (as ImageGeometry)."""
        node = _makeNode()
        node._inFlight = False
        node.service().channelByID.return_value.isEmpty.return_value = True
        node._onDataCome(self._receiver(_pack()))
        node._decoder.decode.assert_not_called()
        node._ticker.loop().invokeSlot.assert_not_called()

    def test_busy_ticker_drops_the_frame(self):
        """(f) a frame in flight -> the new one is dropped, not decoded."""
        node = _makeNode()
        node._onDataCome(self._receiver(_pack()))
        node._decoder.decode.assert_not_called()
        node._ticker.loop().invokeSlot.assert_not_called()

    def test_free_ticker_takes_the_next_frame_at_once(self):
        """(f) no rate limit of its own: a frame right after the previous estimate is sent."""
        node = _makeNode()
        node._inFlight = False
        node._onDataCome(self._receiver(_pack()))
        node._onEstimate({"sample": {"status": "no_face"}, "ts": 1, "packID": 1})
        node._onDataCome(self._receiver(_pack()))
        self.assertEqual(node._ticker.loop().invokeSlot.call_count, 2)


if __name__ == "__main__":
    unittest.main()
