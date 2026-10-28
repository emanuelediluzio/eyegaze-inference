"""The existing GAZE converter must consume calibrated screen points unchanged.

The producer keeps browser CSS pixels for the page UI and exposes the normalized
GAZE point used by Tobii consumers. Missing physical pupil data remains missing.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from N2uCore.dyn.modalinference import DataConverter
from N2uCore.dyn.screenpointofgaze import ScreenPointOfGaze
from N2uCore.miniflow.channel import FlowChannelType

CALIBRATION = {
    'cx': [500, 100, 0, 0, 0, 0],
    'cy': [300, 0, 100, 0, 0, 0],
    'screen_w': 1000,
    'screen_h': 600,
}


def publish(yaw=.2, pitch=-.1, status='ok', calibrated=True):
    node = ScreenPointOfGaze()
    node._service = MagicMock()
    node._output = 7
    if calibrated:
        node._cal._calibration = CALIBRATION
    sample = node._process({'schema': 'gaze-direction.v1', 'status': status,
                            'yaw_rad': yaw, 'pitch_rad': pitch}, 1234567)
    node._publish(sample, 1234567)
    channel, data, timestamp = node._service.publishString.call_args.args
    assert channel == 7 and timestamp == 1234567
    return json.loads(data), SimpleNamespace(data=data.encode(), ts=timestamp)


def test_existing_gaze_converter_reads_normalized_screen_coordinates():
    sample, packet = publish()
    assert sample['gaze2d'] == pytest.approx([.52, 290 / 600])
    assert sample['screen_xy_px'] == [520, 290]
    assert sample['gaze2d_coordinate_space'] == 'viewport_normalized'
    actual = DataConverter().convert(FlowChannelType.GAZE, packet)
    tobii = SimpleNamespace(ts=packet.ts, data=json.dumps({
        'gaze2d': sample['gaze2d'],
        'eyeleft': {'pupildiameter': None},
        'eyeright': {'pupildiameter': None},
    }).encode())
    expected = DataConverter().convert(FlowChannelType.GAZE, tobii)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(actual[2:4], sample['gaze2d'])
    assert np.isnan(actual[4:]).all()
    assert sample['eyeleft']['pupildiameter'] is None
    assert sample['eyeright']['pupildiameter'] is None


@pytest.mark.parametrize('status', ['no_face', 'model_ready', 'model_unavailable', 'error'])
def test_unavailable_points_are_missing_for_existing_consumers(status):
    sample, packet = publish(status=status)
    assert sample['gaze2d'] is None and sample['screen_xy_px'] is None
    assert np.isnan(DataConverter().convert(FlowChannelType.GAZE, packet)[2:]).all()


def test_uncalibrated_output_does_not_fabricate_a_normalized_point():
    sample, packet = publish(calibrated=False)
    assert sample['status'] == 'uncalibrated'
    assert sample['gaze2d'] is None
    assert np.isnan(DataConverter().convert(FlowChannelType.GAZE, packet)[2:]).all()


@pytest.mark.parametrize('yaw,pitch,expected', [(100, 100, [.999, 599 / 600]), (-100, -100, [0., 0.])])
def test_viewport_boundaries_use_width_and_height_without_changing_pixels(yaw, pitch, expected):
    sample, _ = publish(yaw, pitch)
    assert sample['gaze2d'] == pytest.approx(expected)
    assert all(0 <= value <= 1 for value in sample['gaze2d'])


def test_disconnect_clears_point_and_preserves_gaze_json_shape():
    node = ScreenPointOfGaze()
    node._service = MagicMock()
    node._receiver = MagicMock()
    node.isRunning = lambda: True
    node._onDisable(None)
    sample = json.loads(node._service.publishString.call_args.args[1])
    assert sample['status'] == 'input_disconnected'
    assert sample['gaze2d'] is None
    assert sample['eyeleft']['pupildiameter'] is None
    assert sample['eyeright']['pupildiameter'] is None
