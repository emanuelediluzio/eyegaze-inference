"""DD-038/039 failures, inherited lifecycle and cross-host gaze clocks.

Checks intentionally fail before the bounded governance corrections. Model
weights, physical cameras and accelerator hardware are not required.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from N2uCore.dyn.eyegaze import EyeGazeTicker, GazeDirection
from N2uCore.dyn.screenpointofgaze import ScreenPointOfGaze
from N2uCore.dyn.wrk import memprovider
from N2uCore.dyn.wrk.memprovider import SystemMemoryProvider
from N2uCore.inference.gazedinoengine import GazeDinoEngine
from N2uCore.inference.screengaze import ScreenCalibration


def test_invalid_selection_logs_and_returns_neutral(monkeypatch):
    error = MagicMock()
    monkeypatch.setattr(memprovider.log, "err", error)
    assert memprovider.resolveComputeResources("unknown") is None
    assert error.call_count == 1
    assert "Invalid compute device" in error.call_args.args[0]


def test_screen_uses_the_inherited_start_once(monkeypatch):
    node = ScreenPointOfGaze()
    node._service = MagicMock()
    node._service.flowPack.return_value = {"GazeDirection": {"node": "Gaze", "channel": "GazeDirection"}}
    node.start = MagicMock()
    monkeypatch.setattr("N2uCore.dyn.screenpointofgaze.WorkerFlowInput", MagicMock())
    assert node._onSetup()
    node.initialized.trigger([])
    node.start.assert_called_once_with()


def test_clock_offset_does_not_discard_fresh_gaze(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("N2uCore.dyn.screenpointofgaze.time.monotonic", lambda: clock[0])
    node = ScreenPointOfGaze()
    sample = {
        "schema": "gaze-direction.v1",
        "status": "ok",
        "yaw_rad": 0.1,
        "pitch_rad": 0.2,
        "source_age_secs": 0.1,
    }
    assert node._process(sample, 1_000_000)["status"] == "uncalibrated"
    clock[0] += 0.1
    assert node._process(sample, 1_100_000)["status"] == "uncalibrated"
    assert node._process(sample, 1_100_000)["status"] == "stale_input"
    clock[0] += 3
    assert node._process(sample, 1_200_000)["status"] == "stale_input"


def test_target_clock_is_derived_from_latest_source_sample(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("N2uCore.dyn.screenpointofgaze.time.monotonic", lambda: clock[0])
    node = ScreenPointOfGaze()
    node._service = MagicMock()
    sample = {"schema": "gaze-direction.v1", "status": "ok", "yaw_rad": 0.1, "pitch_rad": 0.2}
    node._process(sample, 1_000_000)
    node.beginCalibration(None, {"screen_w": 1000, "screen_h": 600})
    node.calibrationTarget(None, {"x": 500, "y": 300})
    for k in range(3):
        clock[0] = 101.1 + k * 0.2
        node._process(sample, 2_100_000 + k * 200_000)
    node.finishTarget(None, {})
    assert node._service.sendResponse.call_args.args[1] is True
    assert node._cal.status()["targets"] == 1


@pytest.mark.parametrize("timestamps", [
    [1_000_000, 2_100_000, 2_200_000],
    [1_000_000, 1_100_000, 2_200_000],
])
def test_screen_recovers_when_source_clock_offset_decreases(monkeypatch, timestamps):
    clock = iter([100.0, 100.1, 100.2])
    monkeypatch.setattr("N2uCore.dyn.screenpointofgaze.time.monotonic", lambda: next(clock))
    node = ScreenPointOfGaze()
    sample = {"schema": "gaze-direction.v1", "status": "ok", "yaw_rad": .1,
              "pitch_rad": .2, "source_age_secs": .1}
    assert [node._process(sample, ts)["status"] for ts in timestamps] == ["uncalibrated"] * 3
    assert node._lastTs == timestamps[-1]


def test_rebased_clock_still_rejects_replay_processing_age_and_backlog(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("N2uCore.dyn.screenpointofgaze.time.monotonic", lambda: clock[0])
    node = ScreenPointOfGaze()
    sample = {"schema": "gaze-direction.v1", "status": "ok", "yaw_rad": .1,
              "pitch_rad": .2, "source_age_secs": .1}
    assert node._process(sample, 1_000_000)["status"] == "uncalibrated"
    clock[0] = 100.1
    assert node._process(sample, 2_100_000)["status"] == "uncalibrated"
    assert node._process(sample, 2_100_000)["status"] == "stale_input"
    assert node._process({**sample, "source_age_secs": 3}, 2_200_000)["status"] == "stale_input"
    clock[0] = 103.1
    assert node._process(sample, 2_200_000)["status"] == "stale_input"
    clock[0] = 103.2
    assert node._process(sample, 5_200_000)["status"] == "uncalibrated"


def test_ticker_does_not_accept_a_failed_engine_load(monkeypatch):
    engine = MagicMock()
    engine.load.return_value = False
    monkeypatch.setattr("N2uCore.dyn.eyegaze.GazeDinoEngine", MagicMock(return_value=engine))
    ticker = EyeGazeTicker(SystemMemoryProvider(), "/unused", "cpu", 600)
    assert ticker._loadModel() is False
    assert ticker._engine is None
    engine.close.assert_called_once()


def test_calibration_commands_send_failures_without_overwriting_previous_fit():
    node = ScreenPointOfGaze()
    node._service = MagicMock()
    existing = {"cx": [0] * 6, "cy": [0] * 6, "screen_w": 1000, "screen_h": 600}
    node._cal._calibration = existing
    node.finishTarget(None, {})
    assert node._service.sendResponse.call_args.args[1] is False
    assert node._cal.calibration() is existing
    node.beginCalibration(None, {"screen_w": 0, "screen_h": 600})
    assert node._service.sendResponse.call_args.args[1] is False
    assert not node._cal.status()["collecting"]
    assert node._cal.calibration() is existing


def test_slow_model_result_is_rejected_without_wall_clock_comparison(monkeypatch):
    clock = [20.0]
    monkeypatch.setattr("N2uCore.dyn.screenpointofgaze.time.monotonic", lambda: clock[0])
    node = ScreenPointOfGaze()
    sample = {
        "schema": "gaze-direction.v1",
        "status": "ok",
        "yaw_rad": 0.1,
        "pitch_rad": 0.2,
        "source_age_secs": 3.0,
    }
    assert node._process(sample, 1_000_000)["status"] == "stale_input"
    sample["source_age_secs"] = 0.1
    clock[0] += 0.1
    assert node._process(sample, 1_100_000)["status"] == "uncalibrated"


def test_failure_of_external_engine_is_logged_and_neutral(monkeypatch):
    error = MagicMock()
    monkeypatch.setattr("N2uCore.inference.gazedinoengine.log.err", error)
    engine = GazeDinoEngine("/unused")
    engine._mesh = MagicMock()
    engine._mesh.process.side_effect = RuntimeError("external tracker failed")
    assert engine.analyze(np.zeros((10, 10, 3), dtype=np.uint8)) is None
    assert "external tracker failed" in error.call_args.args[0]
