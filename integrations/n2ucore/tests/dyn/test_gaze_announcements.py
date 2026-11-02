"""Input-dependent discovery follows the other processing nodes."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from N2uCore.dyn import eyegaze, screenpointofgaze


@pytest.mark.parametrize("module,nodeClass,inputName", [
    (eyegaze, eyegaze.GazeDirection, "VideoStream"),
    (screenpointofgaze, screenpointofgaze.ScreenPointOfGaze, "GazeDirection"),
])
def test_announces_only_after_subscription_and_resumes_after_reconnection(monkeypatch, module, nodeClass, inputName):
    value = nodeClass()
    value._module, value._class = module.__name__, nodeClass.__name__
    value._props = {"device": "CPU", "maxFps": 15} if module is eyegaze else {}
    value._service = MagicMock()
    value._service.flowPack.return_value = {inputName: {"node": "Source", "channel": inputName}}
    value.isRunning = lambda: True
    flow = MagicMock()
    monkeypatch.setattr(module, "WorkerFlowInput", lambda service: flow)
    if module is eyegaze:
        monkeypatch.setattr(eyegaze.fs, "isFile", lambda path: True)
        monkeypatch.setattr(eyegaze, "resolveComputeResources", lambda **kwargs: ("cpu", MagicMock()))
        monkeypatch.setattr(eyegaze.MlDevice, "_onInit", lambda self: True)

    state = {"announcing": False, "discovering": False}
    spot = MagicMock()
    value._spot = spot
    spot.isSendingEnabled.side_effect = lambda: state["announcing"]

    def startAnnouncing(**kwargs):
        state["announcing"] = True
        return True

    spot.startSendingAnnounce.side_effect = startAnnouncing
    spot.stopSendingAnnounce.side_effect = lambda: state.update(announcing=False)
    spot.startGrabbingAnnounce.side_effect = lambda: state.update(discovering=True) or True
    spot.stopGrabbingAnnounce.side_effect = lambda: state.update(discovering=False)

    assert value._onSetup()
    assert not value._autoAnnounce
    assert value._onInit()
    assert state == {"announcing": False, "discovering": True}

    value._onSubscribe(flow, 1)
    assert state == {"announcing": True, "discovering": False}
    for _ in range(2):
        value._onDisable(flow)
        assert state == {"announcing": False, "discovering": True}
        assert value._receiver is None
        value._onSubscribe(flow, 1)
        assert state == {"announcing": True, "discovering": False}
    assert spot.startSendingAnnounce.call_count == 3
    assert spot.stopSendingAnnounce.call_count == 2
