"""Gaze regression semantics, bounded streaming and stale-result isolation."""
import copy
import json
import subprocess
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import numpy as np
import pytest
from N2uCore.dyn.eyegaze import EyeGaze, EyeGazeTicker
from N2uCore.dyn.wrk.memprovider import AcceleratorMemoryProvider, CudaMemoryProvider, SystemMemoryProvider
from N2uCore.inference.gazedinoengine import GazeDinoEngine
from N2uCore.inference.screengaze import screenPoint
from N2uCore.validator import JsonValidator
from N2uCore.dyn.uinode import PipelinePlan

ROOT = Path(__file__).resolve().parents[2]
KEY = 'N2uCore.dyn.eyegaze.GazeDirection'

def props():
    schema=json.loads((ROOT/'N2uCore/res/dyndefs.json').read_text())[KEY]['PROPS_SCHEMA']
    return {key: value['default'] for key,value in schema.items()}

def engine():
    e=GazeDinoEngine('/unused')
    e._mesh=MagicMock()
    e._mesh.process.return_value=NS(multi_face_landmarks=[NS(landmark=[NS(x=.1,y=.2),NS(x=.9,y=.8)])])
    e._predictCrop=MagicMock(return_value=np.array([.2,-.1]))
    return e

def test_face_angles_and_no_pupil_or_fake_confidence():
    sample=engine().analyze(np.zeros((100,100,3),np.uint8))
    assert sample['status']=='ok'
    assert sample['yaw_deg']==pytest.approx(np.degrees(.2))
    assert sample['bbox_xyxy_px']==[10,20,90,80]
    assert 'screen_xy_px' not in sample and sample['confidence'] is None
    assert np.linalg.norm(sample['direction_xyz'])==pytest.approx(1.0)
    assert 'gaze2d' not in sample and 'eyeleft' not in sample
    json.dumps(sample,allow_nan=False)

def test_missing_face_clears_prediction_and_smoothing():
    e=engine();e.analyze(np.zeros((100,100,3),np.uint8))
    e._mesh.process.return_value=NS(multi_face_landmarks=None)
    sample=e.analyze(np.zeros((100,100,3),np.uint8))
    assert sample['status']=='no_face' and sample['yaw_rad'] is None
    assert e._ema is None

def test_non_rgb_input_rejected():
    e=engine()
    assert e.analyze(np.zeros((20,20),np.uint8)) is None

def test_screen_coordinates_are_clipped_pixels():
    c={'cx':[500,100,0,0,0,0],'cy':[300,0,100,0,0,0],'screen_w':1000,'screen_h':600}
    assert screenPoint(.2,-.1,c)==[520,290]
    assert screenPoint(100,100,c)==[999,599]
    assert screenPoint(0,0,{**c,'cx':[1]}) is None

def bare_node():
    node=EyeGaze(); node.isRunning=lambda:True
    node._props=props();node._ticker=MagicMock();node._service=MagicMock()
    node._output=11
    return node

def receiver(*timestamps):
    q=deque(NS(data=b'rgb',ts=t) for t in timestamps)
    r=MagicMock();r.next.side_effect=lambda:q.popleft() if q else None
    r.isEmpty.side_effect=lambda:not q
    r.props.return_value={'depth':3}
    return r

def test_drains_to_latest_and_allows_only_one_ml_job():
    n=bare_node();n._onData(receiver(1,2,3))
    assert n._inFlight
    call=n._ticker.loop().invokeSlot.call_args.args[1]
    assert call['ts']==3 and n._dropped==2
    n._onData(receiver(4,5))
    assert n._ticker.loop().invokeSlot.call_count==1
    assert n._dropped==4

def test_stale_result_after_disconnect_is_discarded():
    n=bare_node();n._generation=4;n._inFlight=True
    n._onEstimate({'generation':3,'sample':{'status':'ok'},'ts':123})
    assert not n._inFlight
    n._service.publishString.assert_not_called()

def test_capture_timestamp_is_preserved():
    n=bare_node()
    n._onEstimate({'generation':0,'sample':{'status':'no_face'},'ts':123456})
    args=n._service.publishString.call_args.args
    assert args[2]==123456 and json.loads(args[1])['source_ts_us']==123456

def test_ticker_reports_error_and_always_completes():
    t=EyeGazeTicker(SystemMemoryProvider(), "/unused", "cpu", 600);t._ensureLoaded=MagicMock(side_effect=RuntimeError('load failed'))
    t.estimate=MagicMock()
    t.infer(b'',{},123,7)
    pack=t.estimate.trigger.call_args.args[0][0]
    assert pack['sample']['status']=='error' and pack['generation']==7
    t.infer(b'',{},124,7)
    assert t._ensureLoaded.call_count==1 # backoff prevents repeated model loads

def test_load_trigger_frame_is_not_used_for_a_stale_prediction():
    t=EyeGazeTicker(SystemMemoryProvider(), "/unused", "cpu", 600);t._engine=MagicMock();t.estimate=MagicMock()
    def load():
        t._loaded=True
        return True
    t._ensureLoaded=load
    t.infer(b'old frame',{},123,7)
    t._engine.analyze.assert_not_called()
    pack=t.estimate.trigger.call_args.args[0][0]
    assert pack['sample']['status']=='model_ready'

def test_schema_and_pipeline_contract():
    definitions=json.loads((ROOT/'N2uCore/res/dyndefs.json').read_text())
    v=JsonValidator();v.setSchema(definitions[KEY]['PROPS_SCHEMA'])
    p=props();assert v.validate(p)
    assert definitions[KEY]['outputs']==[{'name':'GazeDirection','type':'GAZE'}]
    doc=json.loads((ROOT/'examples/eyegaze/gazedino.iride.json').read_text())
    plan=PipelinePlan(definitions,['SysInit'])
    assert plan.build(doc),plan.error()

def test_display_metadata_contains_only_frame_size_and_two_iris_centers():
    e = engine()
    landmarks = [NS(x=.1, y=.2), NS(x=.9, y=.8)] + [NS(x=.5, y=.5) for _ in range(476)]
    e._mesh.process.return_value = NS(multi_face_landmarks=[NS(landmark=landmarks)])
    sample = e.analyze(np.zeros((100, 200, 3), np.uint8))
    assert sample['image_size_px'] == [200, 100]
    assert sample['eyes_px'] == [[100, 50], [100, 50]]
    assert len(sample['eyes_px']) == 2

def test_default_rate_accepts_fresh_frame_within_old_quarter_second_limit(monkeypatch):
    n=bare_node()
    ticks=iter([10.,10.1]);monkeypatch.setattr('N2uCore.dyn.eyegaze.time.monotonic',lambda:next(ticks))
    n._onData(receiver(1));n._inFlight=False;n._onData(receiver(2))
    assert n._ticker.loop().invokeSlot.call_count==2
    assert n._props == {"device": "auto", "maxFps": 15}


def test_head_roll_uses_pixel_aspect_ratio_and_does_not_change_gaze():
    e = engine()
    points = [NS(x=.5, y=.5) for _ in range(478)]
    points[0], points[1] = NS(x=.1,y=.1), NS(x=.9,y=.9)
    points[33], points[263] = NS(x=.3,y=.4), NS(x=.7,y=.6)
    e._mesh.process.return_value = NS(multi_face_landmarks=[NS(landmark=points)])
    sample = e.analyze(np.zeros((100,200,3),np.uint8))
    assert sample['roll_rad'] == pytest.approx(np.arctan2(20,80))
    assert sample['roll_deg'] == pytest.approx(np.degrees(np.arctan2(20,80)))
    assert sample['roll_source'] == 'head_landmarks'
    assert sample['yaw_rad'] == pytest.approx(.2)
    assert sample['pitch_rad'] == pytest.approx(-.1)
    points[33], points[263] = NS(x=.3,y=.6), NS(x=.7,y=.4)
    assert e.analyze(np.zeros((100,200,3),np.uint8))['roll_rad'] < 0
    e._mesh.process.return_value = NS(multi_face_landmarks=None)
    missing = e.analyze(np.zeros((100,200,3),np.uint8))
    assert missing['roll_rad'] is None and missing['roll_deg'] is None


def test_degenerate_eye_axis_never_fabricates_a_head_roll():
    e = engine()
    points = [NS(x=.5,y=.5) for _ in range(478)]
    points[0],points[1] = NS(x=.1,y=.2),NS(x=.9,y=.8)
    e._mesh.process.return_value = NS(multi_face_landmarks=[NS(landmark=points)])
    sample = e.analyze(np.zeros((100,100,3),np.uint8))
    assert sample['status'] == 'ok' and sample['roll_rad'] is None


def test_node_and_ticker_construction_do_not_load_weights_or_face_tracker(monkeypatch):
    hubLoad = MagicMock()
    faceMesh = MagicMock()
    monkeypatch.setattr('N2uCore.inference.gazedinoengine.torch.hub.load',hubLoad)
    monkeypatch.setattr('N2uCore.inference.gazedinoengine.mp.solutions.face_mesh.FaceMesh',faceMesh)
    node = EyeGaze()
    ticker = EyeGazeTicker(SystemMemoryProvider(), "/unused", "cpu", 600)
    assert node._ticker is None and not node._inFlight
    assert ticker._engine is None and not ticker._loaded
    hubLoad.assert_not_called()
    faceMesh.assert_not_called()


@pytest.mark.parametrize('artifactKind', ['missing', 'directory', 'file'])
def test_setup_checks_fixed_artifact_without_loading_model(tmp_path, monkeypatch, artifactKind):
    modelPath = tmp_path / 'model.joblib'
    if artifactKind == 'directory':
        modelPath.mkdir()
    elif artifactKind == 'file':
        modelPath.write_bytes(b'loading remains in the ticker')
    monkeypatch.setattr('N2uCore.dyn.eyegaze.MlDevice._onSetup', lambda self: True)
    monkeypatch.setattr('N2uCore.dyn.eyegaze.ResourceManager', lambda: NS(absolutePath=lambda path: str(modelPath)))
    resolve = MagicMock(return_value=('cpu', SystemMemoryProvider()))
    monkeypatch.setattr('N2uCore.dyn.eyegaze.resolveComputeResources', resolve)
    flow = MagicMock()
    monkeypatch.setattr('N2uCore.dyn.eyegaze.WorkerFlowInput', flow)
    engineClass = MagicMock()
    monkeypatch.setattr('N2uCore.dyn.eyegaze.GazeDinoEngine', engineClass)
    node = bare_node()
    node._service.flowPack.return_value = {'VideoStream': {'node': 'Camera', 'channel': 'VideoStream'}}
    node.err = MagicMock()
    assert node._onSetup() is (artifactKind == 'file')
    engineClass.assert_not_called()
    if artifactKind != 'file':
        resolve.assert_not_called()
        flow.assert_not_called()
        assert str(modelPath) in node.err.call_args.args[0]


def test_missing_offline_source_never_attempts_hub_download(tmp_path,monkeypatch):
    checkpoint = tmp_path/'model.joblib';checkpoint.write_bytes(b'not loaded without source')
    hubLoad = MagicMock()
    monkeypatch.setattr('N2uCore.inference.gazedinoengine.torch.hub.get_dir',lambda:str(tmp_path))
    monkeypatch.setattr('N2uCore.inference.gazedinoengine.torch.hub.load',hubLoad)
    assert GazeDinoEngine(checkpoint).load() is False
    hubLoad.assert_not_called()


@pytest.mark.parametrize('expected,providerType',[
    ('cuda:2',CudaMemoryProvider),
    ('xpu:1',AcceleratorMemoryProvider),
    ('mps',SystemMemoryProvider),
    ('cpu',SystemMemoryProvider),
])
def test_standard_provider_and_engine_use_the_same_device(monkeypatch,expected,providerType):
    provider=providerType() if providerType is SystemMemoryProvider else providerType(expected)
    resolve=MagicMock(return_value=(expected,provider))
    monkeypatch.setattr('N2uCore.dyn.eyegaze.resolveComputeResources',resolve)
    monkeypatch.setattr('N2uCore.dyn.eyegaze.MlDevice._onSetup',lambda self:True)
    monkeypatch.setattr('N2uCore.dyn.eyegaze.WorkerFlowInput',lambda service:MagicMock())
    monkeypatch.setattr('N2uCore.dyn.eyegaze.ResourceManager',lambda:NS(absolutePath=lambda path:'/resources/'+path))
    monkeypatch.setattr('N2uCore.dyn.eyegaze.fs.isFile', lambda path: True)
    node=EyeGaze();node._props={'device':'auto','maxFps':15};node._service=MagicMock();node._service.flowPack.return_value={'VideoStream':{'node':'Camera','channel':'VideoStream'}}
    assert node._onSetup()
    args=node._tickerArgs()
    assert type(args['memProvider']) is providerType
    assert args['deviceName']==expected and args['idleTimeoutSecs']==600
    assert args['modelPath']=='/resources/models/gazedino/model.joblib'
    assert args['memProvider'] is provider
    engineClass=MagicMock()
    monkeypatch.setattr('N2uCore.dyn.eyegaze.GazeDinoEngine',engineClass)
    ticker=node._tickerClass()(**args)
    assert ticker._memProvider is args['memProvider']
    assert ticker._loadModel()
    engineClass.assert_called_once_with(args['modelPath'],deviceName=expected)
    engineClass.return_value.load.assert_called_once_with()
    resolve.assert_called_once_with(deviceName='auto',supportedDevices=GazeDinoEngine.SUPPORTED_DEVICES)


def test_standalone_engine_delegates_automatic_device_selection(tmp_path,monkeypatch):
    checkpoint=tmp_path/'model.joblib';checkpoint.touch()
    resolve=MagicMock(return_value=('cuda:2',CudaMemoryProvider('cuda:2')))
    monkeypatch.setattr('N2uCore.inference.gazedinoengine.resolveComputeResources',resolve)
    engine=GazeDinoEngine(checkpoint,hubDir=str(tmp_path/'missing-source'))
    assert engine.load() is False
    assert engine._device.type=='cuda' and engine._device.index==2
    resolve.assert_called_once_with(supportedDevices=GazeDinoEngine.SUPPORTED_DEVICES)


@pytest.mark.parametrize('maxFps', [1, 5, 30, 120])
def test_configured_fps_limits_dispatch_without_queuing_old_frames(monkeypatch, maxFps):
    node = bare_node()
    node._maxFps = maxFps
    ticks = iter([100., 100. + .5 / maxFps, 100. + 1.1 / maxFps])
    monkeypatch.setattr('N2uCore.dyn.eyegaze.time.monotonic', lambda: next(ticks))
    node._onData(receiver(1))
    node._inFlight = False
    node._onData(receiver(2))
    assert node._ticker.loop().invokeSlot.call_count == 1
    assert not node._inFlight
    node._onData(receiver(3))
    assert node._ticker.loop().invokeSlot.call_count == 2
    assert node._ticker.loop().invokeSlot.call_args.args[1]['ts'] == 3
    assert node._dropped == 1


@pytest.mark.parametrize('value', [0, -1, 121, 15.5, '15', None])
def test_invalid_fps_is_rejected_by_standard_schema(value):
    definitions = json.loads((ROOT / 'N2uCore/res/dyndefs.json').read_text())
    validator = JsonValidator()
    validator.setSchema(definitions[KEY]['PROPS_SCHEMA'])
    assert not validator.validate({'maxFps': value})


def test_setup_applies_fps_and_gaze_channel_metadata(monkeypatch):
    from N2uCore.miniflow.channel import FlowChannelType
    monkeypatch.setattr('N2uCore.dyn.eyegaze.MlDevice._onSetup', lambda self: True)
    monkeypatch.setattr('N2uCore.dyn.eyegaze.MlDevice._onInit', lambda self: True)
    monkeypatch.setattr('N2uCore.dyn.eyegaze.resolveComputeResources', lambda **kwargs: ('cpu', SystemMemoryProvider()))
    monkeypatch.setattr('N2uCore.dyn.eyegaze.WorkerFlowInput', lambda service: MagicMock())
    monkeypatch.setattr('N2uCore.dyn.eyegaze.fs.isFile', lambda path: True)
    node = bare_node()
    node._props['maxFps'] = 5
    node._service.flowPack.return_value = {'VideoStream': {'node': 'Camera', 'channel': 'VideoStream'}}
    node.spot = lambda: MagicMock()
    assert node._onSetup()
    assert node._maxFps == 5
    assert node._onInit()
    name, channelType, metadata = node._service.addChannel.call_args.args
    assert name == 'GazeDirection' and channelType == FlowChannelType.GAZE
    assert metadata['schema'] == 'gaze-direction.v1'
    assert metadata['maxFps'] == 5
    assert metadata['dependsOn'] == ['Camera.VideoStream']
