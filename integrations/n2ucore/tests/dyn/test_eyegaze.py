"""Gaze regression semantics, bounded streaming and stale-result isolation."""
import copy
import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import numpy as np
import pytest
from N2uCore.dyn.eyegaze import EyeGaze, EyeGazeTicker
from N2uCore.inference.gazedino import GazeDinoEngine
from N2uCore.inference.screengaze import screen_point
from N2uCore.validator import JsonValidator

ROOT = Path(__file__).resolve().parents[2]
KEY = 'N2uCore.dyn.eyegaze.GazeDirection'

def props():
    schema=json.loads((ROOT/'N2uCore/res/dyndefs.json').read_text())[KEY]['PROPS_SCHEMA']
    return {key: value['default'] for key,value in schema.items()}

def engine():
    e=GazeDinoEngine('/unused')
    e.mesh=MagicMock()
    e.mesh.process.return_value=NS(multi_face_landmarks=[NS(landmark=[NS(x=.1,y=.2),NS(x=.9,y=.8)])])
    e.predict_crop=MagicMock(return_value=np.array([.2,-.1]))
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
    e.mesh.process.return_value=NS(multi_face_landmarks=None)
    sample=e.analyze(np.zeros((100,100,3),np.uint8))
    assert sample['status']=='no_face' and sample['yaw_rad'] is None
    assert e._ema is None

def test_non_rgb_input_rejected():
    e=engine()
    with pytest.raises(ValueError,match='RGB'):
        e.analyze(np.zeros((20,20),np.uint8))

def test_screen_coordinates_are_clipped_pixels():
    c={'cx':[500,100,0,0,0,0],'cy':[300,0,100,0,0,0],'screen_w':1000,'screen_h':600}
    assert screen_point(.2,-.1,c)==[520,290]
    assert screen_point(100,100,c)==[999,599]
    with pytest.raises(ValueError): screen_point(0,0,{**c,'cx':[1]})

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
    t=EyeGazeTicker(props());t._ensureLoaded=MagicMock(side_effect=RuntimeError('load failed'))
    t.estimate=MagicMock()
    t.infer(b'',{},123,7)
    pack=t.estimate.trigger.call_args.args[0][0]
    assert pack['sample']['status']=='error' and pack['generation']==7
    t.infer(b'',{},124,7)
    assert t._ensureLoaded.call_count==1 # backoff prevents repeated model loads

def test_load_trigger_frame_is_not_used_for_a_stale_prediction():
    t=EyeGazeTicker(props());t.engine=MagicMock();t.estimate=MagicMock()
    def load():
        t._loaded=True
        return True
    t._ensureLoaded=load
    t.infer(b'old frame',{},123,7)
    t.engine.analyze.assert_not_called()
    pack=t.estimate.trigger.call_args.args[0][0]
    assert pack['sample']['status']=='model_ready'

def test_schema_and_pipeline_contract():
    definitions=json.loads((ROOT/'N2uCore/res/dyndefs.json').read_text())
    v=JsonValidator();v.setSchema(definitions[KEY]['PROPS_SCHEMA'])
    p=props();assert v.validate(p)
    assert definitions[KEY]['outputs']==[{'name':'GazeDirection','type':'TEXT'}]
    from N2uCore.dyn.uinode import PipelinePlan
    doc=json.loads((ROOT/'examples/eyegaze/gazedino.iride.json').read_text())
    plan=PipelinePlan(definitions,['SysInit'])
    assert plan.build(doc),plan.error()

def test_display_metadata_contains_only_frame_size_and_two_iris_centers():
    e = engine()
    landmarks = [NS(x=.1, y=.2), NS(x=.9, y=.8)] + [NS(x=.5, y=.5) for _ in range(476)]
    e.mesh.process.return_value = NS(multi_face_landmarks=[NS(landmark=landmarks)])
    sample = e.analyze(np.zeros((100, 200, 3), np.uint8))
    assert sample['image_size_px'] == [200, 100]
    assert sample['eyes_px'] == [[100, 50], [100, 50]]
    assert len(sample['eyes_px']) == 2
