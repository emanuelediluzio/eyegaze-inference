import json
import time
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
from N2uCore.inference.screengaze import ScreenCalibration, screen_point, validate_calibration
from N2uCore.dyn.screenpointofgaze import ScreenPointOfGaze

CAL={'cx':[500,100,0,0,0,0],'cy':[300,0,100,0,0,0],'screen_w':1000,'screen_h':600}

def node():
    n=ScreenPointOfGaze();n._props={'maxSampleAgeSecs':2.0};n._service=MagicMock()
    return n

def direction(**kw):
    return {'schema':'gaze-direction.v1','status':'ok','yaw_rad':.2,'pitch_rad':-.1,**kw}

def now():return time.time_ns()//1000

def test_mapping_and_uncalibrated():
    n=node();assert n._process(direction(),now())['status']=='uncalibrated'
    n._last_ts=None;n._cal.calibration=CAL
    assert n._process(direction(),now())['screen_xy_px']==[520,290]

def test_missing_face_never_reuses_pixels():
    n=node();n._cal.calibration=CAL
    assert n._process(direction(status='no_face',yaw_rad=None,pitch_rad=None),now())['screen_xy_px'] is None

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),None,True,'1'])
def test_bad_angles(bad):
    assert node()._process(direction(yaw_rad=bad),now())['status']=='invalid_input'

def test_stale_out_of_order_wrong_schema():
    n=node();ts=now();n._process(direction(),ts)
    assert n._process(direction(),ts)['status']=='stale_input'
    assert n._process(direction(),ts-3000000)['status']=='stale_input'
    assert n._process({'schema':'gazedino.v1','status':'ok'},ts+1)['status']=='invalid_input'

def test_invalid_calibration_is_atomic():
    n=node();n._cal.calibration=CAL
    n.setCalibration(None,{'calibration':{**CAL,'cx':[float('nan')]*6}})
    assert n._cal.calibration==CAL
    assert n._service.sendResponse.call_args.args[1] is False

def test_clear_and_cancel():
    n=node();n._cal.calibration=CAL;n._cal.begin(1000,600)
    n.cancelCalibration(None,{})
    assert n._cal.calibration==CAL and n._cal.size is None
    n.clearCalibration(None,{})
    assert n._cal.calibration is None

def test_guided_fit_16_targets_and_export():
    c=ScreenCalibration();c.begin(1000,600)
    for i,x in enumerate(np.linspace(-.3,.3,4)):
        for j,y in enumerate(np.linspace(-.2,.2,4)):
            c.target(500+1000*x,300+1000*y,0,0)
            for k in range(6):c.feed(x,y,1+k*.2,1000000+k*200000)
            c.finish_target()
    r=c.fit();assert r['fit_residual_px']<1e-6
    # Held-out synthetic point verifies coefficients independently of calibration targets.
    px=screen_point(.123,.087,c.calibration)
    assert abs(px[0]-623)<=1 and abs(px[1]-387)<=1
    assert validate_calibration(json.loads(json.dumps(c.calibration)))==c.calibration

def test_degenerate_fit_keeps_old_calibration():
    c=ScreenCalibration();c.calibration=CAL;c.begin(1000,600)
    c.targets=[[.1,.1,10+i,20+i] for i in range(16)]
    with pytest.raises(ValueError):c.fit()
    assert c.calibration==CAL

def test_target_requires_new_samples_after_settling():
    c=ScreenCalibration();c.begin(1000,600);c.target(500,300,0,1000000)
    c.feed(.1,.2,.1,1100000) # before settling
    c.feed(.1,.2,1.2,1100000) # arrived late but captured before target settled
    with pytest.raises(ValueError):c.finish_target()
    assert c.targets==[]

def test_screen_node_has_no_model_dependency():
    import sys,subprocess
    output=subprocess.check_output([sys.executable,'-c',
        'import sys; from N2uCore.dyn.screenpointofgaze import ScreenPointOfGaze; '
        'assert "torch" not in sys.modules and "mediapipe" not in sys.modules; print("ok")'],text=True)
    assert output.strip()=='ok'

def test_registered_command_schemas_accept_real_payloads():
    from N2uCore.validator import JsonValidator
    root=Path(__file__).resolve().parents[2]
    defs=json.loads((root/'N2uCore/res/dyndefs.json').read_text())
    commands=defs['N2uCore.dyn.screenpointofgaze.ScreenPointOfGaze']['commands']
    for name,args in [('setCalibration',{'calibration':CAL}),('beginCalibration',{'screen_w':1000,'screen_h':600}),('calibrationTarget',{'x':500.,'y':300.})]:
        v=JsonValidator();v.setSchema(commands[name]['args'])
        assert v.validate(args),name
