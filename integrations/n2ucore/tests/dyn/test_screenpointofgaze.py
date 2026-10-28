import json
import time
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
from N2uCore.inference.screengaze import ScreenCalibration, screenPoint, validateCalibration
from N2uCore.dyn.screenpointofgaze import ScreenPointOfGaze

CAL={'cx':[500,100,0,0,0,0],'cy':[300,0,100,0,0,0],'screen_w':1000,'screen_h':600}

def node():
    n=ScreenPointOfGaze();n._props={};n._service=MagicMock()
    return n

def direction(**kw):
    return {'schema':'gaze-direction.v1','status':'ok','yaw_rad':.2,'pitch_rad':-.1,**kw}

def now():return time.time_ns()//1000

def test_mapping_and_uncalibrated():
    n=node();assert n._process(direction(),now())['status']=='uncalibrated'
    n._lastTs=None;n._cal._calibration=CAL
    assert n._process(direction(),now())['screen_xy_px']==[520,290]

def test_missing_face_never_reuses_pixels():
    n=node();n._cal._calibration=CAL
    assert n._process(direction(status='no_face',yaw_rad=None,pitch_rad=None),now())['screen_xy_px'] is None

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),None,True,'1'])
def test_bad_angles(bad):
    assert node()._process(direction(yaw_rad=bad),now())['status']=='invalid_input'

def test_stale_out_of_order_wrong_schema():
    n=node();ts=now();n._process(direction(),ts)
    assert n._process(direction(),ts)['status']=='stale_input'
    assert n._process(direction(),ts-3000000)['status']=='stale_input'
    assert n._process({'schema':'gazedino.v1','status':'ok'},ts+1)['status']=='invalid_input'

def test_cancel_keeps_current_session_calibration_and_import_commands_are_absent():
    n=node();n._cal._calibration=CAL;n._cal.begin(1000,600)
    n.cancelCalibration(None,{})
    assert n._cal.calibration()==CAL and not n._cal.status()['collecting']
    assert not hasattr(n,'setCalibration') and not hasattr(n,'clearCalibration')

def test_guided_fit_16_targets_and_projection():
    c=ScreenCalibration();c.begin(1000,600)
    for i,x in enumerate(np.linspace(-.3,.3,4)):
        for j,y in enumerate(np.linspace(-.2,.2,4)):
            c.target(500+1000*x,300+1000*y,0,0)
            for k in range(6):c.feed(x,y,1+k*.2,1000000+k*200000)
            c.finishTarget()
    r=c.fit();assert r['fit_residual_px']<1e-6
    # Held-out synthetic point verifies coefficients independently of calibration targets.
    px=screenPoint(.123,.087,c.calibration())
    assert abs(px[0]-623)<=1 and abs(px[1]-387)<=1
    assert validateCalibration(json.loads(json.dumps(c.calibration())))==c.calibration()

def test_degenerate_fit_keeps_old_calibration():
    c=ScreenCalibration();c._calibration=CAL;c.begin(1000,600)
    c._targets=[[.1,.1,10+i,20+i] for i in range(16)]
    assert c.fit() is None
    assert c._calibration==CAL

def test_target_requires_new_samples_after_settling():
    c=ScreenCalibration();c.begin(1000,600);c.target(500,300,0,1000000)
    c.feed(.1,.2,.1,1100000) # before settling
    c.feed(.1,.2,1.2,1100000) # arrived late but captured before target settled
    assert c.finishTarget() is None
    assert c._targets==[]

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
    for name,args in [('beginCalibration',{'screen_w':1000,'screen_h':600}),('calibrationTarget',{'x':500.,'y':300.})]:
        v=JsonValidator();v.setSchema(commands[name]['args'])
        assert v.validate(args),name


def test_calibration_has_a_bounded_16_target_limit():
    c=ScreenCalibration();c.begin(1000,600)
    c._targets=[[.1,.1,10+i,20+i] for i in range(16)]
    assert c.target(500,300,0,0) is None
    assert len(c._targets)==16


@pytest.mark.parametrize("count", [10, 15, 17, 32])
def test_fit_requires_exactly_16_completed_targets_and_preserves_current_fit(count):
    c=ScreenCalibration();c._calibration=CAL;c.begin(1000,600)
    c._targets=[[.1,.1,10+i,20+i] for i in range(count)]
    assert c.fit() is None
    assert c._calibration==CAL
