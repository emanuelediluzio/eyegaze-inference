"""Isolated Iride WS -> uinode -> sysinit -> CvCamera replay -> GazeDINO -> WS.

Uses a public face image repeated in a local video; tests transport and inference,
not angular accuracy. Leaves existing Iride services and physical cameras alone.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import queue
import time

import cv2
import websocket

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

class WS:
    def __init__(self,url):
        self.ws=websocket.create_connection(url,timeout=2)
        self.seq=0
        self.inbox=queue.Queue()
        self.closed=False
        self.reader=threading.Thread(target=self._read,daemon=True)
        self.reader.start()
    def _read(self):
        # Both sockets must keep answering pings even while the other is busy.
        while not self.closed:
            try:self.inbox.put(self.ws.recv())
            except websocket.WebSocketTimeoutException:continue
            except Exception as exc:
                if not self.closed:self.inbox.put(exc)
                return
    def recv(self,timeout=2):
        raw=self.inbox.get(timeout=timeout)
        if isinstance(raw,Exception):raise raw
        return raw
    def call(self,cmd,args=None):
        self.seq+=1;rid=self.seq
        self.ws.send(json.dumps({'cmd':cmd,'args':args or {},'id':rid}))
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            try:raw=self.recv()
            except queue.Empty:continue
            if isinstance(raw,bytes):
                if not raw.startswith(b'{'):continue
                raw=raw.decode('utf-8')
            msg=json.loads(raw)
            if msg.get('id')==rid:
                if not msg.get('status'):raise RuntimeError(msg)
                return msg.get('val')
        raise TimeoutError(cmd)
    def close(self):
        self.closed=True
        self.ws.close()
        self.reader.join(timeout=3)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-url',default='',help='Use an existing IDLE Iride backend instead of creating isolated services')
    parser.add_argument('--sysinit',default='SysInit',help='Executor for the live-server test')
    parser.add_argument('--palette-defaults',action='store_true',help='Use GazeDirection defaults from the actual server catalogue')
    parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=ROOT/'examples/eyegaze/iride-integration-test.json')
    args=parser.parse_args()
    frame=cv2.imread(str(args.fixture));assert frame is not None,'Unreadable fixture'
    processes=[];logs=[];ctrl=data=None;running=False;reconnects=0
    with tempfile.TemporaryDirectory(prefix='eyegaze-iride-') as temp:
        folder=Path(temp)
        video=folder/'fixture.avi'
        frame=cv2.resize(frame,(640,480))
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),15,(640,480))
        assert writer.isOpened()
        for _ in range(900):writer.write(frame)
        writer.release()
        # Reserve free loopback ports for this test's service pair.
        holders=[]
        for _ in range(2):
            sock=socket.socket();sock.bind(('127.0.0.1',0));holders.append(sock)
        uiport,sysport=[s.getsockname()[1] for s in holders]
        for sock in holders:sock.close()
        base=json.loads((ROOT/'examples/dynamic-sketches/uinode-cfg.json').read_text())['network']
        base['address']='127.0.0.1'
        unique=secrets.token_hex(4);sysname=args.sysinit if args.server_url else 'GazeTestSys'+unique
        cfgs=[{'module':'N2uCore.dyn.sysinit','class':'Init','name':sysname,
               'network':{**base,'port':sysport},'props':{'sketches':[],
               'runDirectory':str(folder/'nodes'),'vEnvPath':str(Path(sys.executable).parent.parent)}},
              {'module':'N2uCore.dyn.uinode','class':'UINode','name':'GazeTestUI'+unique,
               'network':{**base,'port':uiport},'props':{'workingFolder':str(folder/'ui'),
               'sysinits':[{'name':sysname,'address':'127.0.0.1','port':sysport}]}}]
        env={**os.environ,'PYTHONPATH':str(ROOT)}
        server_url=args.server_url.rstrip('/') or f'ws://127.0.0.1:{uiport}'
        def control_state():
            nonlocal ctrl,reconnects
            try:return ctrl.call('getPipelineState')
            except (websocket.WebSocketException,OSError):
                ctrl.close()
                ctrl=WS(server_url+'/dashWS')
                deadline=time.monotonic()+15
                while True:
                    try:
                        ctrl.call('registerSocketRole',{'role':'ControlCollector'})
                        break
                    except RuntimeError as exc:
                        # The old socket's disconnect event may still be queued.
                        if 'ONLY one ControlCollector' not in str(exc) or time.monotonic()>deadline:
                            raise
                        time.sleep(.1)
                reconnects+=1
                return ctrl.call('getPipelineState')
        try:
            for i,cfg in enumerate([] if args.server_url else cfgs):
                path=folder/f'{i}.json';path.write_text(json.dumps(cfg))
                log=open(folder/f'{i}.log','w+');logs.append(log)
                processes.append(subprocess.Popen([sys.executable,'-u','-m','N2uCore.executor',
                    str(ROOT/'N2uCore/sketch.py'),'--config',str(path)],cwd=ROOT,env=env,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            deadline=time.monotonic()+30
            while ctrl is None:
                if any(p.poll() is not None for p in processes):raise RuntimeError('Service exited')
                try:ctrl=WS(server_url+'/dashWS')
                except OSError:
                    if time.monotonic()>deadline:raise
                    time.sleep(.2)
            assert ctrl.call('getPipelineState')['state']=='IDLE','Existing Iride pipeline is active; live test refused'
            ctrl.call('registerSocketRole',{'role':'ControlCollector'})
            deadline=time.monotonic()+30
            while not all(x.get('connected') for x in ctrl.call('getSysinits')):
                assert time.monotonic()<deadline,'sysinit did not connect'
                time.sleep(.2)
            schema=ctrl.call('getSchema')
            assert 'N2uCore.dyn.eyegaze.GazeDirection' in schema
            assert 'N2uCore.dyn.screenpointofgaze.ScreenPointOfGaze' in schema
            doc=json.loads((ROOT/'examples/eyegaze/gazedino.iride.json').read_text())
            doc['appKey']=secrets.token_hex(64)
            for node in doc['nodes']:node['sysinit']=sysname
            camera,gaze,screen=doc['nodes'];camera['settings']['name']='GazeCam'+unique
            camera['settings']['id']=str(video)
            gaze['settings']={'name':'GazeModel'+unique}
            if args.palette_defaults:
                gaze['settings']={'name':'GazeModel'+unique,**{
                    k:v['default'] for k,v in schema[gaze['type']]['PROPS_SCHEMA'].items()}}
            ctrl.call('runPipeline',{'pipeline':doc});running=True
            deadline=time.monotonic()+60
            while True:
                state=control_state()
                if state['state']=='RUNNING' and state['nodes'] and all(node['alive'] for node in state['nodes'].values()):break
                assert state['state']!='IDLE',state
                assert time.monotonic()<deadline,state
                time.sleep(.2)
            print('RUNNING: isolated pipeline',flush=True)
            data=WS(server_url+'/appWS')
            data.call('registerSocketRole',{'role':'DataCollector'})
            subscription=data.call('subscribeNodeChannel',{'nodeName':gaze['settings']['name'],'chanName':'GazeDirection'})
            assert subscription['props']['type']=='GAZE',subscription
            assert subscription['props']['schema']=='gaze-direction.v1',subscription
            samples=[];heartbeats=[];deadline=time.monotonic()+90
            while len(samples)<3 and time.monotonic()<deadline:
                t=time.perf_counter();control_state();heartbeats.append(time.perf_counter()-t)
                try:raw=data.recv()
                except queue.Empty:continue
                if not isinstance(raw,bytes):continue
                assert len(raw)>32
                stream,ts,pack,seq=struct.unpack('<QQQQ',raw[:32])
                sample=json.loads(raw[32:]);assert sample['source_ts_us']==ts
                if sample['status']=='error':raise AssertionError(sample)
                if sample['status']=='ok':
                    assert sample['yaw_rad'] is not None and sample['confidence'] is None
                    samples.append(sample)
                    print('WS gaze sample',len(samples),flush=True)
            assert len(samples)==3,'No face predictions received through Iride data WS'
            # Second node must initially report uncalibrated, then collect fresh guided calibration samples.
            screen_name=screen['settings']['name']
            screenSubscription=data.call('subscribeNodeChannel',{'nodeName':screen_name,'chanName':'ScreenPoint'})
            assert screenSubscription['props']['type']=='GAZE',screenSubscription
            assert screenSubscription['props']['schema']=='screen-point-of-gaze.v1',screenSubscription
            uncalibrated=False;deadline=time.monotonic()+20
            while time.monotonic()<deadline:
                try:raw=data.recv()
                except queue.Empty:continue
                if not isinstance(raw,bytes) or len(raw)<=32:continue
                point=json.loads(raw[32:])
                if point.get('schema')=='screen-point-of-gaze.v1' and point['status']=='uncalibrated':
                    assert point['gaze2d'] is None
                    assert point['eyeleft']['pupildiameter'] is None and point['eyeright']['pupildiameter'] is None
                    uncalibrated=True;break
            assert uncalibrated,'Screen node must not fabricate uncalibrated pixels'
            def nodeCommand(cmd,args=None):
                return ctrl.call('nodeCommand',{'nodeName':screen_name,'cmd':cmd,'args':args or {}})
            assert nodeCommand('getCalibration') is None
            nodeCommand('beginCalibration',{'screen_w':1000,'screen_h':600})
            nodeCommand('calibrationTarget',{'x':500.,'y':300.})
            time.sleep(3)
            collected=nodeCommand('finishTarget')
            assert collected['targets']==1 and collected['samples']>=3
            nodeCommand('cancelCalibration')
            assert nodeCommand('getCalibration') is None
            ctrl.call('stopPipeline');deadline=time.monotonic()+30
            while control_state()['state']!='IDLE':
                assert time.monotonic()<deadline,'Stop did not reach IDLE'
                time.sleep(.2)
            running=False
            args.out.write_text(json.dumps({'passed':True,'transport':'WebSocket -> uinode -> sysinit -> CvCamera -> Flow -> GazeDirection/MlTicker -> Flow -> ScreenPointOfGaze -> Flow -> uinode -> WebSocket',
                'physical_camera':False,'accuracy_evaluated':False,'fixture':args.fixture.name,
                'checkpoint':'models/gazedino/model.joblib','samples':samples,'calibration_target':collected,'uncalibrated_verified':uncalibrated,'gaze_json_contract_verified':True,'guided_collection_verified':True,'max_control_rtt_ms':max(heartbeats)*1000,
                'control_reconnects':reconnects,'server_url':server_url,'sysinit':sysname,
                'palette_defaults':args.palette_defaults,'stopped_to_idle':True},indent=2)+'\n')
            print('PASS: GAZE direction and screen streams, guided collection and cancellation; stopped to IDLE',flush=True)
        finally:
            if ctrl and running:
                try:ctrl.call('stopPipeline')
                except Exception:pass
            if data:data.close()
            if ctrl:ctrl.close()
            for proc in reversed(processes):
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGINT)
            for proc in reversed(processes):
                try:proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=5)
            for index,log in enumerate(logs):
                log.flush();log.seek(0)
                if sys.exc_info()[0] is not None:
                    content=log.read()
                    Path(tempfile.gettempdir(),f'eyegaze-service-{index}-last.log').write_text(content)
                    print(content[-9000:],file=sys.stderr)
                log.close()

if __name__=='__main__':main()
