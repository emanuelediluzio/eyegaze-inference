import {test} from 'node:test';
import assert from 'node:assert/strict';
import {ScreenGazePanel} from '../../N2uCore/res/web/iride/lib/ui/forms/screen-gaze-panel.js';
import {Element,environment,find,descendants,until,point,calibration} from './gaze-dom.mjs';
function setup(send){const env=environment();const root=new Element('root');const panel=new ScreenGazePanel({node:{settings:{name:'Screen'}},renderer:env.renderer,send,alive:true});panel.render(root);panel.update({...point,status:'uncalibrated',screen_xy_px:null});return {env,root,panel,close(){panel.release();env.restore();}};}
test('guided calibration routes 16 targets via uinode and replaces only on success',async()=>{
 const calls=[];const h=setup(async(node,cmd,args)=>{calls.push({node,cmd,args});return {status:true,val:cmd==='fitCalibration'?{fit_residual_px:2}:cmd==='getCalibration'?calibration:null};});
 try{find(h.root,'Calibra 16 punti').click();await until(()=>calls.some(c=>c.cmd==='getCalibration'));await new Promise(r=>setImmediate(r));
 assert.equal(calls.filter(c=>c.cmd==='calibrationTarget').length,16);assert.equal(calls.filter(c=>c.cmd==='finishTarget').length,16);
 assert.deepEqual(calls.find(c=>c.cmd==='beginCalibration').args,{screen_w:1000,screen_h:600});
 assert(!calls.some(c=>c.cmd==='clearCalibration'));assert.equal(find(h.root,'Cancella calibrazione'),null);
 }finally{h.close();}
});
test('closing during calibration cancels without fitting or deleting existing calibration',async()=>{
 const calls=[];let h;h=setup(async(_node,cmd)=>{calls.push(cmd);if(cmd==='calibrationTarget')h.panel.release();return {status:true,val:null};});
 try{find(h.root,'Calibra 16 punti').click();await until(()=>calls.includes('cancelCalibration'));assert(!calls.includes('fitCalibration'));assert(!calls.includes('clearCalibration'));}finally{h.close();}
});
test('loss of face disables calibration and clears visible point',async()=>{
 const h=setup(async(_node,cmd)=>({status:true,val:cmd==='getCalibration'?calibration:null}));
 try{h.panel.attach();await new Promise(r=>setImmediate(r));h.panel.update(point);const toggle=descendants(h.root,'input').find(i=>i.type==='checkbox');toggle.checked=true;toggle.events.change();
 const dot=document.body.children.find(c=>c.className==='gaze-page-dot');assert.equal(dot.hidden,false);
 h.panel.update({...point,status:'no_face',screen_xy_px:null});assert.equal(dot.hidden,true);assert.equal(find(h.root,'Calibra 16 punti').disabled,true);
 }finally{h.close();}
});
test('resizing invalidates point coordinates and release unsubscribes and removes overlay',async()=>{
 const h=setup(async(_node,cmd)=>({status:true,val:cmd==='getCalibration'?calibration:null}));
 try{h.panel.attach();await new Promise(r=>setImmediate(r));h.panel.update(point);const toggle=descendants(h.root,'input').find(i=>i.type==='checkbox');toggle.checked=true;toggle.events.change();
 const dot=document.body.children.find(c=>c.className==='gaze-page-dot');assert(!dot.hidden);
 window.innerWidth=1200;window.events.resize();assert(dot.hidden);assert(toggle.disabled);
 h.panel.release();assert.equal(h.env.views.size,0);assert(!dot.isConnected);assert.equal(window.events.resize,undefined);
 }finally{h.close();}
});
test('invalid or nonfinite screen coordinates never display a point',async()=>{
 const h=setup(async(_node,cmd)=>({status:true,val:cmd==='getCalibration'?calibration:null}));
 try{h.panel.attach();await new Promise(r=>setImmediate(r));const toggle=descendants(h.root,'input').find(i=>i.type==='checkbox');toggle.checked=true;toggle.events.change();
 for(const screen_xy_px of [[NaN,12],[-1,20],[1000,10],[20,Infinity]]){h.panel.update({...point,screen_xy_px});assert(document.body.children.find(c=>c.className==='gaze-page-dot').hidden);}
 }finally{h.close();}
});
test('stalled stream hides page point even if the last sample was valid',async()=>{
 const h=setup(async(_node,cmd)=>({status:true,val:cmd==='getCalibration'?calibration:null}));
 try{h.panel.attach();await new Promise(r=>setImmediate(r));h.panel.update(point);const toggle=descendants(h.root,'input').find(i=>i.type==='checkbox');toggle.checked=true;toggle.events.change();
 const dot=document.body.children.find(c=>c.className==='gaze-page-dot');assert(!dot.hidden);await new Promise(r=>setTimeout(r,1900));assert(dot.hidden);
 }finally{h.close();}
});
test('hidden page cancels calibration and cannot retain a gaze dot',async()=>{
 const calls=[];let h;h=setup(async(_node,cmd)=>{calls.push(cmd);if(cmd==='calibrationTarget'){document.hidden=true;document.events.visibilitychange();}return {status:true,val:null};});
 try{find(h.root,'Calibra 16 punti').click();await until(()=>calls.includes('cancelCalibration'));assert(!calls.includes('fitCalibration'));assert(document.body.children.find(c=>c.className==='gaze-page-dot').hidden);}finally{h.close();}
});
