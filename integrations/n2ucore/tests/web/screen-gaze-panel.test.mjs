import {test} from 'node:test';
import assert from 'node:assert/strict';
import {PanelSessions} from '../../N2uCore/res/web/iride/lib/ui/forms/panel-sessions.js';
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
 window.innerWidth=1200;window.events.resize();assert(dot.hidden);assert(toggle.checked);assert(!toggle.disabled);
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

function persistentSetup() {
 const env=environment(), sessions=new PanelSessions();let loads=0;
 const mount=()=>{const root=new Element('root');const panel=new ScreenGazePanel({node:{settings:{name:'Screen'}},renderer:env.renderer,alive:true,sessionFor:(key,create)=>sessions.get(key,create),send:async(_node,cmd)=>{if(cmd==='getCalibration')loads++;return {status:true,val:cmd==='getCalibration'?calibration:null};}});panel.render(root);panel.attach();return {root,panel,toggle:descendants(root,'input').find(i=>i.type==='checkbox')};};
 const emit=(sample)=>{for(const view of env.views.values())view.handlers.onInfo({topic:'text',text:JSON.stringify(sample)});};
 return {env,sessions,mount,emit,loads:()=>loads,dot:()=>document.body.children.find(c=>c.className==='gaze-page-dot'),close(){sessions.clear();env.restore();}};
}
test('closing and reopening a card preserves the page point and selected checkbox with one stream',async()=>{
 const h=persistentSetup();let a,b;
 try{a=h.mount();await new Promise(r=>setImmediate(r));h.emit(point);a.toggle.checked=true;a.toggle.events.change();const dot=h.dot();assert(!dot.hidden);
 a.panel.release();assert.equal(h.env.views.size,1);h.emit({...point,screen_xy_px:[750,400]});assert(!dot.hidden);assert.equal(dot.style.left,'750px');
 b=h.mount();await new Promise(r=>setImmediate(r));assert.equal(descendants(b.root,'select').length,0);assert(find(b.root,'Calibra 16 punti'));assert(b.toggle.checked);assert.equal(h.loads(),1);assert.equal(h.env.views.size,1);assert.equal(document.body.children.filter(c=>c.className==='gaze-page-dot').length,1);
 b.toggle.checked=false;b.toggle.events.change();assert(dot.hidden);b.panel.release();assert.equal(h.env.views.size,0);
 }finally{a?.panel.release();b?.panel.release();h.close();}
});
test('closed-card point hides temporarily on stale data, disconnect and viewport changes without losing preference',async()=>{
 const h=persistentSetup();let a,b;
 try{a=h.mount();await new Promise(r=>setImmediate(r));h.emit(point);a.toggle.checked=true;a.toggle.events.change();a.panel.release();const dot=h.dot();
 await new Promise(r=>setTimeout(r,1900));assert(dot.hidden);h.emit(point);assert(!dot.hidden);
 h.sessions.refreshAlive(()=>false);assert(dot.hidden);h.sessions.refreshAlive(()=>true);h.emit(point);assert(!dot.hidden);
 document.hidden=true;document.events.visibilitychange();assert(dot.hidden);document.hidden=false;document.events.visibilitychange();h.emit(point);assert(!dot.hidden);
 window.innerWidth=1200;window.events.resize();assert(dot.hidden);window.innerWidth=1000;window.events.resize();h.emit(point);assert(dot.hidden);
 b=h.mount();assert(b.toggle.checked);b.toggle.checked=false;b.toggle.events.change();assert(dot.hidden);
 }finally{a?.panel.release();b?.panel.release();h.close();}
});
test('stopping the pipeline clears closed-card sessions and the next run starts unchecked',async()=>{
 const h=persistentSetup();let a,b;
 try{a=h.mount();await new Promise(r=>setImmediate(r));h.emit(point);a.toggle.checked=true;a.toggle.events.change();a.panel.release();const dot=h.dot();h.sessions.clear();
 assert(!dot.isConnected);assert.equal(h.env.views.size,0);assert.equal(window.events.resize,undefined);
 b=h.mount();assert(!b.toggle.checked);await new Promise(r=>setImmediate(r));assert.equal(h.loads(),2);
 }finally{a?.panel.release();b?.panel.release();h.close();}
});
test('independent validation measures nine new targets without calibration commands and restores the dot',async()=>{
 const calls=[];const h=setup(async(_node,cmd)=>{calls.push(cmd);return {status:true,val:cmd==='getCalibration'?calibration:null};});
 const oldPerformance=globalThis.performance,oldTimeout=globalThis.setTimeout;let clock=100,ts=1000000;
 globalThis.performance={now:()=>clock};
 try{h.panel.attach();await new Promise(r=>setImmediate(r));h.panel.update({...point,source_ts_us:ts});const toggle=descendants(h.root,'input').find(i=>i.type==='checkbox');toggle.checked=true;toggle.events.change();
 globalThis.setTimeout=(fn,ms)=>oldTimeout(()=>{
  if(ms===3000){const target=document.body.children.find(c=>c.className==='gaze-calibration').children[0];const x=parseFloat(target.style.left),y=parseFloat(target.style.top);const start=clock;
   for(let i=0;i<3;i++){clock=start+1200+i*200;ts+=1200000;h.panel.update({...point,screen_xy_px:[x+3,y+4],source_ts_us:ts});}}
  fn();},ms===3000?0:ms);
 find(h.root,'Verifica precisione (9 punti)').click();await until(()=>descendants(h.root,'td').length===4);await new Promise(r=>setImmediate(r));
 assert.equal(find(h.root,'Salva risultati'),null);assert.equal(descendants(h.root,'a').length,0);assert.equal(descendants(h.root,'td')[0].textContent,'16 punti');assert.equal(descendants(h.root,'td')[1].textContent,'5.0 px');assert(!calls.some(cmd=>cmd!=='getCalibration'));assert(toggle.checked);assert(!document.body.children.find(c=>c.className==='gaze-page-dot').hidden);
 }finally{globalThis.performance=oldPerformance;globalThis.setTimeout=oldTimeout;h.close();}
});
test('cancelling validation saves no partial report and never changes calibration',async()=>{
 const calls=[];const h=setup(async(_node,cmd)=>{calls.push(cmd);return {status:true,val:cmd==='getCalibration'?calibration:null};});
 const oldTimeout=globalThis.setTimeout;
 try{h.panel.attach();await new Promise(r=>setImmediate(r));h.panel.update({...point,source_ts_us:1000000});
 globalThis.setTimeout=(fn,ms)=>oldTimeout(()=>{if(ms===3000){const overlay=document.body.children.find(c=>c.className==='gaze-calibration');overlay.children[2].click();}fn();},ms===3000?0:ms);
 find(h.root,'Verifica precisione (9 punti)').click();await until(()=>!document.body.children.some(c=>c.className==='gaze-calibration'));await new Promise(r=>setImmediate(r));
 assert.equal(descendants(h.root,'td').length,0);assert.deepEqual(calls,['getCalibration']);
 }finally{globalThis.setTimeout=oldTimeout;h.close();}
});

test('screen card has no saved-calibration file controls',()=>{
 const h=setup(async()=>({status:true,val:null}));
 try {
  assert.equal(descendants(h.root,'input').filter(input=>input.type==='file').length,0);
  assert.equal(descendants(h.root,'select').length,0);
  assert.equal(find(h.root,'Salva risultati'),null);
  assert.equal(find(h.root,'Calibra 32 punti'),null);
  const labels=descendants(h.root,'button').map(button=>button.textContent);
  assert(!labels.includes('Carica calibrazione') && !labels.includes('Salva calibrazione'));
 } finally {h.close();}
});
