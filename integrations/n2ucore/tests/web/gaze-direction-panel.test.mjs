import {test} from 'node:test';
import assert from 'node:assert/strict';
import {GazeDirectionPanel} from '../../N2uCore/res/web/iride/lib/ui/forms/gaze-direction-panel.js';
import {Element,environment,descendants,direction} from './gaze-dom.mjs';
test('direction card shares DataSocket video and gaze streams; no webcam acquisition in browser',()=>{
 const env=environment();const panel=new GazeDirectionPanel({node:{settings:{name:'Gaze'}},renderer:env.renderer,inputSources:{VideoStream:{nodeName:'WebCam',chanName:'VideoStream'}},alive:true});const root=new Element('root');
 try{panel.render(root);assert.equal(env.views.size,0);panel.attach();assert.equal(env.views.size,2);
 assert([...env.views.values()].some(v=>v.channels[0].nodeName==='WebCam'&&v.canvas));
 const stream=[...env.views.values()].find(v=>v.options.kind==='text');stream.handlers.onInfo({topic:'text',text:JSON.stringify(direction)});
 assert.equal(descendants(root,'path').length,2);assert.equal(descendants(root,'circle').length,2);assert.equal(descendants(root,'rect').length,1);
 panel.update({...direction,status:'no_face'});assert.equal(descendants(root,'path').length,0);
 panel.release();assert.equal(env.views.size,0);
 }finally{panel.release();env.restore();}
});
test('pausing and disconnecting remove stale overlays and preserve shared subscription lifecycle',()=>{
 const env=environment();const panel=new GazeDirectionPanel({node:{settings:{name:'Gaze'}},renderer:env.renderer,inputSources:{VideoStream:{nodeName:'Cam',chanName:'VideoStream'}}});const root=new Element('root');
 try{panel.render(root);panel.attach();panel.update(direction);assert.equal(descendants(root,'path').length,2);panel.setPaused(true);assert.equal(descendants(root,'path').length,0);panel.update(direction);assert.equal(descendants(root,'path').length,0);
 panel.setPaused(false);panel.update(direction);panel.setAlive(false);assert.equal(descendants(root,'path').length,0);
 }finally{panel.release();env.restore();}
});
test('live updates reuse metric elements and format technical JSON only when expanded',()=>{
 const env=environment();const panel=new GazeDirectionPanel({node:{settings:{name:'Gaze'}},renderer:env.renderer});const root=new Element('root');
 try{panel.render(root);const values=descendants(root,'dd');const details=descendants(root,'details')[0];const log=descendants(root,'pre')[0];
 panel.update(direction);panel.update({...direction,yaw_deg:20});
 assert.equal(descendants(root,'dd')[0],values[0]);assert.equal(values[0].textContent,'20.0°');assert.equal(log.textContent,'');
 details.open=true;details.events.toggle();assert.equal(JSON.parse(log.textContent).yaw_deg,20);
 }finally{panel.release();env.restore();}
});

test('direction card labels landmark head roll separately from model angles',()=>{
 const env=environment();const panel=new GazeDirectionPanel({node:{settings:{name:'Gaze'}},renderer:env.renderer});const root=new Element('root');
 try {panel.render(root);panel.update({...direction,roll_deg:12.5,roll_source:'head_landmarks'});
 assert(descendants(root,'dt').some(el=>el.textContent==='Inclinazione testa · roll'));
 assert.equal(descendants(root,'dd')[2].textContent,'12.5°');
 panel.update({...direction,status:'no_face',roll_deg:null});assert.equal(descendants(root,'dd')[2].textContent,'—');
 }finally{panel.release();env.restore();}
});

test('gaze panels use the framework language tables and shared form controls', async () => {
 const {Texts}=await import('../../N2uCore/res/web/iride/lib/core/texts.js');
 const {ScreenGazePanel}=await import('../../N2uCore/res/web/iride/lib/ui/forms/screen-gaze-panel.js');
 const env=environment();Texts.setLanguage('en');
 const directionPanel=new GazeDirectionPanel({node:{settings:{name:'Gaze'}},renderer:env.renderer});
 const screenPanel=new ScreenGazePanel({node:{settings:{name:'Screen'}},renderer:env.renderer,send:async()=>({status:true,val:null})});
 const directionRoot=new Element('root'),screenRoot=new Element('root');
 try {
  directionPanel.render(directionRoot);screenPanel.render(screenRoot);
  assert(descendants(directionRoot,'strong').some(node=>node.textContent==='Gaze direction'));
  assert(descendants(directionRoot,'dt').some(node=>node.textContent==='Head tilt · roll'));
  assert(descendants(screenRoot,'button').some(node=>node.textContent==='Calibrate 16 points'));
  assert(descendants(screenRoot,'button').every(node=>node.className==='command-run'));
  assert.equal(descendants(screenRoot,'select').length,0);
  assert(descendants(directionRoot,'input').every(node=>node.parentElement.className==='field'));
  assert.equal(Texts.get('gaze.calibrationTarget',{count:3,total:16}),'Look at point 3/16 · Keep your head still');
 } finally {directionPanel.release();screenPanel.release();env.restore();}
});
