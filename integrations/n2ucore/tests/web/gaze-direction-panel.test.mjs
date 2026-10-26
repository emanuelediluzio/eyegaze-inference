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
