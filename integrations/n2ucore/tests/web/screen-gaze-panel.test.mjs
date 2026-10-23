// Minimal DOM harness: tests command routing and cancellation without a browser/camera.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
const source = await readFile(new URL('../../N2uCore/res/web/iride/lib/ui/forms/screen-gaze-panel.js', import.meta.url), 'utf8');
const { ScreenGazePanel } = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.style = {}; this.events = {}; this.isConnected = true; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  setAttribute() {}
  addEventListener(name, fn) { this.events[name] = fn; }
  remove() { this.isConnected = false; }
  focus() {}
  getBoundingClientRect() { return { width: 1000, height: 600 }; }
  click() { this.events.click?.(); this.onclick?.(); }
}
function setup(send) {
  globalThis.document = { createElement: tag => new Element(tag), body: new Element('body'), addEventListener() {}, removeEventListener() {}, hidden: false };
  globalThis.requestAnimationFrame = fn => { fn(); };
  const timers = globalThis.setTimeout;
  globalThis.setTimeout = (fn, ms) => timers(fn, ms === 3000 ? 0 : ms);
  const panel = new ScreenGazePanel({ node: 'screen-node', send, alive: true });
  const root = new Element('root'); panel.render(root);
  return { panel, root, restore: () => { globalThis.setTimeout = timers; panel.release(); } };
}
async function until(predicate) {
  for (let i = 0; i < 300; i++) {
    if (predicate()) return;
    await new Promise(resolve => setTimeout(resolve, 2));
  }
  assert.fail('Panel did not complete');
}
test('guided calibration routes all 16 targets and fits through supplied uinode sender', async () => {
  const calls=[];
  const h=setup(async (node,cmd,args) => { calls.push({node,cmd,args}); return {status:true,val: cmd==='fitCalibration'?{fit_residual_px:2}:null}; });
  try {
    const button=h.root.children[0].children.find(x=>x.textContent==='Calibra 16 punti'); button.click();
    await until(()=>calls.some(x=>x.cmd==='fitCalibration'));
    assert(calls.every(x=>x.node==='screen-node'));
    const targets=calls.filter(x=>x.cmd==='calibrationTarget');
    assert.equal(targets.length,16); assert.equal(calls.filter(x=>x.cmd==='finishTarget').length,16);
    assert.deepEqual(calls[0].args,{screen_w:1000,screen_h:600});
    assert.deepEqual(targets[0].args,{x:80,y:48}); assert.deepEqual(targets.at(-1).args,{x:920,y:552});
    assert.equal(calls.filter(x=>x.cmd==='cancelCalibration').length,0);
  } finally { h.restore(); }
});
test('closing the panel during collection cancels and does not fit', async () => {
  const calls=[]; let h;
  h=setup(async (_node,cmd) => { calls.push(cmd); if(cmd==='calibrationTarget')h.panel.release(); return {status:true,val:null}; });
  try {
    h.root.children[0].children.find(x=>x.textContent==='Calibra 16 punti').click();
    await until(()=>calls.includes('cancelCalibration'));
    assert(!calls.includes('fitCalibration')); assert(!calls.includes('finishTarget'));
  } finally { h.restore(); }
});
test('missing face samples abort collection and show the backend error', async () => {
  const calls=[];
  const h=setup(async (_node,cmd) => { calls.push(cmd); return cmd==='finishTarget'?{status:false,err:'At least 3 fresh predictions required'}:{status:true,val:null}; });
  try {
    h.root.children[0].children.find(x=>x.textContent==='Calibra 16 punti').click();
    await until(()=>calls.includes('cancelCalibration'));
    await new Promise(resolve=>setImmediate(resolve));
    assert(!calls.includes('fitCalibration'));
    assert(h.root.children[0].children.some(x=>x.textContent?.includes('3 fresh predictions')));
  } finally { h.restore(); }
});
