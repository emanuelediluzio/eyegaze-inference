import {test} from 'node:test';
import assert from 'node:assert/strict';
import {calibrationTargets,validationTargets,GazeValidation} from '../../N2uCore/res/web/iride/lib/ui/forms/gaze-calibration.js';
const size={width:1000,height:600,x:0,y:0,dpr:1};
const sample=(xy,ts)=>({schema:'screen-point-of-gaze.v1',status:'ok',screen_xy_px:xy,screen_size_px:[1000,600],source_ts_us:ts});
test('16 and 32 grids cover the same area and validation uses nine distinct new targets',()=>{
 for(const count of [16,32]){
  const points=calibrationTargets(count,1000,600);assert.equal(points.length,count);assert.equal(new Set(points.map(String)).size,count);
  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);assert.equal(Math.min(...xs),80);assert.equal(Math.max(...xs),920);assert.equal(Math.min(...ys),48);assert.equal(Math.max(...ys),552);
  for(const point of validationTargets(1000,600))assert(!points.some(p=>String(p)===String(point)));
 }
 assert.throws(()=>calibrationTargets(20,1000,600));assert.equal(validationTargets(1000,600).length,9);
});
test('held-out errors use unique fresh post-settling samples and calculate equal-target mean and RMSE',()=>{
 const validation=new GazeValidation(size);let ts=1000000;
 for(const [x,y] of validationTargets(1000,600)){
  validation.beginTarget([x,y],0,ts);
  validation.observe(sample([x+100,y],ts+100),100); // settling
  validation.observe(sample([x+100,y],ts+500000),1200); // captured before settling
  for(let k=0;k<3;k++){
   const value=sample([x+3,y+4],ts+1200000+k*100000);validation.observe(value,1200+k*100);validation.observe(value,1400+k*100);
  }
  validation.observe({...sample([NaN,y],ts+1500000)},1500);
  validation.finishTarget();ts+=4000000;
 }
 const report=validation.result(32);assert.equal(report.samples,27);assert.equal(report.mean_error_px,5);assert.equal(report.rmse_px,5);assert.equal(report.p95_error_px,5);assert.equal(report.calibration_points,32);
});
test('missing or invalid observations cannot produce an accuracy result',()=>{
 const validation=new GazeValidation(size);assert.throws(()=>validation.beginTarget([500,300],0,undefined));validation.beginTarget([500,300],0,1000000);
 validation.observe({...sample([500,300],2200000),status:'no_face'},1200);validation.observe({...sample([500,300],2300000),screen_size_px:[800,600]},1300);
 assert.throws(()=>validation.finishTarget());assert.throws(()=>validation.result(16));
});
