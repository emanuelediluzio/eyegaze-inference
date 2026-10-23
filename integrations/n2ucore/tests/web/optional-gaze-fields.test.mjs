import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { SchemaUi } from '../../N2uCore/res/web/iride/lib/model/schema-ui.js';
import { PipelineCheck } from '../../N2uCore/res/web/iride/lib/model/pipeline-check.js';
const defs=JSON.parse(await readFile(new URL('../../N2uCore/res/dyndefs.json',import.meta.url)));
const ui=JSON.parse(await readFile(new URL('../../N2uCore/res/web/iride/schema-ui.json',import.meta.url)));
function errors(type,field,value,optional=true) {
  const merged=structuredClone(defs);
  if(optional) SchemaUi.apply(merged, ui.nodes);
  const schema=merged[type].PROPS_SCHEMA;
  const settings=Object.fromEntries(Object.entries(schema).map(([k,v])=>[k,v.default]));
  settings.name='Test';settings[field]=value;
  const node={id:'test',type,settings};
  const model={addListener(){},schema:()=>({propsSchemaFor:()=>schema,inputsFor:()=>[]}),nodes:()=>[node],edges:()=>[]};
  const sysinits={addListener(){},sysinitOf:()=> 'SysInitGaze',isConnected:()=>true};
  return new PipelineCheck(model,sysinits).nodeProblems(node).fields;
}
test('Screen calibration JSON accepts empty value with actual UI schema',()=>{
 assert.deepEqual(errors('N2uCore.dyn.screenpointofgaze.ScreenPointOfGaze','calibrationPath',''),[]);
});
test('DINO source directory accepts empty value with actual UI schema',()=>{
 assert.deepEqual(errors('N2uCore.dyn.eyegaze.GazeDirection','hubDir',''),[]);
});
test('optional does not allow wrong value types',()=>{
 assert(errors('N2uCore.dyn.screenpointofgaze.ScreenPointOfGaze','calibrationPath',42).some(e=>e.path==='calibrationPath'));
});
test('unmarked text fields remain required',()=>{
 assert(errors('N2uCore.dyn.screenpointofgaze.ScreenPointOfGaze','calibrationPath','',false).some(e=>e.message==='required (empty field)'));
});
