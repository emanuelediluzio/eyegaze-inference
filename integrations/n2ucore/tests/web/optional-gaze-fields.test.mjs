import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
const defs=JSON.parse(await readFile(new URL('../../N2uCore/res/dyndefs.json',import.meta.url)));
const ui=JSON.parse(await readFile(new URL('../../N2uCore/res/web/iride/schema-ui.json',import.meta.url)));
for(const type of ['N2uCore.dyn.eyegaze.GazeDirection','N2uCore.dyn.screenpointofgaze.ScreenPointOfGaze']) {
 test(`${type} exposes only its standard settings`,()=>{
  const expected=type.endsWith('.GazeDirection')?{device:{type:'ENUM',itemType:'STR',values:['auto','cpu'],default:'auto'},maxFps:{type:'INT',min:1,max:120,default:15}}:{};
  assert.deepEqual(defs[type].PROPS_SCHEMA,expected);
  assert.deepEqual(ui.nodes[type].PROPS_SCHEMA,type.endsWith('.GazeDirection')?{device:{label:'Compute device',visibility:'expert'},maxFps:{label:'Maximum inference FPS',visibility:'beginner'}}:{});
 });
}
test('screen node exposes guided calibration without import or clearing commands',()=>{
 const commands=defs['N2uCore.dyn.screenpointofgaze.ScreenPointOfGaze'].commands;
 for(const name of ['beginCalibration','calibrationTarget','finishTarget','fitCalibration','cancelCalibration']) assert(commands[name]);
 assert(!commands.setCalibration && !commands.clearCalibration);
});
