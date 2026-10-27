import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
const defs=JSON.parse(await readFile(new URL('../../N2uCore/res/dyndefs.json',import.meta.url)));
const ui=JSON.parse(await readFile(new URL('../../N2uCore/res/web/iride/schema-ui.json',import.meta.url)));
for(const type of ['N2uCore.dyn.eyegaze.GazeDirection','N2uCore.dyn.screenpointofgaze.ScreenPointOfGaze']) {
 test(`${type} exposes no model or saved-calibration settings`,()=>{
  assert.deepEqual(defs[type].PROPS_SCHEMA,{});
  assert.deepEqual(ui.nodes[type].PROPS_SCHEMA,{});
 });
}
test('screen node exposes guided calibration without import or clearing commands',()=>{
 const commands=defs['N2uCore.dyn.screenpointofgaze.ScreenPointOfGaze'].commands;
 for(const name of ['beginCalibration','calibrationTarget','finishTarget','fitCalibration','cancelCalibration']) assert(commands[name]);
 assert(!commands.setCalibration && !commands.clearCalibration);
});
