import { test } from "node:test";
import assert from "node:assert/strict";
import { GazeDecoder } from "../../N2uCore/res/web/iride/lib/render/decoders/gaze-decoder.js";

const decode = sample => GazeDecoder.decode(new TextEncoder().encode(JSON.stringify(sample)).buffer);

test("Tobii scene gaze and pupil measurements keep their existing interpretation", () => {
  const result = decode({ data: { gaze2d: [.2, .8], eyeleft: { pupildiameter: 3 }, eyeright: { pupildiameter: 4 } } });
  assert.deepEqual(result.gaze2d, [.2, .8]);
  assert.deepEqual(result.frame.values, [3, 4]);
  assert.deepEqual(result.frame.names, ["Left Pupil (mm)", "Right Pupil (mm)"]);
});

test("direction GAZE samples expose angles without fabricating Tobii pupil or scene data", () => {
  const result = decode({ schema: "gaze-direction.v1", status: "ok", yaw_deg: 12, pitch_deg: -5, roll_deg: 3 });
  assert.deepEqual(result.frame.values, [12, -5, 3]);
  assert.deepEqual(result.frame.names, ["Gaze yaw (deg)", "Gaze pitch (deg)", "Head roll (deg)"]);
  assert.equal(result.gaze2d, null);
  const absent = decode({ schema: "gaze-direction.v1", status: "no_face", yaw_deg: 12 });
  assert(absent.frame.values.every(Number.isNaN));
});

test("legacy screen recordings without gaze2d retain pixel charts", () => {
  const result = decode({ schema: "screen-point-of-gaze.v1", status: "ok", screen_xy_px: [640, 480] });
  assert.deepEqual(result.frame.values, [640, 480]);
  assert.deepEqual(result.frame.names, ["Screen X (px)", "Screen Y (px)"]);
  assert.equal(result.gaze2d, null);
  assert(decode({ schema: "screen-point-of-gaze.v1", status: "uncalibrated" }).frame.values.every(Number.isNaN));
  assert.equal(GazeDecoder.decode(new TextEncoder().encode("invalid").buffer), null);
});

test("GAZE frames reach card sessions through the existing worker and shared subscription", async t => {
  const messages = [];
  const sent = [];
  let socket;
  const previousSelf = globalThis.self;
  globalThis.self = { postMessage: message => messages.push(message) };
  t.mock.method(globalThis, "setInterval", () => 1);
  t.mock.method(globalThis, "setTimeout", () => 2);
  const { ReconnectingSocket } = await import("../../N2uCore/res/web/iride/lib/core/reconnecting-socket.js");
  t.mock.method(ReconnectingSocket.prototype, "start", function () { socket = this; this.emitEvent("status", "open"); });
  t.mock.method(ReconnectingSocket.prototype, "send", data => { sent.push(JSON.parse(data)); return true; });
  t.mock.method(ReconnectingSocket.prototype, "stop", () => {});
  try {
    await import("../../N2uCore/res/web/iride/lib/render/data-worker.js");
    const command = data => self.onmessage({ data });
    command({ type: "connect", url: "ws://test/appWS" });
    socket.emitEvent("message", JSON.stringify({ id: sent.at(-1).id, status: true }));
    command({ type: "nodeAlive", nodeName: "Gaze", alive: true });
    for (const viewId of ["card", "persistent-point"]) {
      command({ type: "show", viewId, channels: [{ nodeName: "Gaze", chanName: "GazeDirection" }], options: { kind: "text" } });
    }
    assert.equal(sent.filter(item => item.cmd === "subscribeNodeChannel").length, 1);
    socket.emitEvent("message", JSON.stringify({ id: sent.at(-1).id, status: true, val: { props: {
      streamID: "1", type: "GAZE", structure: { rows: 1, columns: 1, dataType: "JSON_ANON_T" }
    } } }));
    const sample = { schema: "gaze-direction.v1", status: "ok", yaw_deg: 10, pitch_deg: 5, roll_deg: 2 };
    const payload = new TextEncoder().encode(JSON.stringify(sample));
    const frame = new ArrayBuffer(32 + payload.length);
    new DataView(frame).setBigUint64(0, 1n, true);
    new DataView(frame).setBigUint64(8, 1000000n, true);
    new Uint8Array(frame, 32).set(payload);
    socket.emitEvent("message", frame);
    for (const viewId of ["card", "persistent-point"]) {
      const message = messages.find(item => item.viewId === viewId && item.info?.topic === "text");
      assert.deepEqual(JSON.parse(message.info.text), sample);
    }
    command({ type: "hide", viewId: "card" });
    assert.equal(sent.filter(item => item.cmd === "unsubscribeNodeChannel").length, 0);
    const before = messages.length;
    socket.emitEvent("message", frame);
    assert(messages.slice(before).some(item => item.viewId === "persistent-point" && item.info?.topic === "text"));
    assert(!messages.slice(before).some(item => item.viewId === "card"));
    command({ type: "hide", viewId: "persistent-point" });
    assert.equal(sent.filter(item => item.cmd === "unsubscribeNodeChannel").length, 1);
    command({ type: "close" });
  } finally {
    globalThis.self = previousSelf;
  }
});


test("screen point uses the existing normalized GAZE decoder with missing pupils", () => {
  const result = decode({schema: "screen-point-of-gaze.v1", status: "ok",
    gaze2d: [.52, 290 / 600], gaze2d_coordinate_space: "viewport_normalized",
    eyeleft: {pupildiameter: null}, eyeright: {pupildiameter: null},
    screen_xy_px: [520, 290], screen_size_px: [1000, 600]});
  assert.deepEqual(result.gaze2d, [.52, 290 / 600]);
  assert.deepEqual(result.frame.names, GazeDecoder.COLUMN_NAMES);
  assert(result.frame.values.every(Number.isNaN));
  const missing = decode({schema: "screen-point-of-gaze.v1", status: "no_face", gaze2d: null,
    eyeleft: {pupildiameter: null}, eyeright: {pupildiameter: null}});
  assert.equal(missing.gaze2d, null);
  assert(missing.frame.values.every(Number.isNaN));
});

test("viewport gaze is not assigned to a camera image through angular dependencies", async () => {
  const {OverlayResolver} = await import("../../N2uCore/res/web/iride/app/run/overlay-resolver.js");
  const types = {"Camera.VideoStream": "VIDEO_VISIBLE", "Direction.GazeDirection": "GAZE"};
  const overlays = new OverlayResolver(() => {}, label => types[label]);
  overlays.addChannel("Screen", "ScreenPoint", {type: "GAZE", dependsOn: ["Direction.GazeDirection"]});
  assert.deepEqual(overlays.overlaysFor("Camera.VideoStream", ["Screen.ScreenPoint"]), []);
  overlays.addChannel("Tobii", "GazeData", {type: "GAZE", dependsOn: ["Camera.VideoStream"]});
  assert.equal(overlays.overlaysFor("Camera.VideoStream", ["Tobii.GazeData"]).length, 1);
});
