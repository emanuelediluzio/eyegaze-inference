import { Texts } from "../../core/texts.js";
/* Shared gaze UI helpers. All streams use the existing DataSocket/Worker. */
export function el(tag, className = "", text = "") {
  const node = document.createElement(tag); node.className = className; node.textContent = text; return node;
}
export function gazeStyles() {
  if (document.querySelector?.('link[data-gaze-ui]')) return;
  const link = document.createElement("link"); link.rel = "stylesheet";
  link.href = new URL("../../../css/app/gaze-panels.css", import.meta.url).href;
  link.setAttribute("data-gaze-ui", ""); document.head?.append(link);
}
export const statusText = status => ({ok:Texts.get("gaze.status.ok"), no_face:Texts.get("gaze.status.no_face"), uncalibrated:Texts.get("gaze.status.uncalibrated"), model_ready:Texts.get("gaze.status.model_ready"), model_unavailable:Texts.get("gaze.status.model_unavailable"), stale_input:Texts.get("gaze.status.stale_input"), input_disconnected:Texts.get("gaze.status.input_disconnected"), error:Texts.get("gaze.status.error")}[status] ?? Texts.get("gaze.waitingFace"));
let nextView = 0;
export class GazeStream {
  constructor(options, channel, receive, state) { this.options=options; this.channel=channel; this.receive=receive; this.state=state; this.id="gaze-ui-"+(++nextView); this.active=false; this.released=false; }
  attach() {
    if(this.active || this.released || !this.options.renderer) return;
    this.active=true;
    this.options.renderer.show(this.id,[{nodeName:this.options.node.settings.name,chanName:this.channel}],undefined,{kind:"text"},{
      onInfo: info => { if(this.released || info?.topic!=="text") return; try { const sample=JSON.parse(info.text); if(sample && typeof sample==="object") this.receive(sample); } catch { this.state?.(Texts.get("gaze.invalidData")); } },
      onState: (state,detail) => { if(state==="error") this.state?.(detail || Texts.get("gaze.streamUnavailable")); }
    });
  }
  setVisible(visible) { if(visible) this.attach(); else if(this.active) { this.options.renderer.hide(this.id); this.active=false; } }
  pause(paused) { this.options.renderer?.control(this.id,"pause",{paused}); }
  release() { this.released=true; if(this.active) this.options.renderer.hide(this.id); this.active=false; }
}
export function viewport() {
  return {width:window.innerWidth,height:window.innerHeight,x:window.screenX ?? 0,y:window.screenY ?? 0,dpr:window.devicePixelRatio ?? 1};
}
export function sameViewport(a,b) { return !!a && !!b && ["width","height","x","y","dpr"].every(k=>a[k]===b[k]); }
export function validPoint(sample, size) {
  const p=sample?.screen_xy_px;
  return sample?.schema==="screen-point-of-gaze.v1" && sample.status==="ok" && Array.isArray(p) && p.length===2 && p.every(Number.isFinite)
    && Array.isArray(sample.screen_size_px) && sample.screen_size_px[0]===size.width && sample.screen_size_px[1]===size.height
    && p[0]>=0 && p[0]<size.width && p[1]>=0 && p[1]<size.height;
}
