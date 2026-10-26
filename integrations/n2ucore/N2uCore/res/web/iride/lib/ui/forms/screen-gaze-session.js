import { el, GazeStream, viewport, sameViewport, validPoint } from "./gaze-ui.js";

/** One screen-point view per node/run, independently of open cards. */
export class ScreenGazeSession {
  sample = null; last = 0; calibration = null; geometry = null;
  invalid = false; enabled = false; alive;
  #dot; #stream; #timer; #listeners = new Set(); #released = false;
  #suppressed = false; #loading = null; #initialized = false; #visibility; #resize;
  constructor(options) {
    this.alive = options.alive ?? true;
    this.#dot = el("div", "gaze-page-dot"); this.#dot.hidden = true;
    this.#dot.setAttribute("aria-hidden", "true"); document.body.append(this.#dot);
    this.#stream = new GazeStream(options, "ScreenPoint", sample => this.update(sample), () => {
      this.last = 0; this.#paint(); this.#notify();
    });
    this.#visibility = () => { if (document.hidden) this.last = 0; this.#paint(); this.#notify(); };
    this.#resize = () => { this.checkGeometry(); this.#paint(); this.#notify(); };
    document.addEventListener("visibilitychange", this.#visibility);
    document.addEventListener("fullscreenchange", this.#resize);
    window.addEventListener("resize", this.#resize);
    this.#timer = setInterval(() => { this.checkGeometry(); this.#paint(); this.#notify(); }, 300);
  }
  subscribe(listener) {
    if (this.#released) return () => {};
    this.#listeners.add(listener); this.#syncStream(); listener(this);
    return () => { this.#listeners.delete(listener); this.#syncStream(); };
  }
  async ensureCalibration(load) {
    if (this.#initialized || this.#released) return;
    if (!this.#loading) this.#loading = Promise.resolve().then(load).then(value => {
      if (!this.#released && !this.#initialized) this.setCalibration(value);
    }).finally(() => { this.#loading = null; });
    return this.#loading;
  }
  setCalibration(value, geometry = viewport()) {
    if (this.#released) return;
    this.#initialized = true; this.calibration = value; this.geometry = geometry;
    this.invalid = !!value && !this.dimensionsMatch(); this.#paint(); this.#notify();
  }
  setEnabled(enabled) {
    if (this.#released) return;
    if (enabled && (this.invalid || !this.dimensionsMatch())) return;
    this.enabled = enabled; this.#syncStream(); this.#paint(); this.#notify();
  }
  setAlive(alive) {
    this.alive = alive; if (!alive) this.last = 0;
    this.#paint(); this.#notify();
  }
  setSuppressed(value) { this.#suppressed = value; this.#paint(); }
  fresh() { return this.last > 0 && performance.now() - this.last < 1500 && ["ok", "uncalibrated"].includes(this.sample?.status); }
  dimensionsMatch() { const v = viewport(); return this.calibration?.screen_w === v.width && this.calibration?.screen_h === v.height; }
  checkGeometry() {
    if (this.geometry && !sameViewport(this.geometry, viewport())) this.invalid = true;
  }
  update(sample) {
    if (this.#released || !this.alive || sample?.schema !== "screen-point-of-gaze.v1") return;
    this.sample = sample; this.last = performance.now(); this.checkGeometry(); this.#paint(); this.#notify();
  }
  #syncStream() { this.#stream.setVisible(!this.#released && (this.enabled || this.#listeners.size > 0)); }
  #notify() { for (const listener of this.#listeners) listener(this); }
  #paint() {
    const usable = !this.#released && !this.#suppressed && this.alive && !document.hidden && this.fresh() && !this.invalid && this.dimensionsMatch() && validPoint(this.sample, viewport());
    this.#dot.hidden = !this.enabled || !usable;
    if (usable) {
      const [x, y] = this.sample.screen_xy_px;
      this.#dot.style.left = `${x}px`; this.#dot.style.top = `${y}px`;
    }
  }
  release() {
    if (this.#released) return;
    this.#released = true; this.enabled = false; clearInterval(this.#timer);
    this.#stream.release(); this.#listeners.clear(); this.#dot.remove();
    document.removeEventListener("visibilitychange", this.#visibility);
    document.removeEventListener("fullscreenchange", this.#resize);
    window.removeEventListener("resize", this.#resize);
  }
}
