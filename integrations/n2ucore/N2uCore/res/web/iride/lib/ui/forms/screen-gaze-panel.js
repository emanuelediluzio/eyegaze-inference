import { Texts } from "../../core/texts.js";
import { el, gazeStyles, statusText, viewport, sameViewport, validPoint } from "./gaze-ui.js";

import { ScreenGazeSession } from "./screen-gaze-session.js";
import { calibrationTargets, validationTargets, GazeValidation } from "./gaze-calibration.js";

export class ScreenGazePanel {
  static channels = ["ScreenPoint"];
  static preferredWidth = 620;
  #options; #alive; #released = false; #busy = false; #paused = false;
  #message; #status; #details; #calibrateButton; #toggle; #buttons = [];
  #session; #unsubscribe; #ownsSession; #attached = false; #sample; #last = 0; #geometry = null; #invalid = false;
  #miniDot; #previewLabel; #calibration = null;
  #overlay; #cancelled = false; #cancelWait = null; #showPoint = false;
  #visibility; #resize; #validateButton; #results; #validation = null; #shownResults = null; #shownViewport = "";

  constructor(options) { this.#options = options; this.#alive = options.alive ?? true; }

  render(container) {
    gazeStyles();
    const root = el("section", "gaze-panel gaze-screen");
    const heading = el("header", "gaze-heading");
    this.#status = el("span", "gaze-status", Texts.get("gaze.waitingFace"));
    heading.append(el("strong", "", Texts.get("gaze.screenTitle")), this.#status);
    const preview = el("div", "gaze-preview");
    preview.setAttribute("aria-label", Texts.get("gaze.previewAria"));
    this.#previewLabel = el("span", "gaze-preview-label", Texts.get("gaze.calibratePreview"));
    this.#miniDot = el("span", "gaze-mini-dot"); this.#miniDot.hidden = true;
    preview.append(this.#previewLabel, this.#miniDot);
    this.#message = el("p", "gaze-message"); this.#message.setAttribute("aria-live", "polite");
    const hint = el("p", "gaze-hint", Texts.get("gaze.screenHint"));
    const actions = el("div", "gaze-actions");
    const add = (label, fn) => {
      const button = el("button", "command-run", label); button.type = "button";
      button.addEventListener("click", () => this.#run(fn)); this.#buttons.push(button); actions.append(button); return button;
    };
    this.#calibrateButton = add(Texts.get("gaze.calibrate", { count: 16 }), () => this.#calibrate());
    this.#validateButton = add(Texts.get("gaze.verify"), () => this.#validate());
    add(Texts.get("gaze.fullscreen"), async () => {
      if (!document.fullscreenElement) {
        if (!document.documentElement.requestFullscreen) throw new Error(Texts.get("gaze.fullscreenUnavailable"));
        await document.documentElement.requestFullscreen();
        await new Promise(resolve => requestAnimationFrame(resolve));
      }
      this.#checkGeometry();
      if (!this.#calibration || this.#invalid || !this.#dimensionsMatch()) {
        this.#message.textContent = Texts.get("gaze.fullscreenHint");
      } else { this.#session.setEnabled(true); this.#paint(); }
    });
    const toggleRow = el("label", "gaze-switches field");
    this.#toggle = el("input"); this.#toggle.type = "checkbox";
    this.#toggle.addEventListener("change", () => {
      this.#session.setEnabled(this.#toggle.checked);
      if (!this.#dimensionsMatch() || this.#invalid) this.#message.textContent = Texts.get("gaze.calibrateBeforePoint");
      this.#toggle.checked = this.#session.enabled;
      this.#paint();
    });
    toggleRow.append(this.#toggle, document.createTextNode(Texts.get("gaze.showPoint")));
    const technical = el("details", "gaze-details"); technical.append(el("summary", "", Texts.get("gaze.technical")));
    this.#details = el("pre"); technical.append(this.#details);
    this.#results = el("section", "gaze-validation-results");
    root.append(heading, preview, actions, toggleRow, hint, this.#message, this.#results, technical);
    container.replaceChildren(root);
    this.#ownsSession = !this.#options.sessionFor;
    this.#session = this.#options.sessionFor?.(this.#options.node.settings.name, () => new ScreenGazeSession(this.#options)) ?? new ScreenGazeSession(this.#options);
    this.#readSession(this.#session);
    this.#visibility = () => { if (document.hidden) { this.#cancel(); } this.#paint(); };
    this.#resize = () => { this.#checkGeometry(); this.#apply(); this.#paint(); };
    document.addEventListener("visibilitychange", this.#visibility);
    document.addEventListener("fullscreenchange", this.#resize);
    window.addEventListener("resize", this.#resize);
    this.#apply();
  }

  attach() {
    if (this.#attached || this.#released) return;
    this.#attached = true;
    this.#unsubscribe = this.#session.subscribe(session => this.#readSession(session));
    this.#session.ensureCalibration(() => this.#command("getCalibration")).catch(error => {
      if (!this.#released) this.#message.textContent = error.message;
    });
  }

  update(sample) { this.#session.update(sample); if (!this.#attached) this.#readSession(this.#session); }
  #readSession(session) {
    if (this.#released) return;
    this.#alive = session.alive; this.#sample = session.sample; this.#last = session.last; this.#calibration = session.calibration;
    this.#geometry = session.geometry; this.#invalid = session.invalid; this.#showPoint = session.enabled;
    this.#toggle.checked = session.enabled;
    if (this.#validation) this.#validation.observe(session.sample, performance.now());
    this.#showResults();
    if (this.#sample) { this.#status.textContent = statusText(this.#sample.status); this.#details.textContent = JSON.stringify(this.#sample, null, 2); }
    if (session.invalid) this.#message.textContent = Texts.get("gaze.geometryChanged");
    if (!session.alive || document.hidden) this.#cancel();
    this.#apply(); this.#paint();
  }

  #fresh() { return this.#session.fresh(); }
  #dimensionsMatch() { return this.#session.dimensionsMatch(); }
  #checkGeometry() { this.#session.checkGeometry(); this.#readSession(this.#session); }
  #paint() {
    const usable = !this.#released && !this.#paused && this.#alive && !document.hidden && this.#fresh() && !this.#busy && !this.#invalid && this.#dimensionsMatch() && validPoint(this.#sample, viewport());
    this.#miniDot.hidden = !usable;
    this.#previewLabel.hidden = usable;
    if (usable) {
      const [x, y] = this.#sample.screen_xy_px;
      this.#miniDot.style.left = `${100*x/this.#sample.screen_size_px[0]}%`;
      this.#miniDot.style.top = `${100*y/this.#sample.screen_size_px[1]}%`;

    } else this.#previewLabel.textContent = this.#invalid ? Texts.get("gaze.repeatCalibration") : this.#sample?.status === "no_face" ? Texts.get("gaze.faceCamera") : this.#calibration ? Texts.get("gaze.waitingGaze") : Texts.get("gaze.calibratePreview");
    if (this.#last && !this.#fresh() && !["no_face","error"].includes(this.#sample?.status)) this.#status.textContent = Texts.get("gaze.noRecentData");
  }
  #apply() {
    for (const button of this.#buttons) button.disabled = this.#busy || !this.#alive || this.#released;
    if (this.#calibrateButton) {
      this.#calibrateButton.disabled ||= !this.#fresh();
    }
    this.#validateButton.disabled ||= !this.#calibration || this.#invalid || !this.#dimensionsMatch() || !this.#fresh() || this.#sample?.status !== "ok";
    if (this.#toggle) this.#toggle.disabled = this.#busy || !this.#alive || this.#released || (!this.#showPoint && (!this.#calibration || this.#invalid));
  }
  setVisible(visible) { this.setPaused(!visible); }
  setAlive(alive) { this.#alive = alive; this.#session.setAlive(alive); if (!alive) { this.#cancel(); this.#last = 0; } this.#apply(); this.#paint(); }
  setPaused(paused) { this.#paused = paused;  if (paused) { this.#cancel(); this.#last = 0; } this.#apply(); this.#paint(); }
  release() {
    if (this.#released) return;
    this.#released = true; this.#cancel(); this.#unsubscribe?.();
    if (this.#ownsSession) this.#session.release();
    document.removeEventListener("visibilitychange", this.#visibility);
    document.removeEventListener("fullscreenchange", this.#resize); window.removeEventListener("resize", this.#resize); this.#apply();
  }
  #cancel() { this.#cancelled = true; this.#cancelWait?.(); this.#overlay?.remove(); }
  async #command(name, args = {}) {
    const response = await this.#options.send(this.#options.node, name, args);
    if (response?.status !== true) throw new Error(response?.err || Texts.get("gaze.commandFailed"));
    return response.val;
  }
  async #run(fn) {
    if (this.#busy || !this.#alive || this.#released) return;
    this.#busy = true; this.#session.setSuppressed(true); this.#apply(); this.#paint();
    try { await fn(); } catch (error) { this.#message.textContent = error.message; }
    finally { this.#busy = false; this.#session.setSuppressed(false); this.#apply(); this.#paint(); }
  }
  #wait(ms) {
    return new Promise(resolve => {
      const timer = setTimeout(() => { this.#cancelWait = null; resolve(); }, ms);
      this.#cancelWait = () => { clearTimeout(timer); this.#cancelWait = null; resolve(); };
    });
  }
  #showResults() {
    const currentViewport = JSON.stringify(viewport());
    if (this.#shownResults === this.#session.validationResults && this.#shownViewport === currentViewport) return;
    this.#shownResults = this.#session.validationResults; this.#shownViewport = currentViewport;
    const reports = this.#session.validationResults.filter(report => sameViewport(report.viewport, viewport()));
    this.#results.replaceChildren();
    if (!reports.length) return;
    this.#results.append(el("strong", "", Texts.get("gaze.resultsTitle")));
    const table = el("table"), head = el("tr");
    for (const label of [Texts.get("gaze.calibration"), Texts.get("gaze.meanError"), "RMSE", "P95"]) head.append(el("th", "", label));
    table.append(head);
    for (const report of [...reports].sort((a, b) => (a.calibration_points ?? 0) - (b.calibration_points ?? 0))) {
      const row = el("tr");
      for (const value of [report.calibration_points ? Texts.get("gaze.points", { count: report.calibration_points }) : Texts.get("gaze.imported"), ...[report.mean_error_px, report.rmse_px, report.p95_error_px].map(value => `${value.toFixed(1)} px`)]) row.append(el("td", "", value));
      table.append(row);
    }
    this.#results.append(table, el("p", "gaze-hint", Texts.get("gaze.resultsHint")));
  }
  async #validate() {
    if (!this.#calibration || this.#invalid || !this.#dimensionsMatch() || !this.#fresh() || this.#sample?.status !== "ok") throw new Error(Texts.get("gaze.prepareValidation"));
    this.#cancelled = false;
    const context = viewport(), revision = this.#session.calibrationRevision;
    const validation = new GazeValidation(context); this.#validation = validation;
    const overlay = el("div", "gaze-calibration"); this.#overlay = overlay;
    overlay.setAttribute("role", "dialog"); overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", Texts.get("gaze.validationAria")); overlay.tabIndex = -1;
    const dot = el("div", "gaze-target"), text = el("p"), cancel = el("button", "command-run", Texts.get("gaze.cancel"));
    cancel.type = "button"; cancel.onclick = () => this.#cancel();
    overlay.append(dot, text, cancel); document.body.append(overlay); overlay.focus();
    const check = () => {
      if (this.#cancelled || this.#released || this.#paused || !this.#alive || document.hidden) throw new Error(Texts.get("gaze.validationCancelled"));
      if (!sameViewport(context, viewport()) || revision !== this.#session.calibrationRevision) throw new Error(Texts.get("gaze.validationChanged"));
      if (!this.#fresh() || this.#sample?.status !== "ok") throw new Error(Texts.get("gaze.validationLostFace"));
    };
    const escape = event => { if (event.key === "Escape") { event.preventDefault(); this.#cancel(); } };
    document.addEventListener("keydown", escape);
    try {
      let count = 0;
      for (const target of validationTargets(context.width, context.height)) {
        check(); dot.style.left = `${target[0]}px`; dot.style.top = `${target[1]}px`;
        text.textContent = Texts.get("gaze.validationTarget", { count: ++count });
        await new Promise(resolve => requestAnimationFrame(resolve)); check();
        validation.beginTarget(target, performance.now(), this.#sample.source_ts_us);
        await this.#wait(3000); check(); validation.finishTarget();
      }
      const report = validation.result(this.#session.calibrationPoints);
      this.#session.recordValidation(report);
      this.#message.textContent = Texts.get("gaze.validationComplete", { mean: report.mean_error_px.toFixed(1), rmse: report.rmse_px.toFixed(1) });
    } finally {
      this.#validation = null; document.removeEventListener("keydown", escape); overlay.remove(); this.#overlay = null;
    }
  }
  async #calibrate() {
    if (!this.#fresh()) throw new Error(Texts.get("gaze.prepareCalibration"));
    this.#cancelled = false;
    const overlay = el("div", "gaze-calibration"); this.#overlay = overlay;
    overlay.setAttribute("role", "dialog"); overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", Texts.get("gaze.calibrationAria")); overlay.tabIndex = -1;
    const dot = el("div", "gaze-target"), text = el("p"), cancel = el("button", "command-run", Texts.get("gaze.cancel"));
    cancel.type = "button"; cancel.onclick = () => this.#cancel();
    overlay.append(dot, text, cancel); document.body.append(overlay); overlay.focus();
    const context = viewport(), width = context.width, height = context.height;
    const targets = calibrationTargets(width, height);
    const check = () => {
      if (this.#cancelled || !this.#alive || this.#released || this.#paused || document.hidden) throw new Error(Texts.get("gaze.calibrationCancelled"));
      if (!sameViewport(context, viewport())) throw new Error(Texts.get("gaze.viewportChanged"));
      if (!this.#fresh()) throw new Error(Texts.get("gaze.calibrationLostFace"));
    };
    const escape = e => { if (e.key === "Escape") { e.preventDefault(); this.#cancel(); } };
    document.addEventListener("keydown", escape);
    let started = false;
    try {
      await this.#command("beginCalibration", { screen_w: width, screen_h: height }); started = true;
      let count = 0;
      for (const [x, y] of targets) {
        check();
        dot.style.left = `${x}px`; dot.style.top = `${y}px`;
        text.textContent = Texts.get("gaze.calibrationTarget", { count: ++count, total: targets.length });
        await new Promise(resolve => requestAnimationFrame(resolve)); check();
        await this.#command("calibrationTarget", { x, y }); await this.#wait(3000); check();
        await this.#command("finishTarget");
      }
      check(); const result = await this.#command("fitCalibration");
      this.#calibration = await this.#command("getCalibration"); this.#session.setCalibration(this.#calibration, context);
      this.#session.setEnabled(true);
      this.#message.textContent = Texts.get("gaze.calibrationComplete", { residual: result.fit_residual_px.toFixed(1) });
      started = false;
    } finally {
      document.removeEventListener("keydown", escape); overlay.remove(); this.#overlay = null;
      if (started) { try { await this.#command("cancelCalibration"); } catch { /* Backend also cancels on input loss. */ } }
    }
  }
}
