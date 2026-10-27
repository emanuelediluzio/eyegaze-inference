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
  #visibility; #resize; #pointCount; #validateButton; #results; #saveResults; #validation = null; #shownResults = null; #shownViewport = "";

  constructor(options) { this.#options = options; this.#alive = options.alive ?? true; }

  render(container) {
    gazeStyles();
    const root = el("section", "gaze-panel gaze-screen");
    const heading = el("header", "gaze-heading");
    this.#status = el("span", "gaze-status", "In attesa del volto");
    heading.append(el("strong", "", "Punto di sguardo sullo schermo"), this.#status);
    const preview = el("div", "gaze-preview");
    preview.setAttribute("aria-label", "Anteprima del punto sullo schermo");
    this.#previewLabel = el("span", "gaze-preview-label", "Calibra per vedere dove guardi");
    this.#miniDot = el("span", "gaze-mini-dot"); this.#miniDot.hidden = true;
    preview.append(this.#previewLabel, this.#miniDot);
    this.#message = el("p", "gaze-message"); this.#message.setAttribute("aria-live", "polite");
    const hint = el("p", "gaze-hint", "Il punto resta attivo anche chiudendo questa card: togli il check per nasconderlo. Mantieni la stessa finestra, lo stesso schermo e la stessa posizione della webcam usati per calibrare.");
    const actions = el("div", "gaze-actions");
    const add = (label, fn) => {
      const button = el("button", "command-run", label); button.type = "button";
      button.addEventListener("click", () => this.#run(fn)); this.#buttons.push(button); actions.append(button); return button;
    };
    const countLabel = el("label", "gaze-count", "Punti di calibrazione ");
    this.#pointCount = el("select", "command-arg-input"); this.#pointCount.setAttribute("aria-label", "Punti di calibrazione");
    for (const count of [16, 32]) { const option = el("option", "", `${count} punti · circa ${count * 3} s`); option.value = String(count); this.#pointCount.append(option); }
    this.#pointCount.value = "16";
    this.#pointCount.addEventListener("change", () => {
      this.#session.targetCount = Number(this.#pointCount.value); this.#apply();
    });
    countLabel.append(this.#pointCount);
    this.#calibrateButton = add("Calibra 16 punti", () => this.#calibrate());
    this.#validateButton = add("Verifica precisione (9 punti)", () => this.#validate());
    add("Prova a schermo intero", async () => {
      if (!document.fullscreenElement) {
        if (!document.documentElement.requestFullscreen) throw new Error("Schermo intero non disponibile in questo browser.");
        await document.documentElement.requestFullscreen();
        await new Promise(resolve => requestAnimationFrame(resolve));
      }
      this.#checkGeometry();
      if (!this.#calibration || this.#invalid || !this.#dimensionsMatch()) {
        this.#message.textContent = "Calibra qui a schermo intero, poi attiva Mostra punto. Esc esce dallo schermo intero.";
      } else { this.#session.setEnabled(true); this.#paint(); }
    });
    const toggleRow = el("label", "gaze-switches field");
    this.#toggle = el("input"); this.#toggle.type = "checkbox";
    this.#toggle.addEventListener("change", () => {
      this.#session.setEnabled(this.#toggle.checked);
      if (!this.#dimensionsMatch() || this.#invalid) this.#message.textContent = "Calibra questa finestra prima di mostrare il punto.";
      this.#toggle.checked = this.#session.enabled;
      this.#paint();
    });
    toggleRow.append(this.#toggle, document.createTextNode("Mostra punto nella pagina"));
    const advanced = el("details", "gaze-details"); advanced.append(el("summary", "", "Opzioni avanzate"));
    const explanation = el("p", "gaze-hint", "Salva la calibrazione per riutilizzarla con la stessa postazione. Il file contiene coefficienti e dimensioni dell’area calibrata, non immagini o pesi del modello.");
    const file = el("input"); file.type = "file"; file.accept = ".json,application/json"; file.hidden = true;
    file.addEventListener("change", () => {
      const selected = file.files?.[0]; file.value = "";
      if (selected) this.#run(async () => {
        const value = JSON.parse(await selected.text());
        this.#calibration = await this.#command("setCalibration", { calibration: value });
        this.#session.setCalibration(this.#calibration);
        this.#message.textContent = this.#invalid ? "Dimensioni diverse: ricalibra questa finestra." : "Calibrazione caricata. Verifica il punto con la stessa postazione.";
        this.#apply();
      });
    });
    const importButton = el("button", "command-run", "Carica calibrazione"); importButton.type = "button";
    importButton.addEventListener("click", () => file.click()); this.#buttons.push(importButton);
    const exportButton = el("button", "command-run", "Salva calibrazione"); exportButton.type = "button";
    exportButton.addEventListener("click", () => this.#run(async () => {
      const value = await this.#command("getCalibration");
      if (!value) throw new Error("Nessuna calibrazione da salvare.");
      const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }));
      const link = el("a"); link.href = url; link.download = "screen-calibration.json"; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    })); this.#buttons.push(exportButton);
    const importActions = el("div", "gaze-actions"); importActions.append(importButton, exportButton, file);
    const technical = el("details", "gaze-details"); technical.append(el("summary", "", "Dettagli tecnici"));
    this.#details = el("pre"); technical.append(this.#details); advanced.append(explanation, importActions, technical);
    this.#results = el("section", "gaze-validation-results");
    this.#saveResults = el("button", "command-run", "Salva risultati"); this.#saveResults.type = "button";
    this.#buttons.push(this.#saveResults);
    this.#saveResults.addEventListener("click", () => {
      const reports = this.#session.validationResults;
      const url = URL.createObjectURL(new Blob([JSON.stringify(reports, null, 2)], { type: "application/json" }));
      const link = el("a"); link.href = url; link.download = "gaze-validation.json"; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    root.append(heading, preview, countLabel, actions, toggleRow, hint, this.#message, this.#results, advanced);
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
    this.#pointCount.value = String(session.targetCount);
    if (this.#validation) this.#validation.observe(session.sample, performance.now());
    this.#showResults();
    if (this.#sample) { this.#status.textContent = statusText(this.#sample.status); this.#details.textContent = JSON.stringify(this.#sample, null, 2); }
    if (session.invalid) this.#message.textContent = "Finestra o schermo cambiati: ripeti la calibrazione.";
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

    } else this.#previewLabel.textContent = this.#invalid ? "Ripeti la calibrazione per questa finestra" : this.#sample?.status === "no_face" ? "Posizionati davanti alla webcam" : this.#calibration ? "In attesa dello sguardo" : "Calibra per vedere dove guardi";
    if (this.#last && !this.#fresh() && !["no_face","error"].includes(this.#sample?.status)) this.#status.textContent = "Nessun dato recente";
  }
  #apply() {
    for (const button of this.#buttons) button.disabled = this.#busy || !this.#alive || this.#released;
    if (this.#calibrateButton) {
      this.#calibrateButton.disabled ||= !this.#fresh();
      this.#calibrateButton.textContent = `Calibra ${this.#pointCount.value} punti`;
    }
    this.#pointCount.disabled = this.#busy || !this.#alive || this.#released;
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
    if (response?.status !== true) throw new Error(response?.err || "Comando non riuscito");
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
    this.#results.append(el("strong", "", "Verifica su 9 punti nuovi · pixel CSS"));
    const table = el("table"), head = el("tr");
    for (const label of ["Calibrazione", "Errore medio", "RMSE", "P95"]) head.append(el("th", "", label));
    table.append(head);
    for (const report of [...reports].sort((a, b) => (a.calibration_points ?? 0) - (b.calibration_points ?? 0))) {
      const row = el("tr");
      for (const value of [report.calibration_points ? `${report.calibration_points} punti` : "Importata", ...[report.mean_error_px, report.rmse_px, report.p95_error_px].map(value => `${value.toFixed(1)} px`)]) row.append(el("td", "", value));
      table.append(row);
    }
    this.#results.append(table, el("p", "gaze-hint", "Ultima verifica per modalità, nella stessa finestra. Valori più bassi sono migliori. Mantieni invariati testa, webcam e postazione; ripeti per confermare."), this.#saveResults);
  }
  async #validate() {
    if (!this.#calibration || this.#invalid || !this.#dimensionsMatch() || !this.#fresh() || this.#sample?.status !== "ok") throw new Error("Calibra questa finestra e posizionati davanti alla webcam.");
    this.#cancelled = false;
    const context = viewport(), revision = this.#session.calibrationRevision;
    const validation = new GazeValidation(context); this.#validation = validation;
    const overlay = el("div", "gaze-calibration"); this.#overlay = overlay;
    overlay.setAttribute("role", "dialog"); overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", "Verifica indipendente dello sguardo"); overlay.tabIndex = -1;
    const dot = el("div", "gaze-target"), text = el("p"), cancel = el("button", "command-run", "Annulla (Esc)");
    cancel.type = "button"; cancel.onclick = () => this.#cancel();
    overlay.append(dot, text, cancel); document.body.append(overlay); overlay.focus();
    const check = () => {
      if (this.#cancelled || this.#released || this.#paused || !this.#alive || document.hidden) throw new Error("Verifica annullata. La calibrazione resta invariata.");
      if (!sameViewport(context, viewport()) || revision !== this.#session.calibrationRevision) throw new Error("Finestra o calibrazione cambiate: ripeti la verifica.");
      if (!this.#fresh() || this.#sample?.status !== "ok") throw new Error("Volto perso o dati non recenti: ripeti la verifica.");
    };
    const escape = event => { if (event.key === "Escape") { event.preventDefault(); this.#cancel(); } };
    document.addEventListener("keydown", escape);
    try {
      let count = 0;
      for (const target of validationTargets(context.width, context.height)) {
        check(); dot.style.left = `${target[0]}px`; dot.style.top = `${target[1]}px`;
        text.textContent = `Verifica ${++count}/9 · Guarda il punto e tieni ferma la testa`;
        await new Promise(resolve => requestAnimationFrame(resolve)); check();
        validation.beginTarget(target, performance.now(), this.#sample.source_ts_us);
        await this.#wait(3000); check(); validation.finishTarget();
      }
      const report = validation.result(this.#session.calibrationPoints);
      this.#session.recordValidation(report);
      this.#message.textContent = `Verifica completata: errore medio ${report.mean_error_px.toFixed(1)} px, RMSE ${report.rmse_px.toFixed(1)} px. La calibrazione non è stata modificata.`;
    } finally {
      this.#validation = null; document.removeEventListener("keydown", escape); overlay.remove(); this.#overlay = null;
    }
  }
  async #calibrate() {
    if (!this.#fresh()) throw new Error("Posizionati davanti alla webcam prima di calibrare.");
    const targetCount = this.#session.targetCount;
    this.#cancelled = false;
    const overlay = el("div", "gaze-calibration"); this.#overlay = overlay;
    overlay.setAttribute("role", "dialog"); overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", "Calibrazione dello sguardo"); overlay.tabIndex = -1;
    const dot = el("div", "gaze-target"), text = el("p"), cancel = el("button", "command-run", "Annulla (Esc)");
    cancel.type = "button"; cancel.onclick = () => this.#cancel();
    overlay.append(dot, text, cancel); document.body.append(overlay); overlay.focus();
    const context = viewport(), width = context.width, height = context.height;
    const check = () => {
      if (this.#cancelled || !this.#alive || this.#released || this.#paused || document.hidden) throw new Error("Calibrazione annullata. La precedente resta disponibile.");
      if (!sameViewport(context, viewport())) throw new Error("La finestra è cambiata: ripeti la calibrazione.");
      if (!this.#fresh()) throw new Error("Volto perso o dati non recenti: riposizionati e ripeti la calibrazione.");
    };
    const escape = e => { if (e.key === "Escape") { e.preventDefault(); this.#cancel(); } };
    document.addEventListener("keydown", escape);
    let started = false;
    try {
      await this.#command("beginCalibration", { screen_w: width, screen_h: height }); started = true;
      let count = 0;
      for (const [x, y] of calibrationTargets(targetCount, width, height)) {
        check();
        dot.style.left = `${x}px`; dot.style.top = `${y}px`;
        text.textContent = `Guarda il punto ${++count}/${targetCount} · Tieni ferma la testa`;
        await new Promise(resolve => requestAnimationFrame(resolve)); check();
        await this.#command("calibrationTarget", { x, y }); await this.#wait(3000); check();
        await this.#command("finishTarget");
      }
      check(); const result = await this.#command("fitCalibration");
      this.#calibration = await this.#command("getCalibration"); this.#session.setCalibration(this.#calibration, context, targetCount);
      this.#session.setEnabled(true);
      this.#message.textContent = `Calibrazione completata. Punto attivo nella pagina. Residuo del fit: ${result.fit_residual_px.toFixed(1)} px. Avvia Verifica precisione per misurare su punti nuovi.`;
      started = false;
    } finally {
      document.removeEventListener("keydown", escape); overlay.remove(); this.#overlay = null;
      if (started) { try { await this.#command("cancelCalibration"); } catch { /* Backend also cancels on input loss. */ } }
    }
  }
}
