import { el, gazeStyles, GazeStream, statusText, viewport, sameViewport, validPoint } from "./gaze-ui.js";

/** Calibration commands via uinode; visual samples via the shared DataSocket. */
export class ScreenGazePanel {
  static channels = ["ScreenPoint"];
  static preferredWidth = 620;
  #options; #alive; #released = false; #busy = false; #paused = false;
  #message; #status; #details; #calibrateButton; #toggle; #buttons = [];
  #stream; #timer; #sample; #last = 0; #geometry = null; #invalid = false;
  #pageDot; #miniDot; #previewLabel; #calibration = null;
  #overlay; #cancelled = false; #cancelWait = null; #showPoint = false;
  #visibility; #resize;

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
    const hint = el("p", "gaze-hint", "Il punto segue lo sguardo nella pagina Iride. Mantieni la stessa finestra, lo stesso schermo e la stessa posizione della webcam usati per calibrare.");
    const actions = el("div", "gaze-actions");
    const add = (label, fn, primary = false) => {
      const button = el("button", primary ? "gaze-primary" : "", label); button.type = "button";
      button.addEventListener("click", () => this.#run(fn)); this.#buttons.push(button); actions.append(button); return button;
    };
    this.#calibrateButton = add("Calibra 16 punti", () => this.#calibrate(), true);
    add("Prova a schermo intero", async () => {
      if (!document.fullscreenElement) {
        if (!document.documentElement.requestFullscreen) throw new Error("Schermo intero non disponibile in questo browser.");
        await document.documentElement.requestFullscreen();
        await new Promise(resolve => requestAnimationFrame(resolve));
      }
      this.#checkGeometry();
      if (!this.#calibration || this.#invalid || !this.#dimensionsMatch()) {
        this.#message.textContent = "Calibra qui a schermo intero, poi attiva Mostra punto. Esc esce dallo schermo intero.";
      } else { this.#toggle.checked = true; this.#showPoint = true; this.#paint(); }
    });
    const toggleRow = el("label", "gaze-switches");
    this.#toggle = el("input"); this.#toggle.type = "checkbox";
    this.#toggle.addEventListener("change", () => {
      this.#showPoint = this.#toggle.checked;
      if (!this.#dimensionsMatch() || this.#invalid) {
        this.#showPoint = false; this.#toggle.checked = false;
        this.#message.textContent = "Calibra questa finestra prima di mostrare il punto.";
      }
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
        this.#geometry = viewport(); this.#invalid = !this.#dimensionsMatch();
        this.#message.textContent = this.#invalid ? "Dimensioni diverse: ricalibra questa finestra." : "Calibrazione caricata. Verifica il punto con la stessa postazione.";
        this.#apply();
      });
    });
    const importButton = el("button", "", "Carica calibrazione"); importButton.type = "button";
    importButton.addEventListener("click", () => file.click()); this.#buttons.push(importButton);
    const exportButton = el("button", "", "Salva calibrazione"); exportButton.type = "button";
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
    root.append(heading, preview, actions, toggleRow, hint, this.#message, advanced);
    container.replaceChildren(root);
    this.#pageDot = el("div", "gaze-page-dot"); this.#pageDot.hidden = true;
    this.#pageDot.setAttribute("aria-hidden", "true"); document.body.append(this.#pageDot);
    this.#stream = new GazeStream(this.#options, "ScreenPoint", sample => this.update(sample), text => {
      this.#status.textContent = text; this.#last = 0; this.#paint();
    });
    this.#visibility = () => { if (document.hidden) { this.#cancel(); this.#last = 0; } this.#paint(); };
    this.#resize = () => { this.#checkGeometry(); this.#apply(); this.#paint(); };
    document.addEventListener("visibilitychange", this.#visibility);
    document.addEventListener("fullscreenchange", this.#resize);
    window.addEventListener("resize", this.#resize);
    this.#apply();
  }

  attach() {
    if (this.#timer || this.#released) return;
    this.#stream.attach();
    this.#timer = setInterval(() => { this.#checkGeometry(); this.#apply(); this.#paint(); }, 300);
    this.#command("getCalibration").then(value => {
      if (this.#released) return;
      this.#calibration = value; this.#geometry = viewport(); this.#invalid = !!value && !this.#dimensionsMatch(); this.#apply();
    }).catch(error => { if (!this.#released) this.#message.textContent = error.message; });
  }

  update(sample) {
    if (this.#released || this.#paused || !this.#alive || sample.schema !== "screen-point-of-gaze.v1") return;
    this.#sample = sample; this.#last = performance.now();
    this.#status.textContent = statusText(sample.status);
    this.#details.textContent = JSON.stringify(sample, null, 2);
    this.#apply(); this.#paint();
  }

  #fresh() { return this.#last > 0 && performance.now() - this.#last < 1500 && ["ok", "uncalibrated"].includes(this.#sample?.status); }
  #dimensionsMatch() { const v = viewport(); return this.#calibration?.screen_w === v.width && this.#calibration?.screen_h === v.height; }
  #checkGeometry() {
    if (this.#geometry && !sameViewport(this.#geometry, viewport())) {
      this.#invalid = true; this.#showPoint = false; this.#toggle.checked = false;
      this.#message.textContent = "Finestra o schermo cambiati: ripeti la calibrazione.";
    }
  }
  #paint() {
    const usable = !this.#released && !this.#paused && this.#alive && !document.hidden && this.#fresh() && !this.#busy && !this.#invalid && this.#dimensionsMatch() && validPoint(this.#sample, viewport());
    this.#miniDot.hidden = !usable; this.#pageDot.hidden = !usable || !this.#showPoint;
    this.#previewLabel.hidden = usable;
    if (usable) {
      const [x, y] = this.#sample.screen_xy_px;
      this.#miniDot.style.left = `${100*x/this.#sample.screen_size_px[0]}%`;
      this.#miniDot.style.top = `${100*y/this.#sample.screen_size_px[1]}%`;
      this.#pageDot.style.left = `${x}px`; this.#pageDot.style.top = `${y}px`;
    } else this.#previewLabel.textContent = this.#invalid ? "Ripeti la calibrazione per questa finestra" : this.#sample?.status === "no_face" ? "Posizionati davanti alla webcam" : this.#calibration ? "In attesa dello sguardo" : "Calibra per vedere dove guardi";
    if (this.#last && !this.#fresh() && !["no_face","error"].includes(this.#sample?.status)) this.#status.textContent = "Nessun dato recente";
  }
  #apply() {
    for (const button of this.#buttons) button.disabled = this.#busy || !this.#alive || this.#released;
    if (this.#calibrateButton) this.#calibrateButton.disabled ||= !this.#fresh();
    if (this.#toggle) this.#toggle.disabled = this.#busy || !this.#alive || this.#released || !this.#calibration || this.#invalid;
  }
  setVisible(visible) { this.#stream?.setVisible(visible); this.setPaused(!visible); }
  setAlive(alive) { this.#alive = alive; if (!alive) { this.#cancel(); this.#last = 0; } this.#apply(); this.#paint(); }
  setPaused(paused) { this.#paused = paused; this.#stream?.pause(paused); if (paused) { this.#cancel(); this.#last = 0; } this.#apply(); this.#paint(); }
  release() {
    if (this.#released) return;
    this.#released = true; this.#cancel(); clearInterval(this.#timer); this.#stream?.release();
    this.#pageDot?.remove(); document.removeEventListener("visibilitychange", this.#visibility);
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
    this.#busy = true; this.#apply(); this.#paint();
    try { await fn(); } catch (error) { this.#message.textContent = error.message; }
    finally { this.#busy = false; this.#apply(); this.#paint(); }
  }
  #wait(ms) {
    return new Promise(resolve => {
      const timer = setTimeout(() => { this.#cancelWait = null; resolve(); }, ms);
      this.#cancelWait = () => { clearTimeout(timer); this.#cancelWait = null; resolve(); };
    });
  }
  async #calibrate() {
    if (!this.#fresh()) throw new Error("Posizionati davanti alla webcam prima di calibrare.");
    this.#cancelled = false;
    const overlay = el("div", "gaze-calibration"); this.#overlay = overlay;
    overlay.setAttribute("role", "dialog"); overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", "Calibrazione dello sguardo"); overlay.tabIndex = -1;
    const dot = el("div", "gaze-target"), text = el("p"), cancel = el("button", "", "Annulla (Esc)");
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
      for (const ry of [.08, .37, .63, .92]) for (const rx of [.08, .37, .63, .92]) {
        check(); const x = Math.round(rx * width), y = Math.round(ry * height);
        dot.style.left = `${x}px`; dot.style.top = `${y}px`;
        text.textContent = `Guarda il punto ${++count}/16 · Tieni ferma la testa`;
        await new Promise(resolve => requestAnimationFrame(resolve)); check();
        await this.#command("calibrationTarget", { x, y }); await this.#wait(3000); check();
        await this.#command("finishTarget");
      }
      check(); const result = await this.#command("fitCalibration");
      this.#calibration = await this.#command("getCalibration"); this.#geometry = context; this.#invalid = false;
      this.#showPoint = true; this.#toggle.checked = true;
      this.#message.textContent = `Calibrazione completata. Punto attivo nella pagina. Residuo del fit: ${result.fit_residual_px.toFixed(1)} px; verifica il puntamento su punti nuovi.`;
      started = false;
    } finally {
      document.removeEventListener("keydown", escape); overlay.remove(); this.#overlay = null;
      if (started) { try { await this.#command("cancelCalibration"); } catch { /* Backend also cancels on input loss. */ } }
    }
  }
}
