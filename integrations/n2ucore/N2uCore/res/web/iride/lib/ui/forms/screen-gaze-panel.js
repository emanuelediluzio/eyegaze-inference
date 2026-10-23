/* Guided calibration lives in the user's browser; the backend receives only
 * targets and gaze samples over nodeCommand -> uinode -> Flow. No camera here.
 * Screen coordinates are viewport CSS pixels, matching the displayed targets.
 */
export class ScreenGazePanel {
  #options; #alive; #released = false; #busy = false; #message; #buttons = [];
  #overlay; #cancelled = false; #cancelWait = null;
  constructor(options) { this.#options = options; this.#alive = options.alive ?? true; }
  render(container) {
    const root = document.createElement("section"); root.className = "command-card";
    const title = document.createElement("strong"); title.textContent = "Screen Point of Gaze";
    const hint = document.createElement("p");
    hint.textContent = "Calibra 16 punti mantenendo ferma la testa. Le coordinate si riferiscono all’area di Iride usata per calibrare. Ripeti la calibrazione se cambi schermo, dimensioni della finestra o posizione della telecamera.";
    this.#message = document.createElement("p"); this.#message.setAttribute("aria-live", "polite");
    root.append(title, hint, this.#message);
    const add = (label, fn) => {
      const button = document.createElement("button"); button.type = "button";
      button.className = "command-run"; button.textContent = label;
      button.addEventListener("click", () => this.#run(fn)); this.#buttons.push(button); root.append(button);
    };
    add("Calibra 16 punti", () => this.#calibrate());
    const file = document.createElement("input"); file.type = "file"; file.accept = ".json,application/json"; file.hidden = true;
    file.addEventListener("change", () => {
      const selected = file.files?.[0]; file.value = "";
      if (selected) this.#run(async () => {
        const value = JSON.parse(await selected.text());
        await this.#command("setCalibration", { calibration: value });
        this.#message.textContent = "Calibrazione caricata.";
      });
    }); root.append(file);
    add("Carica JSON", async () => file.click());
    add("Salva JSON", async () => {
      const value = await this.#command("getCalibration");
      if (!value) throw new Error("Nessuna calibrazione da salvare.");
      const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }));
      const link = document.createElement("a"); link.href = url; link.download = "screen-calibration.json"; link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    add("Cancella calibrazione", async () => {
      await this.#command("clearCalibration"); this.#message.textContent = "Calibrazione cancellata.";
    });
    container.replaceChildren(root); this.#apply();
  }
  #apply() { for (const button of this.#buttons) button.disabled = this.#busy || !this.#alive || this.#released; }
  setAlive(alive) { this.#alive = alive; if (!alive) this.#cancel(); this.#apply(); }
  setPaused(paused) { if (paused) this.#cancel(); }
  release() { this.#released = true; this.#cancel(); this.#apply(); }
  #cancel() { this.#cancelled = true; this.#cancelWait?.(); this.#overlay?.remove(); }
  async #command(name, args = {}) {
    const response = await this.#options.send(this.#options.node, name, args);
    if (response?.status !== true) throw new Error(response?.err || "Comando non riuscito");
    return response.val;
  }
  async #run(fn) {
    if (this.#busy || !this.#alive || this.#released) return;
    this.#busy = true; this.#apply();
    try { await fn(); } catch (error) { this.#message.textContent = error.message; }
    finally { this.#busy = false; this.#apply(); }
  }
  #wait(ms) {
    return new Promise(resolve => {
      const timer = setTimeout(() => { this.#cancelWait = null; resolve(); }, ms);
      this.#cancelWait = () => { clearTimeout(timer); this.#cancelWait = null; resolve(); };
    });
  }
  async #calibrate() {
    this.#cancelled = false;
    const overlay = document.createElement("div"); this.#overlay = overlay;
    overlay.setAttribute("role", "dialog"); overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", "Calibrazione dello sguardo"); overlay.tabIndex = -1;
    Object.assign(overlay.style, { position: "fixed", inset: "0", zIndex: "2147483647", background: "#000", color: "#fff" });
    const dot = document.createElement("div");
    Object.assign(dot.style, { position: "absolute", width: "24px", height: "24px", borderRadius: "50%", background: "white", transform: "translate(-50%,-50%)" });
    const text = document.createElement("p"); Object.assign(text.style, { position: "absolute", bottom: "24px", width: "100%", textAlign: "center" });
    const cancel = document.createElement("button"); cancel.textContent = "Annulla (Esc)";
    Object.assign(cancel.style, { position: "absolute", top: "12px", right: "12px" });
    cancel.onclick = () => this.#cancel();
    overlay.append(dot, text, cancel); document.body.append(overlay); overlay.focus();
    const rect = overlay.getBoundingClientRect(); const width = Math.round(rect.width), height = Math.round(rect.height);
    const check = () => {
      if (this.#cancelled || !this.#alive || this.#released) throw new Error("Calibrazione annullata.");
      if (Math.round(overlay.getBoundingClientRect().width) !== width || Math.round(overlay.getBoundingClientRect().height) !== height)
        throw new Error("La finestra è cambiata: ripeti la calibrazione.");
    };
    const escape = e => { if (e.key === "Escape") { e.preventDefault(); this.#cancel(); } };
    const visibility = () => { if (document.hidden) this.#cancel(); };
    document.addEventListener("keydown", escape); document.addEventListener("visibilitychange", visibility);
    let started = false;
    try {
      await this.#command("beginCalibration", { screen_w: width, screen_h: height }); started = true;
      let count = 0;
      for (const ry of [.08, .37, .63, .92]) for (const rx of [.08, .37, .63, .92]) {
        check(); const x = Math.round(rx * width), y = Math.round(ry * height);
        dot.style.left = `${x}px`; dot.style.top = `${y}px`;
        text.textContent = `Guarda il punto ${++count}/16 · Tieni ferma la testa`;
        await new Promise(resolve => requestAnimationFrame(resolve)); check();
        await this.#command("calibrationTarget", { x, y });
        await this.#wait(3000); check();
        await this.#command("finishTarget");
      }
      check(); const result = await this.#command("fitCalibration");
      this.#message.textContent = `Calibrazione completata. Residuo del fit: ${result.fit_residual_px.toFixed(1)} px (non è un test su punti nuovi). Salva il JSON per riutilizzarla.`;
      started = false;
    } finally {
      document.removeEventListener("keydown", escape); document.removeEventListener("visibilitychange", visibility);
      this.#overlay?.remove(); this.#overlay = null;
      if (started) { try { await this.#command("cancelCalibration"); } catch { /* disconnection: backend cancels on input loss */ } }
    }
  }
}
