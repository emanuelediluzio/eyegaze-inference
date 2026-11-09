/* iride — app/run/panels/screen-gaze-panel.js
 *
 * ScreenGazePanel: il pannello della scheda di un nodo ScreenPointOfGaze (dyn/screenpointofgaze.py), al posto dei
 * mattoncini (NodePanels). Mostra e comanda la sessione del nodo (ScreenGazeSession, che vive con la pipeline):
 *   - intestazione con lo stato del campione;
 *   - anteprima: un riquadro con le proporzioni della pagina e il punto guardato (o cosa manca: calibrare,
 *     guardare la webcam, aspettare lo sguardo, ripetere la calibrazione);
 *   - qualità della stima in tempo reale (punteggio 0-100 e avviso di distanza o stabilità, da GazeDirection) e
 *     avviso se la testa si è spostata dalla posizione della calibrazione (calibration_drift del nodo), con la
 *     guida per tornarci (volto della calibrazione tratteggiato, volto di adesso pieno, verde se allineati);
 *   - Calibra 16 punti: un riquadro a tutta pagina mostra uno alla volta i 16 punti (GazeTargets.calibration); per
 *     ognuno calibrationTarget, TARGET_MS di attesa con i secondi che restano accanto al punto, finishTarget (se fallisce il punto si rifà, al più TARGET_TRIES volte); alla fine fitCalibration e getCalibration, e il
 *     pallino sulla pagina si accende. Si annulla con Esc o Annulla, e da solo se la pagina cambia misura, il volto
 *     si perde o i dati non sono più recenti (cancelCalibration: resta la calibrazione di prima);
 *   - Verifica precisione (9 punti): come la calibrazione ma senza comandi al nodo (GazeValidation): errore medio,
 *     RMSE e P95 in una tabella; la calibrazione non cambia;
 *   - Prova a schermo intero; Mostra punto nella pagina (il pallino della sessione, che resta anche a scheda chiusa).
 * updated (2026-11-05): tolti i dettagli tecnici (l'ultimo campione JSON a scomparsa), decisione dell'utente.
 * updated (2026-11-06): nel lettore (replay) il nodo non gira, i dati li manda il Player: niente comandi (né la
 * calibrazione all'apertura, né calibrazione e verifica, che restano spente); il canale ha il nome dato dal Player
 * (channelNames). Restano anteprima e stato dei campioni registrati.
 * Da ScreenGazePanel di eyegaze-inference (integrations/n2ucore, del collega). Differenze: classi di iride
 * (PanelDom, GazeTargets, GazeValidation, BrowserViewport, GazeStatus), testi rinnovati al cambio di lingua.
 * Dipende da ScreenGazeSession (screen-gaze-session.js), GazeTargets (gaze-targets.js), GazeValidation
 * (gaze-validation.js), BrowserViewport (browser-viewport.js), GazeQuality (gaze-quality.js), GazeStatus
 * (gaze-status.js), PanelDom (panel-dom.js),
 * Texts (lib/core/texts.js).
 */

import { Texts } from "../../../lib/core/texts.js";
import { BrowserViewport } from "./browser-viewport.js";
import { GazeQuality } from "./gaze-quality.js";
import { GazeStatus } from "./gaze-status.js";
import { GazeTargets } from "./gaze-targets.js";
import { GazeValidation } from "./gaze-validation.js";
import { PanelDom } from "./panel-dom.js";
import { ScreenGazeSession } from "./screen-gaze-session.js";

/**
 * Pannello di ScreenPointOfGaze: anteprima, calibrazione, verifica e punto sulla pagina.
 */

export class ScreenGazePanel {
  static CHANNELS = ["ScreenPoint"];     // canali del nodo che il pannello mostra (Card: filmstrip, setVisible)
  static PREFERRED_WIDTH = 620;          // larghezza della scheda che il pannello chiede
  static TARGET_TRIES = 3;               // updated (2026-11-06): volte che un punto si rifà se il nodo non ha abbastanza stime
  static TARGET_MS = 2000;              // tempo per ogni punto di calibrazione e di verifica (updated 2026-11-05: era 3000; misurato su una registrazione, 2 s danno quasi lo stesso errore)

  #options = null;
  #session = null;
  #ownsSession = false;
  #unsubscribe = null;
  #attached = false;
  #released = false;
  #paused = false;
  #busy = false;
  #cancelled = false;
  #cancelWait = null;
  #overlay = null;
  #validation = null;
  #shownResults = null;
  #shownViewport = "";
  #root = null;
  #status = null;
  #previewLabel = null;
  #miniDot = null;
  #quality = null;       // qualità della stima in tempo reale (dalla direzione): punteggio e avviso
  #qualityHint = null;
  #driftHint = null;      // avviso se la testa si è spostata dalla posizione della calibrazione
  #guide = null;          // guida di posizione: volto della calibrazione e volto di adesso
  #message = null;
  #results = null;
  #toggle = null;
  #calibrateButton = null;
  #validateButton = null;
  #buttons = [];
  #onVisibility = null;
  #onResize = null;

  /**
   * @param {{nodeName: string, renderer: DataSocket, alive?: boolean,
   *   send: function(string, object): Promise<{status: boolean, val?: *, err?: string}>,
   *   sessionFor?: function(string, function(): object): object, channelNames?: Object<string, string>,
   *   replay?: boolean}} options - nome del nodo, Worker dei dati, se il nodo è vivo, invio dei comandi al nodo, sessioni
   *   dei pannelli (PanelSessions) e, updated (2026-11-06), nome vero dei canali e se la scheda è del lettore
   */

  constructor(options) {
    this.#options = options;
  }

  /**
   * Crea il pannello nel contenitore (la casella della scheda) e lo lega alla sessione del nodo.
   * @param {HTMLElement} container - dove metterlo
   */

  render(container) {
    this.#root = PanelDom.element("section", "gaze-panel gaze-screen");
    const heading = PanelDom.element("header", "gaze-heading");
    this.#status = PanelDom.translated("span", "gaze-status", "gaze.waitingFace");
    heading.append(PanelDom.translated("strong", "", "gaze.screenTitle"), this.#status);
    const preview = PanelDom.element("div", "gaze-preview");
    preview.setAttribute("aria-label", Texts.get("gaze.previewAria"));
    this.#previewLabel = PanelDom.translated("span", "gaze-preview-label", "gaze.calibratePreview");
    this.#miniDot = PanelDom.element("span", "gaze-mini-dot");
    this.#miniDot.hidden = true;
    preview.append(this.#previewLabel, this.#miniDot);
    const quality = PanelDom.element("p", "gaze-quality");
    this.#quality = PanelDom.element("span", "gaze-quality-score", "—");
    this.#qualityHint = PanelDom.element("span", "gaze-quality-hint");
    this.#qualityHint.setAttribute("aria-live", "polite");
    this.#driftHint = PanelDom.element("span", "gaze-quality-hint");
    this.#driftHint.setAttribute("aria-live", "polite");
    quality.append(PanelDom.translated("span", "", "gaze.quality"), this.#quality, this.#qualityHint, this.#driftHint);
    this.#guide = GazeQuality.guide();
    const actions = PanelDom.element("div", "gaze-actions");
    this.#calibrateButton = this.#addButton(actions, "gaze.calibrate", { count: ScreenGazeSession.CALIBRATION_POINTS }, () => this.#calibrate());
    this.#validateButton = this.#addButton(actions, "gaze.verify", {}, () => this.#validate());
    this.#addButton(actions, "gaze.fullscreen", {}, () => this.#fullscreen());
    const toggle = PanelDom.checkbox("gaze.showPoint", false, (checked) => this.#onToggle(checked));
    toggle.row.classList.add("gaze-switches");
    this.#toggle = toggle.input;
    this.#message = PanelDom.element("p", "gaze-message");
    this.#message.setAttribute("aria-live", "polite");
    this.#results = PanelDom.element("section", "gaze-validation-results");
    this.#root.append(heading, preview, quality, this.#guide, actions, toggle.row, PanelDom.translated("p", "gaze-hint", "gaze.screenHint"),
      this.#message, this.#results);
    container.replaceChildren(this.#root);
    const chanName = this.#options.channelNames?.[ScreenGazePanel.CHANNELS[0]];   // updated (2026-11-06)
    const create = () => new ScreenGazeSession({ renderer: this.#options.renderer, nodeName: this.#options.nodeName, chanName,
      alive: this.#options.alive });
    this.#ownsSession = !this.#options.sessionFor;
    // updated (2026-11-06): nel lettore la chiave è nodo + canale (tutti i nodi registrati arrivano dal Player)
    const key = this.#options.replay ? this.#options.nodeName + "/" + (chanName ?? ScreenGazeSession.CHANNEL) : this.#options.nodeName;
    this.#session = this.#options.sessionFor?.(key, create, this.#options.nodeName) ?? create();
    this.#onVisibility = () => { if (document.hidden) this.#cancel(); this.#paint(); };
    this.#onResize = () => { this.#session.checkGeometry(); this.#read(this.#session); };
    document.addEventListener("visibilitychange", this.#onVisibility);
    document.addEventListener("fullscreenchange", this.#onResize);
    window.addEventListener("resize", this.#onResize);
    this.#read(this.#session);
  }

  /**
   * Con la scheda nella pagina: ascolta la sessione e, la prima volta, chiede la calibrazione al nodo.
   */

  attach() {
    if (this.#attached || this.#released) return;
    this.#attached = true;
    this.#unsubscribe = this.#session.subscribe((session) => this.#read(session));
    if (this.#options.replay) return;   // updated (2026-11-06): nel lettore nessun nodo a cui chiederla
    this.#session.ensureCalibration(() => this.#command("getCalibration"))?.catch((error) => {
      if (!this.#released) this.#say(error.message);
    });
  }

  /**
   * Il canale del pannello mostrato o nascosto dalla filmstrip.
   * @param {boolean} visible - true = mostrato
   */

  setVisible(visible) {
    this.setPaused(!visible);
  }

  /**
   * Scheda ridotta, parcheggiata o nascosta: annulla calibrazione o verifica in corso.
   * @param {boolean} paused - true = in pausa
   */

  setPaused(paused) {
    this.#paused = paused;
    if (paused) this.#cancel();
    this.#apply();
    this.#paint();
  }

  /**
   * @param {boolean} alive - true se il nodo è vivo
   */

  setAlive(alive) {
    this.#session.setAlive(alive);
    if (!alive) this.#cancel();
    this.#apply();
    this.#paint();
  }

  /**
   * La lingua è cambiata: riscrive i testi del pannello.
   */

  refreshTexts() {
    if (!this.#root) return;
    PanelDom.refreshTexts(this.#root);
    this.#root.querySelector(".gaze-preview")?.setAttribute("aria-label", Texts.get("gaze.previewAria"));
    this.#shownResults = null;
    this.#showResults();
  }

  /**
   * La scheda sta per sparire: annulla quello che è in corso e smette di ascoltare la sessione (che resta, se è
   * della pipeline).
   */

  release() {
    if (this.#released) return;
    this.#released = true;
    this.#cancel();
    this.#unsubscribe?.();
    if (this.#ownsSession) this.#session.release();
    document.removeEventListener("visibilitychange", this.#onVisibility);
    document.removeEventListener("fullscreenchange", this.#onResize);
    window.removeEventListener("resize", this.#onResize);
  }

  /**
   * @param {HTMLElement} actions - la riga dei bottoni
   * @param {string} key - chiave di Texts della scritta
   * @param {object} params - valori dei segnaposto
   * @param {function(): Promise<void>} action - cosa fa (con #run: un'azione alla volta)
   * @returns {HTMLButtonElement} il bottone
   */

  #addButton(actions, key, params, action) {
    const button = PanelDom.button(key, params, () => this.#run(action));
    this.#buttons.push(button);
    actions.append(button);
    return button;
  }

  /**
   * Lo stato della sessione è cambiato: lo riporta nel pannello.
   * @param {ScreenGazeSession} session - la sessione
   */

  #read(session) {
    if (this.#released) return;
    this.#toggle.checked = session.enabled;
    this.#validation?.observe(session.sample, session.sampleTs, performance.now());
    this.#showResults();
    if (session.sample) PanelDom.setText(this.#status, GazeStatus.keyOf(session.sample.status));
    if (session.invalid) PanelDom.setText(this.#message, "gaze.geometryChanged");
    if (!session.alive || document.hidden) this.#cancel();
    this.#apply();
    this.#paint();
  }

  /**
   * L'interruttore del pallino sulla pagina.
   * @param {boolean} checked - spuntato
   */

  #onToggle(checked) {
    this.#session.setEnabled(checked);
    if (checked && !this.#session.enabled) PanelDom.setText(this.#message, "gaze.calibrateBeforePoint");
    this.#toggle.checked = this.#session.enabled;
    this.#paint();
  }

  /**
   * Bottoni e interruttore accesi o spenti secondo lo stato.
   */

  #apply() {
    const session = this.#session;
    const blocked = this.#busy || !session.alive || this.#released || this.#options.replay === true;   // updated (2026-11-06): replay
    for (const button of this.#buttons) button.disabled = blocked;
    this.#calibrateButton.disabled ||= !session.fresh();
    this.#validateButton.disabled ||= !session.calibration || session.invalid || !session.dimensionsMatch()
      || !session.fresh() || session.sample?.status !== "ok";
    this.#toggle.disabled = blocked || (!session.enabled && (!session.calibration || session.invalid));
  }

  /**
   * L'anteprima: il punto guardato in proporzione, o cosa manca per vederlo.
   */

  #paint() {
    const session = this.#session;
    const usable = !this.#released && !this.#paused && session.alive && !document.hidden && session.fresh() && !this.#busy
      && !session.invalid && session.dimensionsMatch() && ScreenGazeSession.validPoint(session.sample, BrowserViewport.current());
    this.#miniDot.hidden = !usable;
    this.#previewLabel.hidden = usable;
    if (usable) {
      const [x, y] = session.sample.screen_xy_px;
      this.#miniDot.style.left = `${100 * x / session.sample.screen_size_px[0]}%`;
      this.#miniDot.style.top = `${100 * y / session.sample.screen_size_px[1]}%`;
    } else {
      PanelDom.setText(this.#previewLabel, session.invalid ? "gaze.repeatCalibration"
        : session.sample?.status === "no_face" ? "gaze.faceCamera"
          : session.calibration ? "gaze.waitingGaze" : "gaze.calibratePreview");
    }
    const quality = session.fresh() ? session.sample?.quality ?? null : null;
    const score = GazeQuality.scoreText(quality);
    if (this.#quality.textContent !== score) this.#quality.textContent = score;
    GazeQuality.showHint(this.#qualityHint, quality);
    const drift = session.fresh() ? session.sample?.calibration_drift ?? null : null;
    GazeQuality.showDrift(this.#driftHint, drift);
    GazeQuality.showGuide(this.#guide, drift, session.sample?.frame_aspect ?? null);
    if (session.last && !session.fresh() && !["no_face", "error"].includes(session.sample?.status)) {
      PanelDom.setText(this.#status, "gaze.noRecentData");
    }
  }

  /**
   * L'ultima verifica della precisione, se è per la pagina di adesso: tabella con errore medio, RMSE e P95.
   */

  #showResults() {
    const viewport = BrowserViewport.current();
    const key = JSON.stringify(viewport);
    if (this.#shownResults === this.#session.validationResults && this.#shownViewport === key) return;
    this.#shownResults = this.#session.validationResults;
    this.#shownViewport = key;
    const reports = this.#session.validationResults.filter(report => BrowserViewport.same(report.viewport, viewport));
    this.#results.replaceChildren();
    if (!reports.length) return;
    const table = PanelDom.element("table");
    const head = PanelDom.element("tr");
    head.append(PanelDom.translated("th", "", "gaze.calibration"), PanelDom.translated("th", "", "gaze.meanError"),
      PanelDom.element("th", "", "RMSE"), PanelDom.element("th", "", "P95"));
    table.append(head);
    for (const report of reports) {
      const row = PanelDom.element("tr");
      row.append(report.calibration_points
        ? PanelDom.translated("td", "", "gaze.points", { count: report.calibration_points })
        : PanelDom.translated("td", "", "gaze.imported"));
      for (const value of [report.mean_error_px, report.rmse_px, report.p95_error_px]) row.append(PanelDom.element("td", "", `${value.toFixed(1)} px`));
      table.append(row);
    }
    this.#results.append(PanelDom.translated("strong", "", "gaze.resultsTitle"), table,
      PanelDom.translated("p", "gaze-hint", "gaze.resultsHint"));
  }

  /**
   * Scrive nel messaggio un testo già pronto (un errore, un risultato coi numeri), che al cambio di lingua resta com'è.
   * @param {string} text - il testo
   */

  #say(text) {
    delete this.#message.dataset.text;
    this.#message.textContent = text;
  }

  /**
   * Un comando al nodo.
   * @param {string} name - il comando
   * @param {object} [args={}] - i suoi argomenti
   * @returns {Promise<*>} il valore della risposta; un rifiuto diventa un errore col messaggio del nodo
   */

  async #command(name, args = {}) {
    const response = await this.#options.send(name, args);
    if (response?.status !== true) throw new Error(response?.err || Texts.get("gaze.commandFailed"));
    return response.val;
  }

  /**
   * Un'azione dei bottoni, una alla volta: intanto il pallino sulla pagina è nascosto; un errore va nel messaggio.
   * @param {function(): Promise<void>} action - l'azione
   */

  async #run(action) {
    if (this.#busy || !this.#session.alive || this.#released) return;
    this.#busy = true;
    this.#session.setSuppressed(true);
    this.#apply();
    this.#paint();
    try {
      await action();
    } catch (error) {
      this.#say(error.message);
    } finally {
      this.#busy = false;
      this.#session.setSuppressed(false);
      this.#apply();
      this.#paint();
    }
  }

  /**
   * Annulla calibrazione o verifica in corso (Esc, Annulla, scheda nascosta, nodo morto, pagina nascosta).
   */

  #cancel() {
    this.#cancelled = true;
    this.#cancelWait?.();
    this.#overlay?.remove();
  }

  /**
   * @param {number} ms - quanto aspettare
   * @returns {Promise<void>} finisce dopo ms, o subito se si annulla
   */

  #wait(ms) {
    return new Promise((resolve) => {
      const timer = setTimeout(() => { this.#cancelWait = null; resolve(); }, ms);
      this.#cancelWait = () => { clearTimeout(timer); this.#cancelWait = null; resolve(); };
    });
  }

  /**
   * Tiene il mirino per TARGET_MS: l'anello si stringe, accanto i secondi che restano (2, 1); arancione finché lo
   * sguardo arriva (GazeValidation.SETTLE_MS, il nodo non raccoglie), poi verde. Si interrompe se annullato.
   * @param {HTMLElement} dot - il mirino, con dentro anello e contatore dei secondi
   * @returns {Promise<void>} finisce allo scadere o all'annullamento
   */

  async #hold(dot) {
    const count = dot.querySelector(".gaze-target-count");
    dot.classList.remove("running");
    void dot.offsetWidth;   // fa ripartire l'animazione dell'anello da capo per questo punto
    dot.style.setProperty("--gaze-target-ms", ScreenGazePanel.TARGET_MS + "ms");
    dot.classList.add("running");
    dot.dataset.phase = "settle";
    const start = performance.now();
    for (let left = Math.ceil(ScreenGazePanel.TARGET_MS / 1000); left > 0 && !this.#cancelled; left--) {
      count.textContent = String(left);
      await this.#wait(Math.min(1000, ScreenGazePanel.TARGET_MS - 1000 * (left - 1)));
      if (performance.now() - start >= GazeValidation.SETTLE_MS) dot.dataset.phase = "hold";
    }
    count.textContent = "";
    dot.classList.remove("running");
  }

  /**
   * Il riquadro a tutta pagina con il punto da guardare, la scritta e Annulla.
   * @param {string} ariaKey - chiave di Texts dell'etichetta del riquadro
   * @returns {{overlay: HTMLElement, dot: HTMLElement, text: HTMLElement}} il riquadro, il punto e la scritta
   */

  #openOverlay(ariaKey) {
    const overlay = PanelDom.element("div", "gaze-calibration");
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.setAttribute("aria-label", Texts.get(ariaKey));
    overlay.tabIndex = -1;
    const dot = PanelDom.element("div", "gaze-target");   // il mirino: anello, croce, punto da fissare, secondi
    dot.append(PanelDom.element("span", "gaze-target-ring"), PanelDom.element("span", "gaze-target-cross"),
      PanelDom.element("span", "gaze-target-center"), PanelDom.element("span", "gaze-target-count"));
    const text = PanelDom.element("p");
    const cancel = PanelDom.button("gaze.cancel", {}, () => this.#cancel());
    overlay.append(dot, text, cancel);
    document.body.append(overlay);
    overlay.focus();
    this.#overlay = overlay;
    return { overlay, dot, text };
  }

  /**
   * Mostra un punto e aspetta che il browser lo disegni.
   * @param {HTMLElement} dot - il punto
   * @param {Array<number>} target - dove, [x, y] in pixel CSS
   * @returns {Promise<void>} finisce al prossimo disegno della pagina
   */

  static #showTarget(dot, target) {
    dot.style.left = `${target[0]}px`;
    dot.style.top = `${target[1]}px`;
    return new Promise(resolve => requestAnimationFrame(resolve));
  }

  /**
   * @returns {function(KeyboardEvent): void} un ascoltatore dei tasti che con Esc annulla
   */

  #escape() {
    return (event) => { if (event.key === "Escape") { event.preventDefault(); this.#cancel(); } };
  }

  /**
   * Calibra 16 punti (vedi in testa al file).
   */

  async #calibrate() {
    if (!this.#session.fresh()) throw new Error(Texts.get("gaze.prepareCalibration"));
    this.#cancelled = false;
    const context = BrowserViewport.current();
    const targets = GazeTargets.calibration(context.width, context.height);
    const { overlay, dot, text } = this.#openOverlay("gaze.calibrationAria");
    const check = () => {
      if (this.#cancelled || !this.#session.alive || this.#released || this.#paused || document.hidden) throw new Error(Texts.get("gaze.calibrationCancelled"));
      if (!BrowserViewport.same(context, BrowserViewport.current())) throw new Error(Texts.get("gaze.viewportChanged"));
      if (!this.#session.fresh()) throw new Error(Texts.get("gaze.calibrationLostFace"));
    };
    const escape = this.#escape();
    document.addEventListener("keydown", escape);
    let started = false;
    try {
      await this.#command("beginCalibration", { screen_w: context.width, screen_h: context.height });
      started = true;
      let count = 0;
      for (const target of targets) {
        check();
        text.textContent = Texts.get("gaze.calibrationTarget", { count: ++count, total: targets.length });
        await ScreenGazePanel.#showTarget(dot, target);
        for (let tries = 1; ; tries++) {
          check();
          await this.#command("calibrationTarget", { x: target[0], y: target[1] });
          await this.#hold(dot);
          check();
          try {
            await this.#command("finishTarget");
            break;
          } catch (error) {   // updated (2026-11-06): troppe poche stime sul punto (es. un battito): si rifà, come chiede il nodo
            if (tries >= ScreenGazePanel.TARGET_TRIES) throw error;
          }
        }
      }
      check();
      const result = await this.#command("fitCalibration");
      this.#session.setCalibration(await this.#command("getCalibration"), context);
      this.#session.setEnabled(true);
      PanelDom.setText(this.#message, "gaze.calibrationComplete", { error: result.cv_error_px.toFixed(0), degree: result.degree });
      started = false;
    } finally {
      document.removeEventListener("keydown", escape);
      overlay.remove();
      this.#overlay = null;
      if (started) await this.#command("cancelCalibration").catch(() => {});   // il nodo annulla anche da solo se perde l'ingresso
    }
  }

  /**
   * Verifica la precisione su 9 punti nuovi (vedi in testa al file).
   */

  async #validate() {
    const session = this.#session;
    if (!session.calibration || session.invalid || !session.dimensionsMatch() || !session.fresh() || session.sample?.status !== "ok") {
      throw new Error(Texts.get("gaze.prepareValidation"));
    }
    this.#cancelled = false;
    const context = BrowserViewport.current();
    const revision = session.calibrationRevision;
    this.#validation = new GazeValidation(context);
    const { overlay, dot, text } = this.#openOverlay("gaze.validationAria");
    const check = () => {
      if (this.#cancelled || this.#released || this.#paused || !session.alive || document.hidden) throw new Error(Texts.get("gaze.validationCancelled"));
      if (!BrowserViewport.same(context, BrowserViewport.current()) || revision !== session.calibrationRevision) throw new Error(Texts.get("gaze.validationChanged"));
      if (!session.fresh() || session.sample?.status !== "ok") throw new Error(Texts.get("gaze.validationLostFace"));
    };
    const escape = this.#escape();
    document.addEventListener("keydown", escape);
    try {
      let count = 0;
      for (const target of GazeTargets.validation(context.width, context.height)) {
        check();
        text.textContent = Texts.get("gaze.validationTarget", { count: ++count });
        await ScreenGazePanel.#showTarget(dot, target);
        check();
        this.#validation.beginTarget(target, performance.now(), session.sampleTs);
        await this.#hold(dot);
        check();
        this.#validation.finishTarget();
      }
      const report = this.#validation.result(session.calibrationPoints);
      session.recordValidation(report);
      PanelDom.setText(this.#message, "gaze.validationComplete", { mean: report.mean_error_px.toFixed(1), rmse: report.rmse_px.toFixed(1) });
    } finally {
      this.#validation = null;
      document.removeEventListener("keydown", escape);
      overlay.remove();
      this.#overlay = null;
    }
  }

  /**
   * Prova a schermo intero: la pagina va a schermo intero; con una calibrazione valida accende il pallino,
   * altrimenti dice di calibrare così.
   */

  async #fullscreen() {
    if (!document.fullscreenElement) {
      if (!document.documentElement.requestFullscreen) throw new Error(Texts.get("gaze.fullscreenUnavailable"));
      await document.documentElement.requestFullscreen();
      await new Promise(resolve => requestAnimationFrame(resolve));
    }
    this.#session.checkGeometry();
    if (!this.#session.calibration || this.#session.invalid || !this.#session.dimensionsMatch()) {
      PanelDom.setText(this.#message, "gaze.fullscreenHint");
    } else {
      this.#session.setEnabled(true);
      this.#paint();
    }
  }
}
