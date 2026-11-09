/* iride — app/run/panels/screen-gaze-session.js
 *
 * ScreenGazeSession: lo stato del punto guardato sullo schermo di un nodo ScreenPointOfGaze per tutta la pipeline,
 * indipendente dalle schede (PanelSessions, una per nodo). Tiene:
 *   - l'ultimo campione "screen-point-of-gaze.v1" e quando è arrivato (fresh: al più FRESH_MS fa, con stato "ok" o
 *     "uncalibrated");
 *   - la calibrazione del nodo (getCalibration) e le misure della pagina con cui è stata fatta (BrowserViewport):
 *     se la finestra cambia posizione, misura o densità la calibrazione non vale più (invalid);
 *   - l'ultima verifica della precisione (validationResults), che una calibrazione nuova cancella;
 *   - il pallino sulla pagina (classe gaze-page-dot), acceso dall'utente (enabled) e mostrato solo quando il punto
 *     vale: nodo vivo, pagina visibile, campione recente e "ok", calibrazione valida per la pagina di adesso, dentro
 *     la pagina. Resta acceso anche chiudendo la scheda; durante calibrazione e verifica si nasconde (setSuppressed).
 * La vista dei campioni (GazeSampleStream) è accesa finché il pallino è acceso o qualche pannello ascolta
 * (subscribe). Ogni CHECK_MS ricontrolla le misure della pagina e la freschezza del campione.
 * Da ScreenGazeSession di eyegaze-inference (integrations/n2ucore, del collega). Differenze: i campioni arrivano già
 * letti dal Worker (GazeSampleStream); validPoint di gaze-ui.js è qui, statico.
 * Dipende da GazeSampleStream (gaze-sample-stream.js), BrowserViewport (browser-viewport.js), PanelDom (panel-dom.js).
 */

import { BrowserViewport } from "./browser-viewport.js";
import { GazeSampleStream } from "./gaze-sample-stream.js";
import { PanelDom } from "./panel-dom.js";

/**
 * Punto guardato sullo schermo, calibrazione e pallino sulla pagina di un nodo, per tutta la pipeline.
 */

export class ScreenGazeSession {
  static CHANNEL = "ScreenPoint";
  static CALIBRATION_POINTS = 16;
  static FRESH_MS = 1500;
  static CHECK_MS = 300;

  sample = null;              // ultimo campione
  sampleTs = null;            // tempo del frame dell'ultimo campione, µs
  last = 0;                   // quando è arrivato (performance.now), 0 = nessuno recente
  calibration = null;         // calibrazione del nodo, o null
  geometry = null;            // misure della pagina calibrata (BrowserViewport.current)
  invalid = false;            // la pagina è cambiata dopo la calibrazione
  enabled = false;            // pallino sulla pagina acceso dall'utente
  alive = true;               // nodo vivo
  calibrationPoints = null;   // punti della calibrazione (16), o null
  calibrationRevision = 0;    // cresce a ogni calibrazione nuova
  validationResults = [];     // l'ultima verifica della precisione, o vuoto

  #dot = null;
  #stream = null;
  #timer = null;
  #listeners = new Set();
  #released = false;
  #suppressed = false;
  #loading = null;
  #initialized = false;
  #onVisibility = null;
  #onResize = null;

  /**
   * @param {{renderer: DataSocket, nodeName: string, chanName?: string, alive?: boolean}} options - il Worker dei dati,
   *   il nome del nodo, updated (2026-11-06) il nome vero del canale (nel lettore quello del Player; CHANNEL se manca)
   *   e se è vivo
   */

  constructor(options) {
    this.alive = options.alive ?? true;
    this.#dot = PanelDom.element("div", "gaze-page-dot");
    this.#dot.hidden = true;
    this.#dot.setAttribute("aria-hidden", "true");
    document.body.append(this.#dot);
    this.#stream = new GazeSampleStream(options.renderer, { nodeName: options.nodeName, chanName: options.chanName ?? ScreenGazeSession.CHANNEL },
      (sample, ts) => this.update(sample, ts), (state) => { if (state === "error") { this.last = 0; this.#changed(); } });
    this.#onVisibility = () => { if (document.hidden) this.last = 0; this.#changed(); };
    this.#onResize = () => { this.checkGeometry(); this.#changed(); };
    document.addEventListener("visibilitychange", this.#onVisibility);
    document.addEventListener("fullscreenchange", this.#onResize);
    window.addEventListener("resize", this.#onResize);
    this.#timer = setInterval(() => { this.checkGeometry(); this.#changed(); }, ScreenGazeSession.CHECK_MS);
  }

  /**
   * @param {object|null} sample - un campione di ScreenPointOfGaze
   * @param {{width: number, height: number}} size - misure della pagina
   * @returns {boolean} true se è un punto "ok" calcolato per una pagina di quelle misure e dentro la pagina
   */

  static validPoint(sample, size) {
    const point = sample?.screen_xy_px;
    return sample?.status === "ok"
      && Array.isArray(point) && point.length === 2 && point.every(Number.isFinite)
      && Array.isArray(sample.screen_size_px) && sample.screen_size_px[0] === size.width && sample.screen_size_px[1] === size.height
      && point[0] >= 0 && point[0] < size.width && point[1] >= 0 && point[1] < size.height;
  }

  /**
   * Un pannello ascolta la sessione: riceve subito lo stato e poi a ogni cambio.
   * @param {function(ScreenGazeSession): void} listener - chiamata con la sessione
   * @returns {function(): void} smette di ascoltare
   */

  subscribe(listener) {
    if (this.#released) return () => {};
    this.#listeners.add(listener);
    this.#syncStream();
    listener(this);
    return () => { this.#listeners.delete(listener); this.#syncStream(); };
  }

  /**
   * Chiede una volta sola la calibrazione del nodo (la prima volta che un pannello si apre).
   * @param {function(): Promise<object|null>} load - legge la calibrazione (comando getCalibration)
   * @returns {Promise<void>|undefined} la richiesta in corso, se c'è
   */

  ensureCalibration(load) {
    if (this.#initialized || this.#released) return undefined;
    if (!this.#loading) {
      this.#loading = Promise.resolve().then(load).then((value) => {
        if (!this.#released && !this.#initialized) this.setCalibration(value);
      }).finally(() => { this.#loading = null; });
    }
    return this.#loading;
  }

  /**
   * Una calibrazione nuova (o quella letta dal nodo): cancella la verifica precedente.
   * @param {object|null} value - la calibrazione, o null
   * @param {object} [geometry] - misure della pagina con cui è stata fatta (adesso, se manca)
   */

  setCalibration(value, geometry = BrowserViewport.current()) {
    if (this.#released) return;
    this.#initialized = true;
    this.calibration = value;
    this.geometry = geometry;
    this.calibrationPoints = value ? ScreenGazeSession.CALIBRATION_POINTS : null;
    this.calibrationRevision++;
    this.validationResults = [];
    this.invalid = !!value && !this.dimensionsMatch();
    this.#changed();
  }

  /**
   * @param {object} report - il risultato di una verifica (GazeValidation.result)
   */

  recordValidation(report) {
    if (this.#released) return;
    this.validationResults = [report];
    this.#notify();
  }

  /**
   * Accende o spegne il pallino sulla pagina; non si accende senza una calibrazione valida per questa pagina.
   * @param {boolean} enabled - true = acceso
   */

  setEnabled(enabled) {
    if (this.#released) return;
    if (enabled && (this.invalid || !this.dimensionsMatch())) return;
    this.enabled = enabled;
    this.#syncStream();
    this.#changed();
  }

  /**
   * @param {boolean} alive - true se il nodo è vivo
   */

  setAlive(alive) {
    this.alive = alive;
    if (!alive) this.last = 0;
    this.#changed();
  }

  /**
   * @param {boolean} value - true = pallino nascosto (calibrazione o verifica in corso)
   */

  setSuppressed(value) {
    this.#suppressed = value;
    this.#paint();
  }

  /**
   * @returns {boolean} true se l'ultimo campione è recente e ha lo stato "ok" o "uncalibrated"
   */

  fresh() {
    return this.last > 0 && performance.now() - this.last < ScreenGazeSession.FRESH_MS
      && ["ok", "uncalibrated"].includes(this.sample?.status);
  }

  /**
   * @returns {boolean} true se la calibrazione è per una pagina larga e alta come quella di adesso
   */

  dimensionsMatch() {
    const viewport = BrowserViewport.current();
    return this.calibration?.screen_w === viewport.width && this.calibration?.screen_h === viewport.height;
  }

  /**
   * La pagina calibrata non è più quella di adesso: la calibrazione non vale più.
   */

  checkGeometry() {
    if (this.geometry && !BrowserViewport.same(this.geometry, BrowserViewport.current())) this.invalid = true;
  }

  /**
   * Un campione nuovo (dalla vista dei campioni).
   * @param {object} sample - il campione
   * @param {number} ts - tempo del suo frame, µs (updated 2026-11-05: il campione non ha più source_ts_us)
   */

  update(sample, ts) {
    if (this.#released || !this.alive) return;
    this.sample = sample;
    this.sampleTs = ts;
    this.last = performance.now();
    this.checkGeometry();
    this.#changed();
  }

  /**
   * Spegne tutto: vista, pallino, timer e ascoltatori.
   */

  release() {
    if (this.#released) return;
    this.#released = true;
    this.enabled = false;
    clearInterval(this.#timer);
    this.#stream.release();
    this.#listeners.clear();
    this.#dot.remove();
    document.removeEventListener("visibilitychange", this.#onVisibility);
    document.removeEventListener("fullscreenchange", this.#onResize);
    window.removeEventListener("resize", this.#onResize);
  }

  /**
   * Accende la vista dei campioni se serve (pallino acceso o qualcuno ascolta), altrimenti la spegne.
   */

  #syncStream() {
    this.#stream.setVisible(!this.#released && (this.enabled || this.#listeners.size > 0));
  }

  /**
   * Qualcosa è cambiato: ridisegna il pallino e avvisa chi ascolta.
   */

  #changed() {
    this.#paint();
    this.#notify();
  }

  /**
   * Avvisa chi ascolta.
   */

  #notify() {
    for (const listener of this.#listeners) listener(this);
  }

  /**
   * Mostra il pallino sul punto guardato, se è acceso e il punto vale; altrimenti lo nasconde.
   */

  #paint() {
    const usable = !this.#released && !this.#suppressed && this.alive && !document.hidden && this.fresh()
      && !this.invalid && this.dimensionsMatch() && ScreenGazeSession.validPoint(this.sample, BrowserViewport.current());
    this.#dot.hidden = !this.enabled || !usable;
    if (!usable) return;
    const [x, y] = this.sample.screen_xy_px;
    this.#dot.style.left = `${x}px`;
    this.#dot.style.top = `${y}px`;
  }
}
