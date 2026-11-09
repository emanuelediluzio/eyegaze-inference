/* iride — app/run/panels/gaze-direction-panel.js
 *
 * GazeDirectionPanel: il pannello della scheda di un nodo GazeDirection (dyn/eyegaze.py), al posto dei mattoncini
 * (NodePanels). A sinistra il video che entra nel nodo (ingresso VideoStream, la sorgente dal collegamento del
 * Design: inputSources) con sopra la direzione dello sguardo; sotto, gli interruttori Riquadro volto, Frecce, Punti
 * occhi e Specchio, e una riga di aiuto. A destra lo sguardo in tempo reale: yaw, pitch in gradi e stime al
 * secondo, la qualità della stima (0-100, da grandezza del volto e stabilità) con l'avviso di distanza.
 * Nell'intestazione lo stato del campione. updated (2026-11-05): tolti i dettagli tecnici (l'ultimo campione
 * JSON a scomparsa), decisione dell'utente.
 * Il disegno è tutto nel Worker dei dati: il video è una vista immagine (canvas ceduto) e la direzione la disegna
 * OverlayRenderer sopra ogni fotogramma (modo "direction", GazeDirectionDrawing), sincronizzata per packID; gli
 * interruttori la comandano con control "gazeLayers". Specchio è solo CSS sul canvas. I valori arrivano dalla vista
 * dei campioni (GazeSampleStream) e si scrivono solo quando cambiano. Senza campioni da STALE_MS: "nessun dato
 * recente" e valori a "—".
 * updated (2026-11-06): nel lettore i canali hanno il nome che dà loro il Player (channelNames, dalla scheda) e un
 * errore delle viste (canale che non c'è ancora: il Player li crea al play) resta nello stato solo finché la vista non
 * riparte o arriva un campione.
 * Da GazeDirectionPanel di eyegaze-inference (integrations/n2ucore, del collega). Differenze: niente SVG sul thread
 * principale (vincolo di iride: decodifica e disegno dei dati nel Worker), video e direzione sullo stesso canvas.
 * Dipende da PanelDom (panel-dom.js), GazeQuality (gaze-quality.js), GazeSampleStream (gaze-sample-stream.js),
 * GazeStatus (gaze-status.js), Texts (lib/core/texts.js), DataSocket (passato da chi crea il pannello).
 */

import { Texts } from "../../../lib/core/texts.js";
import { GazeQuality } from "./gaze-quality.js";
import { GazeSampleStream } from "./gaze-sample-stream.js";
import { GazeStatus } from "./gaze-status.js";
import { PanelDom } from "./panel-dom.js";

/**
 * Pannello di GazeDirection: video con la direzione dello sguardo e valori attuali.
 */

export class GazeDirectionPanel {
  static CHANNELS = ["GazeDirection"];   // canali del nodo che il pannello mostra (Card: filmstrip, setVisible)
  static PREFERRED_WIDTH = 780;          // larghezza della scheda che il pannello chiede
  static INPUT = "VideoStream";          // l'ingresso del video
  static COLOR = "#ffffff";              // colore delle frecce (riquadro e occhi: GazeDirectionDrawing.FACE_COLOR)
  static STALE_MS = 1500;
  static CHECK_MS = 400;
  static RATE_SMOOTHING = 0.2;           // peso dell'ultimo intervallo nella media delle stime al secondo
  static #next = 0;

  #options = null;
  #root = null;
  #stage = null;
  #canvas = null;
  #status = null;
  #values = [];          // yaw, pitch, stime al secondo, qualità (dd)
  #qualityHint = null;   // l'avviso della qualità (avvicinati, allontanati, stabilità)
  #stream = null;
  #timer = null;
  #videoId = "";
  #videoShown = false;
  #layers = { box: true, eyes: true, arrows: true };
  #last = 0;
  #rate = 0;
  #alive = true;
  #paused = false;
  #visible = true;
  #released = false;
  #errorShown = false;   // updated (2026-11-06): lo stato mostra un errore di una vista

  /**
   * @param {{nodeName: string, renderer: DataSocket, alive?: boolean,
   *   inputSources?: Object<string, {nodeName: string, chanName: string}>,
   *   channelNames?: Object<string, string>}} options - nome del nodo, Worker dei dati, se il nodo è vivo, le sorgenti
   *   dei suoi ingressi e, updated (2026-11-06), il nome vero dei suoi canali (nome di dyndefs.json -> nome)
   */

  constructor(options) {
    this.#options = options;
    this.#alive = options.alive ?? true;
    this.#videoId = "gaze-video-" + (++GazeDirectionPanel.#next);
  }

  /**
   * Crea il pannello nel contenitore (la casella della scheda).
   * @param {HTMLElement} container - dove metterlo
   */

  render(container) {
    this.#root = PanelDom.element("section", "gaze-panel gaze-direction");
    const header = PanelDom.element("header", "gaze-heading");
    this.#status = PanelDom.translated("span", "gaze-status", "gaze.waitingFace");
    header.append(PanelDom.translated("strong", "", "gaze.directionTitle"), this.#status);
    const layout = PanelDom.element("div", "gaze-layout");
    const main = PanelDom.element("div", "gaze-main");
    this.#stage = PanelDom.element("div", "gaze-video-stage");
    this.#canvas = this.#newCanvas();
    this.#stage.append(this.#canvas);
    const switches = PanelDom.element("div", "gaze-switches");
    for (const [key, layer] of [["gaze.faceBox", "box"], ["gaze.arrows", "arrows"], ["gaze.eyes", "eyes"]]) {
      switches.append(PanelDom.checkbox(key, true, (checked) => { this.#layers[layer] = checked; this.#sendLayers(); }).row);
    }
    switches.append(PanelDom.checkbox("gaze.mirror", false, (checked) => this.#stage.classList.toggle("mirrored", checked)).row);
    main.append(this.#stage, switches, PanelDom.translated("p", "gaze-hint", "gaze.directionHint"));
    const side = PanelDom.element("aside", "gaze-readings");
    side.append(PanelDom.translated("h3", "", "gaze.liveGaze"));
    const metrics = PanelDom.element("dl", "gaze-metrics");
    for (const key of ["gaze.yaw", "gaze.pitch", "gaze.rate", "gaze.quality"]) {
      const value = PanelDom.element("dd", "", "—");
      this.#values.push(value);
      metrics.append(PanelDom.translated("dt", "", key), value);
    }
    this.#qualityHint = PanelDom.element("p", "gaze-quality-hint");
    this.#qualityHint.setAttribute("aria-live", "polite");
    side.append(metrics, this.#qualityHint);
    layout.append(main, side);
    this.#root.append(header, layout);
    container.replaceChildren(this.#root);
    this.#stream = new GazeSampleStream(this.#options.renderer, { nodeName: this.#options.nodeName, chanName: this.#channel() },
      (sample) => this.#update(sample), (state, detail) => this.#onViewState(state, detail));
  }

  /**
   * Con la scheda nella pagina: accende la vista dei campioni e il video, e il controllo dei dati vecchi.
   */

  attach() {
    if (this.#timer || this.#released) return;
    this.#stream.attach();
    this.#showVideo();
    this.#timer = setInterval(() => this.#checkStale(), GazeDirectionPanel.CHECK_MS);
  }

  /**
   * Il canale del pannello mostrato o nascosto dalla filmstrip: nascosto, spegne video e campioni.
   * @param {boolean} visible - true = mostrato
   */

  setVisible(visible) {
    this.#visible = visible;
    this.#stream?.setVisible(visible);
    if (visible) this.#showVideo();
    else this.#hideVideo();
  }

  /**
   * Scheda ridotta o parcheggiata: video e campioni in pausa.
   * @param {boolean} paused - true = in pausa
   */

  setPaused(paused) {
    this.#paused = paused;
    this.#stream?.pause(paused);
    if (this.#videoShown) this.#options.renderer.control(this.#videoId, "pause", { paused });
    if (paused) this.#clear();
  }

  /**
   * @param {boolean} alive - true se il nodo è vivo
   */

  setAlive(alive) {
    this.#alive = alive;
    if (!alive) {
      this.#clear();
      this.#setStatus("gaze.nodeDisconnected");
    }
  }

  /**
   * La lingua è cambiata: riscrive i testi del pannello.
   */

  refreshTexts() {
    if (!this.#root) return;
    PanelDom.refreshTexts(this.#root);
    this.#canvas.setAttribute("aria-label", Texts.get("gaze.videoAria"));
  }

  /**
   * La scheda sta per sparire: spegne tutto.
   */

  release() {
    this.#released = true;
    clearInterval(this.#timer);
    this.#timer = null;
    this.#stream?.release();
    this.#hideVideo();
  }

  /**
   * @returns {HTMLCanvasElement} un canvas nuovo per il video (uno ceduto al Worker non si riusa)
   */

  #newCanvas() {
    const canvas = PanelDom.element("canvas", "gaze-video");
    canvas.setAttribute("aria-label", Texts.get("gaze.videoAria"));
    return canvas;
  }

  /**
   * Accende il video: il canale collegato all'ingresso VideoStream, con sopra la direzione dello sguardo.
   */

  #showVideo() {
    if (this.#videoShown || this.#released || !this.#visible) return;
    const source = this.#options.inputSources?.[GazeDirectionPanel.INPUT];
    if (!source || !this.#canvas.transferControlToOffscreen) {
      this.#setStatus("gaze.connectVideo");
      return;
    }
    // la vista iscrive anche il canale della direzione: il Worker lo disegna sopra il video (overlay acceso)
    const direction = { nodeName: this.#options.nodeName, chanName: this.#channel() };
    const overlay = direction.nodeName + "." + direction.chanName;
    this.#options.renderer.show(this.#videoId, [source, direction], this.#canvas.transferControlToOffscreen(),
      { paused: this.#paused, overlays: [{ channel: overlay, color: GazeDirectionPanel.COLOR, on: true }] },
      { onState: (state, detail) => this.#onViewState(state, detail) });
    this.#videoShown = true;
    this.#sendLayers();
  }

  // updated (2026-11-06)
  /**
   * @returns {string} il nome vero del canale della direzione (nel lettore quello dato dal Player)
   */

  #channel() {
    const name = GazeDirectionPanel.CHANNELS[0];
    return this.#options.channelNames?.[name] ?? name;
  }

  // updated (2026-11-06)
  /**
   * Stato di una vista (video o campioni): un errore va nello stato; quando la vista riparte l'errore si toglie.
   * @param {string} state - "waiting", "subscribing", "showing" o "error"
   * @param {string|null} detail - per "error" il messaggio di uinode
   */

  #onViewState(state, detail) {
    if (state === "error") {
      this.#errorShown = true;
      this.#setStatus(null, detail);
    } else if (this.#errorShown) {
      this.#errorShown = false;
      this.#setStatus("gaze.waitingFace");
    }
  }

  /**
   * Spegne il video e mette al suo posto un canvas nuovo, per quando si riaccende.
   */

  #hideVideo() {
    if (!this.#videoShown) return;
    this.#options.renderer.hide(this.#videoId);
    this.#videoShown = false;
    const fresh = this.#newCanvas();
    this.#canvas.replaceWith(fresh);
    this.#canvas = fresh;
  }

  /**
   * Dice al Worker quali parti della direzione disegnare.
   */

  #sendLayers() {
    if (this.#videoShown) this.#options.renderer.control(this.#videoId, "gazeLayers", { ...this.#layers });
  }

  /**
   * Un campione nuovo: stato, stime al secondo, valori e dettagli.
   * @param {object} sample - il campione "gaze-direction.v1"
   */

  #update(sample) {
    if (this.#released || this.#paused || !this.#alive || !sample) return;
    const now = performance.now();
    if (this.#last && now > this.#last) {
      const rate = 1000 / (now - this.#last);
      this.#rate = this.#rate ? (1 - GazeDirectionPanel.RATE_SMOOTHING) * this.#rate + GazeDirectionPanel.RATE_SMOOTHING * rate : rate;
    }
    this.#last = now;
    this.#errorShown = false;
    this.#setStatus(GazeStatus.keyOf(sample.status));
    this.#showMetrics(sample);
  }

  /**
   * Nessun campione da STALE_MS: valori a "—" e "nessun dato recente".
   */

  #checkStale() {
    if (!this.#last || performance.now() - this.#last <= GazeDirectionPanel.STALE_MS) return;
    this.#clear();
    this.#setStatus("gaze.noRecentData");
  }

  /**
   * Dimentica l'ultimo campione e rimette i valori a "—".
   */

  #clear() {
    this.#last = 0;
    this.#rate = 0;
    this.#showMetrics(null);
  }

  /**
   * @param {object|null} sample - il campione, o null per "—"
   */

  #showMetrics(sample) {
    const angle = (radians) => Number.isFinite(radians) ? (radians * 180 / Math.PI).toFixed(1) + "°" : "—";   // updated (2026-11-05): il campione ha solo i radianti
    const quality = sample?.status === "ok" ? sample.quality : null;
    const texts = [angle(sample?.yaw_rad), angle(sample?.pitch_rad), sample ? this.#rate.toFixed(1) : "—",
      GazeQuality.scoreText(quality)];
    texts.forEach((text, i) => { if (this.#values[i].textContent !== text) this.#values[i].textContent = text; });
    GazeQuality.showHint(this.#qualityHint, quality);
  }

  /**
   * @param {string|null} key - chiave di Texts dello stato, o null per un testo già pronto
   * @param {string|null} [text=null] - il testo già pronto (es. un errore di uinode)
   */

  #setStatus(key, text = null) {
    if (key) PanelDom.setText(this.#status, key);
    else if (text) {
      delete this.#status.dataset.text;
      this.#status.textContent = text;
    }
  }
}
