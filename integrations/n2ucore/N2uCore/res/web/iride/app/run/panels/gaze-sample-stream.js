/* iride — app/run/panels/gaze-sample-stream.js
 *
 * GazeSampleStream: i campioni JSON di un canale GAZE di un nodo, per un pannello. È una vista "sample" del Worker
 * dei dati (DataSocket.show senza canvas, kind "sample"): il Worker sottoscrive il canale, lo legge con GazeDecoder
 * e manda ogni campione come viewInfo { topic: "sample", sample } (data-worker.js, #postSample). Il JSON non si
 * rilegge qui: arriva già come oggetto. La vista si accende con attach o setVisible(true) e si spegne con
 * setVisible(false) o release; in pausa il Worker non manda campioni.
 * updated (2026-11-06): chi la crea riceve ogni stato della vista (onState: "waiting", "subscribing", "showing",
 * "error"), non solo gli errori: così toglie un errore quando la vista riparte (nel lettore i canali nascono al play).
 * Da GazeStream di eyegaze-inference (integrations/n2ucore, del collega), che usava una vista "text" con il testo
 * del frame. Differenze: il campione è letto nel Worker; un solo campione alla volta, l'ultimo.
 * Dipende da DataSocket (lib/net/data-socket.js), passato da chi crea il pannello.
 */

/**
 * Vista dei campioni di un canale GAZE.
 */

export class GazeSampleStream {
  static KIND = "sample";   // kind della vista nel Worker (DataWorker.SAMPLE_KIND)
  static #next = 0;

  #renderer = null;
  #channel = null;
  #onSample = null;
  #onState = null;
  #id = "";
  #active = false;
  #released = false;

  /**
   * @param {DataSocket} renderer - il Worker dei dati
   * @param {{nodeName: string, chanName: string}} channel - il canale GAZE da leggere
   * @param {function(object, number): void} onSample - chiamata con ogni campione letto e il tempo del suo frame (µs)
   * @param {function(string, string|null): void} [onState] - updated (2026-11-06): stato della vista e dettaglio
   *   (per "error" il messaggio di uinode)
   */

  constructor(renderer, channel, onSample, onState = () => {}) {
    this.#renderer = renderer;
    this.#channel = channel;
    this.#onSample = onSample;
    this.#onState = onState;
    this.#id = "gaze-sample-" + (++GazeSampleStream.#next);
  }

  /**
   * Accende la vista, se non è già accesa o rilasciata.
   */

  attach() {
    if (this.#active || this.#released || !this.#renderer) return;
    this.#active = true;
    this.#renderer.show(this.#id, [this.#channel], null, { kind: GazeSampleStream.KIND }, {
      onInfo: (info) => { if (!this.#released && info?.topic === "sample" && info.sample) this.#onSample(info.sample, info.ts); },
      onState: (state, detail) => this.#onState(state, detail ?? null)
    });
  }

  /**
   * @param {boolean} visible - true = vista accesa, false = spenta (il Worker disiscrive il canale se nessun altro lo usa)
   */

  setVisible(visible) {
    if (visible) this.attach();
    else if (this.#active) {
      this.#renderer.hide(this.#id);
      this.#active = false;
    }
  }

  /**
   * @param {boolean} paused - true = il Worker non manda campioni (la sottoscrizione resta)
   */

  pause(paused) {
    if (this.#active) this.#renderer.control(this.#id, "pause", { paused });
  }

  /**
   * Spegne la vista per sempre.
   */

  release() {
    this.#released = true;
    this.setVisible(false);
  }
}
