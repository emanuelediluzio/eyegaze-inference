/* iride — app/run/panels/gaze-quality.js
 *
 * GazeQuality: l'avviso della qualità di una stima dei nodi gaze (campo quality di "gaze-direction.v1", ripetuto
 * da ScreenPointOfGaze in "screen-point-of-gaze.v1"): score 0-100 da grandezza del volto e stabilità, e hint
 * ok, center_face, move_closer, move_back, unstable (score null finché la stabilità non è misurabile). Scrive nel
 * paragrafo dell'avviso cosa fare, o lo svuota se va tutto bene.
 * Allo stesso modo l'avviso di calibration_drift di ScreenPointOfGaze (testa spostata rispetto alla calibrazione) e
 * la guida di posizione: l'inquadratura con il volto della calibrazione (tratteggiato) e quello di adesso (pieno),
 * specchiati come in uno specchio, per tornare in posizione sovrapponendoli.
 * Usato dai pannelli di GazeDirection e ScreenPointOfGaze. Dipende da PanelDom (panel-dom.js).
 */

import { PanelDom } from "./panel-dom.js";

/**
 * Avvisi della qualità delle stime gaze.
 */

export class GazeQuality {
  static HINTS = new Set(["center_face", "move_closer", "move_back", "unstable"]);   // gli avvisi con un testo ("ok" no)
  static DRIFT_HINTS = new Set(["move_closer", "move_back", "move_left", "move_right", "move_up", "move_down"]);

  /**
   * @param {object|null} quality - il campo quality del campione, o null
   * @returns {string} il punteggio come "NN%", o "—" se manca
   */

  static scoreText(quality) {
    return Number.isFinite(quality?.score) ? quality.score + "%" : "—";
  }

  /**
   * Scrive l'avviso solo quando cambia; vuoto se non c'è avviso.
   * @param {HTMLElement} element - il paragrafo dell'avviso
   * @param {object|null} quality - il campo quality del campione, o null
   */

  static showHint(element, quality) {
    GazeQuality.#show(element, GazeQuality.HINTS.has(quality?.hint) ? "gaze.quality." + quality.hint : "");
  }

  /**
   * L'avviso dello spostamento dalla calibrazione; vuoto se la testa è dove era durante la calibrazione.
   * @param {HTMLElement} element - il paragrafo dell'avviso
   * @param {object|null} drift - il campo calibration_drift del campione, o null
   */

  static showDrift(element, drift) {
    GazeQuality.#show(element, drift?.drifted && GazeQuality.DRIFT_HINTS.has(drift.hint) ? "gaze.drift." + drift.hint : "");
  }

  /**
   * Crea la guida di posizione (nascosta finché non c'è una calibrazione con la posizione della testa).
   * @returns {HTMLElement} la guida, da mettere nel pannello
   */

  static guide() {
    const guide = PanelDom.element("div", "gaze-pose-guide");
    guide.append(PanelDom.element("span", "gaze-pose-box reference"), PanelDom.element("span", "gaze-pose-box current"));
    guide.hidden = true;
    return guide;
  }

  /**
   * Aggiorna la guida: i due volti in percentuale dell'inquadratura, specchiati in orizzontale.
   * @param {HTMLElement} guide - la guida di guide()
   * @param {object|null} drift - il campo calibration_drift del campione, o null per nasconderla
   * @param {number|null} aspect - altezza / larghezza dell'immagine della camera (frame_aspect)
   */

  static showGuide(guide, drift, aspect) {
    const usable = Array.isArray(drift?.reference) && Array.isArray(drift?.current) && Number.isFinite(aspect) && aspect > 0;
    guide.hidden = !usable;
    if (!usable) return;
    guide.style.aspectRatio = `${1 / aspect}`;
    const [reference, current] = guide.children;
    GazeQuality.#place(reference, drift.reference, aspect);
    GazeQuality.#place(current, drift.current, aspect);
    current.classList.toggle("aligned", !drift.drifted);
  }

  /**
   * @param {HTMLElement} box - il riquadro di un volto
   * @param {Array<number>} pose - [centro x, centro y, lato] in unità di larghezza dell'immagine
   * @param {number} aspect - altezza / larghezza dell'immagine
   */

  static #place(box, [cx, cy, size], aspect) {
    box.style.left = `${100 * (1 - cx - size / 2)}%`;     // specchiato: come ci si vede allo specchio
    box.style.top = `${100 * (cy - size / 2) / aspect}%`;
    box.style.width = `${100 * size}%`;
    box.style.height = `${100 * size / aspect}%`;
  }

  /**
   * @param {HTMLElement} element - il paragrafo dell'avviso
   * @param {string} key - chiave di Texts dell'avviso, "" per nessun avviso
   */

  static #show(element, key) {
    if (element.dataset.hint === key) return;
    element.dataset.hint = key;
    element.classList.toggle("warn", key !== "");
    if (key) PanelDom.setText(element, key);
    else {
      delete element.dataset.text;
      element.textContent = "";
    }
  }
}
