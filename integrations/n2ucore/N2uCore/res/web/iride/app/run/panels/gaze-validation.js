/* iride — app/run/panels/gaze-validation.js
 *
 * GazeValidation: la verifica della precisione di una calibrazione dello sguardo sullo schermo, su 9 punti nuovi
 * (GazeTargets.validation), tutta nel browser: la calibrazione non cambia. Per ogni punto (beginTarget) raccoglie
 * la distanza tra il punto guardato (screen_xy_px dei campioni di ScreenPointOfGaze) e il punto da guardare, solo
 * per i campioni nuovi (tempo del frame crescente), dopo il primo SETTLE_MS (lo sguardo si sposta) e fino a
 * WINDOW_MS, con il tempo della sorgente almeno SOURCE_SETTLE_US dopo l'inizio, al più MAX_SAMPLES. Un punto vale
 * con almeno MIN_SAMPLES campioni (finishTarget). Il risultato (result) ha errore medio, RMSE (i 9 punti pesano
 * uguale), mediana e 95° percentile (su tutti i campioni), in pixel CSS della pagina.
 * Da GazeValidation di gaze-calibration.js di eyegaze-inference (integrations/n2ucore, del collega).
 * Dipende da Texts (lib/core/texts.js) per i messaggi d'errore e da ScreenGazeSession.validPoint.
 * Nessun DOM, nessun timer.
 */

import { Texts } from "../../../lib/core/texts.js";
import { ScreenGazeSession } from "./screen-gaze-session.js";

/**
 * Raccolta degli errori sui 9 punti di verifica e loro riassunto.
 */

export class GazeValidation {
  static TARGETS = 9;
  static SETTLE_MS = 700;               // updated (2026-11-05): punti da 2 s, raccolta da 0,7 s come nel nodo (COLLECT_SECS)
  static WINDOW_MS = 2000;
  static SOURCE_SETTLE_US = 700000;
  static MIN_SAMPLES = 3;
  static MAX_SAMPLES = 128;

  #size = null;
  #active = null;     // { target, started, sourceAfterUs, latest, errors } del punto in corso
  #targets = [];      // i punti finiti, come #active

  /**
   * @param {{width: number, height: number}} size - misure della pagina verificata (BrowserViewport.current)
   */

  constructor(size) {
    this.#size = size;
  }

  /**
   * Comincia un punto.
   * @param {Array<number>} target - il punto da guardare [x, y]
   * @param {number} now - adesso (performance.now)
   * @param {number} sourceAfterUs - tempo del frame (µs) dell'ultimo campione: valgono quelli dopo
   */

  beginTarget(target, now, sourceAfterUs) {
    if (!Number.isSafeInteger(sourceAfterUs)) throw new Error(Texts.get("gaze.validationTimestampMissing"));
    this.#active = { target, started: now, sourceAfterUs, latest: sourceAfterUs, errors: [] };
  }

  /**
   * Un campione di ScreenPointOfGaze: se vale per il punto in corso, ne tiene la distanza dal punto.
   * @param {object|null} sample - il campione
   * @param {number} ts - tempo del suo frame, µs
   * @param {number} now - adesso (performance.now)
   */

  observe(sample, ts, now) {
    const active = this.#active;
    if (!active || !ScreenGazeSession.validPoint(sample, this.#size) || !Number.isSafeInteger(ts) || ts <= active.latest) return;
    active.latest = ts;
    const elapsed = now - active.started;
    if (elapsed < GazeValidation.SETTLE_MS || elapsed > GazeValidation.WINDOW_MS) return;
    if (ts < active.sourceAfterUs + GazeValidation.SOURCE_SETTLE_US || active.errors.length >= GazeValidation.MAX_SAMPLES) return;
    const [x, y] = sample.screen_xy_px;
    active.errors.push(Math.hypot(x - active.target[0], y - active.target[1]));
  }

  /**
   * Chiude il punto in corso; senza abbastanza campioni la verifica non vale.
   */

  finishTarget() {
    const active = this.#active;
    if (!active || active.errors.length < GazeValidation.MIN_SAMPLES) throw new Error(Texts.get("gaze.validationSamplesMissing"));
    this.#targets.push(active);
    this.#active = null;
  }

  /**
   * @param {number|null} calibrationPoints - i punti della calibrazione verificata
   * @returns {object} il risultato "screen-gaze-validation.v1": errore medio, RMSE, mediana, P95 e i 9 punti
   */

  result(calibrationPoints) {
    if (this.#active || this.#targets.length !== GazeValidation.TARGETS) throw new Error(Texts.get("gaze.validationTargetsMissing"));
    const targets = this.#targets.map(({ target, errors }) => ({
      target_px: target,
      samples: errors.length,
      mean_error_px: errors.reduce((a, b) => a + b, 0) / errors.length,
      mean_squared_error_px: errors.reduce((a, b) => a + b * b, 0) / errors.length
    }));
    const errors = this.#targets.flatMap(target => target.errors);
    return {
      schema: "screen-gaze-validation.v1",
      calibration_points: calibrationPoints,
      coordinate_space: "viewport_css_pixels",
      viewport: { ...this.#size },
      tested_at: new Date().toISOString(),
      target_count: targets.length,
      samples: errors.length,
      mean_error_px: targets.reduce((sum, target) => sum + target.mean_error_px, 0) / targets.length,
      rmse_px: Math.sqrt(targets.reduce((sum, target) => sum + target.mean_squared_error_px, 0) / targets.length),
      median_error_px: GazeValidation.#percentile(errors, 0.5),
      p95_error_px: GazeValidation.#percentile(errors, 0.95),
      targets
    };
  }

  /**
   * @param {Array<number>} values - i valori (almeno uno)
   * @param {number} fraction - da 0 a 1
   * @returns {number} il percentile, interpolato tra i due valori vicini
   */

  static #percentile(values, fraction) {
    const sorted = [...values].sort((a, b) => a - b);
    const index = (sorted.length - 1) * fraction;
    const low = Math.floor(index);
    const high = Math.ceil(index);
    return sorted[low] + (sorted[high] - sorted[low]) * (index - low);
  }
}
