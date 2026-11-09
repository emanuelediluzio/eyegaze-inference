/* iride — app/run/panels/gaze-targets.js
 *
 * GazeTargets: dove compaiono i punti da guardare. Calibrazione: una griglia fissa 4 × 4 (16 punti, l'unica
 * calibrazione: ScreenPointOfGaze ne vuole esattamente 16), dall'8% al 92% della pagina. Verifica della precisione:
 * una griglia 3 × 3 (9 punti) che non coincide con quella della calibrazione, così misura punti non usati per
 * calibrare.
 * Da calibrationTargets e validationTargets di gaze-calibration.js di eyegaze-inference (integrations/n2ucore, del
 * collega). Nessun DOM, nessun timer.
 */

/**
 * Griglie dei punti di calibrazione e di verifica.
 */

export class GazeTargets {
  static CALIBRATION_COLUMNS = [0.08, 0.36, 0.64, 0.92];   // frazioni della larghezza (0.08 + 0.84 · i / 3)
  static CALIBRATION_ROWS = [0.08, 0.37, 0.63, 0.92];      // frazioni dell'altezza
  static VALIDATION_COLUMNS = [0.18, 0.50, 0.82];
  static VALIDATION_ROWS = [0.20, 0.50, 0.80];

  /**
   * @param {number} width - larghezza della pagina in pixel CSS
   * @param {number} height - altezza della pagina in pixel CSS
   * @returns {Array<Array<number>>} i 16 punti di calibrazione [x, y], riga per riga
   */

  static calibration(width, height) {
    return GazeTargets.#grid(width, height, GazeTargets.CALIBRATION_COLUMNS, GazeTargets.CALIBRATION_ROWS);
  }

  /**
   * @param {number} width - larghezza della pagina in pixel CSS
   * @param {number} height - altezza della pagina in pixel CSS
   * @returns {Array<Array<number>>} i 9 punti di verifica [x, y], riga per riga
   */

  static validation(width, height) {
    return GazeTargets.#grid(width, height, GazeTargets.VALIDATION_COLUMNS, GazeTargets.VALIDATION_ROWS);
  }

  /**
   * @param {number} width - larghezza in pixel
   * @param {number} height - altezza in pixel
   * @param {Array<number>} columns - frazioni della larghezza
   * @param {Array<number>} rows - frazioni dell'altezza
   * @returns {Array<Array<number>>} i punti [x, y] arrotondati, riga per riga
   */

  static #grid(width, height, columns, rows) {
    return rows.flatMap(y => columns.map(x => [Math.round(x * width), Math.round(y * height)]));
  }
}
