/* iride — app/run/panels/gaze-status.js
 *
 * GazeStatus: la chiave di Texts dello stato di un campione dei nodi gaze (campo status di "gaze-direction.v1" e
 * "screen-point-of-gaze.v1": ok, no_face, uncalibrated, model_unavailable, error). Uno stato sconosciuto o mancante
 * vale "in attesa del volto". updated (2026-11-05): i nodi non pubblicano più model_ready, stale_input,
 * input_disconnected (la disconnessione arriva come evento INPUT_DISCONNECTION).
 * Da statusText di gaze-ui.js di eyegaze-inference (integrations/n2ucore, del collega).
 * Nessun DOM, nessun timer.
 */

/**
 * Stati dei campioni gaze e loro testi.
 */

export class GazeStatus {
  static KNOWN = new Set(["ok", "no_face", "uncalibrated", "model_unavailable", "error"]);

  /**
   * @param {string|undefined} status - lo stato del campione
   * @returns {string} la chiave di Texts che lo descrive
   */

  static keyOf(status) {
    return GazeStatus.KNOWN.has(status) ? "gaze.status." + status : "gaze.waitingFace";
  }
}
