/* iride — gaze-decoder.js
 *
 * GazeDecoder: legge i frame dei canali GAZE (un JSON per campione), dentro il Worker dei dati
 * (caricato da data-worker.js con import, modulo ES).
 * updated (2026-11-04): tre formati sullo stesso tipo GAZE, distinti dalla chiave "schema" (quella delle
 * proprietà del canale, nota dalla risposta al subscribe; se manca, quella del campione):
 *   - nessuno schema: lo sguardo degli occhiali Tobii. Dal JSON prende il diametro delle due pupille
 *     (eyeleft.pupildiameter, eyeright.pupildiameter, in mm) come una riga di due colonne con i loro nomi, per i
 *     grafici a linee, e il punto guardato (gaze2d, da 0 a 1 sull'immagine della telecamera frontale), per il
 *     disegno sopra il video;
 *   - "gaze-direction.v1" (nodo GazeDirection, dyn/eyegaze.py): yaw e pitch in gradi per i grafici, e per il
 *     disegno sopra il video (GazeDirectionDrawing) il riquadro del volto, i centri delle iridi e gli angoli in
 *     radianti; niente gaze2d;
 *   - "screen-point-of-gaze.v1" (nodo ScreenPointOfGaze, dyn/screenpointofgaze.py): X e Y del punto guardato sullo
 *     schermo, in pixel; il suo gaze2d è sullo schermo e non sull'immagine di una telecamera, quindi non si disegna.
 * Per gli ultimi due un valore vale solo con status "ok": altrimenti (nessun volto, non calibrato, ...) NaN.
 * Un campo mancante (occhio chiuso, sguardo non trovato) diventa NaN o null, ma il frame c'è
 * sempre: così il grafico continua a scorrere e mostra un buco, come in iride_dev.
 * Accetta il campione così com'è ({gaze2d, eyeleft, ...}) o dentro un campo "data".
 * Da processGazeSample di iride_dev/js/transport/datasocketworker.js. Differenze: restituisce la
 * matrice dei grafici (con i nomi delle colonne) invece dell'oggetto {gaze2d, leftPupil, rightPupil}.
 * updated (2026-11-04): gli schemi da GazeDecoder di eyegaze-inference (integrations/n2ucore, del collega). Differenze:
 * lo schema del canale vale prima di quello del campione, e il punto sullo schermo si legge dallo schema anche se il
 * campione ha gaze2d (ScreenPointOfGaze lo manda sempre).
 * Nessun DOM: vive nel Worker.
 */

/**
 * Lettura dei frame GAZE: valori per i grafici, punto guardato o direzione per il video.
 */

export class GazeDecoder {
  static COLUMN_NAMES = ["Left Pupil (mm)", "Right Pupil (mm)"];
  static DIRECTION_SCHEMA = "gaze-direction.v1";          // updated (2026-11-04)
  static SCREEN_SCHEMA = "screen-point-of-gaze.v1";       // updated (2026-11-04)
  static DIRECTION_NAMES = ["Gaze Yaw (°)", "Gaze Pitch (°)"];   // updated (2026-11-05): senza roll
  static SCREEN_NAMES = ["Screen X (px)", "Screen Y (px)"];                       // updated (2026-11-04)
  static #utf8 = new TextDecoder("utf-8");

  /**
   * @param {string|null} schema - updated (2026-11-04): schema dichiarato dal canale (null = nessuno)
   * @returns {Array<string>} i nomi delle colonne dei grafici per quello schema (Tobii senza schema)
   */

  static namesOf(schema) {
    if (schema === GazeDecoder.DIRECTION_SCHEMA) return GazeDecoder.DIRECTION_NAMES;
    if (schema === GazeDecoder.SCREEN_SCHEMA) return GazeDecoder.SCREEN_NAMES;
    return GazeDecoder.COLUMN_NAMES;
  }

  /**
   * @param {ArrayBuffer} payload - contenuto del frame, testo JSON
   * @param {string|null} [schema=null] - updated (2026-11-04): schema dichiarato dal canale; se manca vale quello del campione
   * @returns {{frame: {values: Array<number>, rows: number, columns: number, names: Array<string>}, gaze2d: Array<number>|null,
   *   direction: object|null, sample: object}|null}
   *   la riga dei grafici (NaN per un valore mancante), il punto guardato da disegnare sull'immagine (null se manca o
   *   non è sull'immagine), updated (2026-11-04): la direzione da disegnare (#directionOf, null se non è una
   *   direzione valida) e il campione letto, o null se il frame non è un JSON
   */

  static decode(payload, schema = null) {
    let sample;
    try { sample = JSON.parse(GazeDecoder.#utf8.decode(payload)); } catch { return null; }
    sample = sample && typeof sample === "object" ? sample : {};
    const data = sample.data && typeof sample.data === "object" ? sample.data : sample;
    const kind = schema ?? (typeof data.schema === "string" ? data.schema : null);
    if (kind === GazeDecoder.DIRECTION_SCHEMA) {
      return GazeDecoder.#result(data, [GazeDecoder.#degrees(data.yaw_rad), GazeDecoder.#degrees(data.pitch_rad)], GazeDecoder.DIRECTION_NAMES,
        GazeDecoder.#directionOf(data));
    }
    if (kind === GazeDecoder.SCREEN_SCHEMA) {
      const point = Array.isArray(data.screen_xy_px) ? data.screen_xy_px : [];
      return GazeDecoder.#result(data, [point[0], point[1]], GazeDecoder.SCREEN_NAMES, null);
    }
    const gaze = data.gaze2d;
    return {
      frame: {
        values: [GazeDecoder.#number(data.eyeleft?.pupildiameter), GazeDecoder.#number(data.eyeright?.pupildiameter)],
        rows: 1,
        columns: 2,
        names: GazeDecoder.COLUMN_NAMES
      },
      gaze2d: Array.isArray(gaze) && gaze.length === 2 && gaze.every(Number.isFinite) ? [gaze[0], gaze[1]] : null,
      direction: null,
      sample: data
    };
  }

  // updated (2026-11-04)
  /**
   * Il risultato di un campione con schema: una riga coi valori (NaN se lo stato non è "ok"), nessun punto guardato.
   * @param {object} data - il campione
   * @param {Array<*>} values - i valori delle colonne, come nel JSON
   * @param {Array<string>} names - i nomi delle colonne
   * @param {object|null} direction - la direzione da disegnare, o null
   * @returns {{frame: object, gaze2d: null, direction: object|null, sample: object}} il risultato di decode
   */

  static #result(data, values, names, direction) {
    const ok = data.status === "ok";
    return {
      frame: { values: values.map(value => ok ? GazeDecoder.#number(value) : NaN), rows: 1, columns: names.length, names },
      gaze2d: null,
      direction,
      sample: data
    };
  }

  // updated (2026-11-04)
  /**
   * Quello che serve per disegnare una direzione sopra il video (GazeDirectionDrawing).
   * @param {object} data - campione "gaze-direction.v1"
   * @returns {{yaw: number, pitch: number, box: Array<number>, eyes: Array<Array<number>>}|null} angoli in radianti,
   *   riquadro del volto [x1, y1, x2, y2] e centri delle iridi [[x, y], ...] in pixel dell'immagine; null se lo
   *   stato non è "ok" o mancano angoli o riquadro
   */

  static #directionOf(data) {
    const box = data.bbox_xyxy_px;
    if (data.status !== "ok" || !Number.isFinite(data.yaw_rad) || !Number.isFinite(data.pitch_rad)) return null;
    if (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite)) return null;
    const eyes = Array.isArray(data.eyes_px)
      ? data.eyes_px.filter(eye => Array.isArray(eye) && eye.length === 2 && eye.every(Number.isFinite))
      : [];
    return { yaw: data.yaw_rad, pitch: data.pitch_rad, box, eyes };
  }

  // updated (2026-11-05): il campione di GazeDirection ha solo i radianti
  /**
   * @param {*} radians - un angolo del JSON, in radianti
   * @returns {number} lo stesso angolo in gradi, o NaN se non è un numero finito (un buco nel grafico)
   */

  static #degrees(radians) {
    return Number.isFinite(radians) ? radians * 180 / Math.PI : NaN;
  }

  /**
   * @param {*} value - un campo del JSON
   * @returns {number} il valore se è un numero finito, altrimenti NaN (un buco nel grafico)
   */

  static #number(value) {
    return Number.isFinite(value) ? value : NaN;
  }
}
