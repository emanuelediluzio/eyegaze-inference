// JSON gaze samples: Tobii pupils/scene points, gaze angles or calibrated screen pixels.

export class GazeDecoder {
  static COLUMN_NAMES = ["Left Pupil (mm)", "Right Pupil (mm)"];
  static #utf8 = new TextDecoder("utf-8");

  /**
   * @param {ArrayBuffer} payload - contenuto del frame, testo JSON
   * @returns {{frame: {values: Array<number>, rows: number, columns: number, names: Array<string>}, gaze2d: Array<number>|null}|null}
   *   la riga delle pupille (NaN per un valore mancante) e il punto guardato (null se manca), o
   *   null se il frame non è un JSON
   */

  static decode(payload) {
    let sample;
    try { sample = JSON.parse(GazeDecoder.#utf8.decode(payload)); } catch { return null; }
    sample = sample && typeof sample === "object" ? sample : {};
    const data = sample.data && typeof sample.data === "object" ? sample.data : sample;
    if (data.schema === "gaze-direction.v1") {
      return GazeDecoder.#frame(data, ["yaw_deg", "pitch_deg", "roll_deg"],
        ["Gaze yaw (deg)", "Gaze pitch (deg)", "Head roll (deg)"]);
    }
    if (data.schema === "screen-point-of-gaze.v1" && !("gaze2d" in data)) {
      return GazeDecoder.#frame(data, ["x", "y"], ["Screen X (px)", "Screen Y (px)"],
        { x: data.screen_xy_px?.[0], y: data.screen_xy_px?.[1] });
    }
    const gaze = data.gaze2d;
    return {
      frame: {
        values: [GazeDecoder.#number(data.eyeleft?.pupildiameter), GazeDecoder.#number(data.eyeright?.pupildiameter)],
        rows: 1,
        columns: 2,
        names: GazeDecoder.COLUMN_NAMES
      },
      gaze2d: Array.isArray(gaze) && gaze.length === 2 && gaze.every(Number.isFinite) ? [gaze[0], gaze[1]] : null
    };
  }

  static #frame(sample, fields, names, values = sample) {
    return {
      frame: {
        values: fields.map(field => sample.status === "ok" ? GazeDecoder.#number(values[field]) : NaN),
        rows: 1, columns: fields.length, names
      },
      gaze2d: null
    };
  }

  /**
   * @param {*} value - un campo del JSON
   * @returns {number} il valore se è un numero, altrimenti NaN (un buco nel grafico)
   */

  static #number(value) {
    return Number.isFinite(value) ? value : NaN;
  }
}
