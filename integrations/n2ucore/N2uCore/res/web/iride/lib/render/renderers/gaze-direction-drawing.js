/* iride — gaze-direction-drawing.js
 *
 * GazeDirectionDrawing: la direzione dello sguardo disegnata sopra il video, dentro il Worker dei dati (usata da
 * OverlayRenderer, modo "direction"). Da un campione di GazeDirection (GazeDecoder: riquadro del volto, centri
 * delle iridi, yaw e pitch in radianti, tutto in pixel dell'immagine) disegna tre livelli, ognuno acceso o spento
 * dal pannello del nodo (GazeDirectionPanel, control "gazeLayers"):
 *   - box: il riquadro del volto, con gli angoli arrotondati;
 *   - eyes: un pallino su ogni centro dell'iride;
 *   - arrows: da ogni occhio (dal centro del riquadro se gli occhi mancano) una freccia lunga ARROW_FACTOR volte la
 *     larghezza del riquadro, verso (sin(yaw), -sin(pitch)) sul piano dell'immagine, con la punta.
 * Da GazeDirectionPanel.#draw di eyegaze-inference (integrations/n2ucore, del collega), che disegnava in SVG sul
 * thread principale a ogni campione. Differenze: sul canvas del video, nel Worker, sincronizzato col fotogramma per
 * packID come gli altri punti (OverlayRenderer); colori fissi (il Worker non legge le variabili CSS).
 * Nessun DOM: vive nel Worker.
 */

/**
 * Disegno di riquadro del volto, occhi e frecce dello sguardo su un canvas.
 */

export class GazeDirectionDrawing {
  static LAYERS = { box: true, eyes: true, arrows: true };   // livelli accesi se il pannello non dice altro
  static FACE_COLOR = "#39ff14";       // riquadro e occhi (il verde dei punti dei tracker)
  static ARROW_FACTOR = 1.3;           // lunghezza della freccia / larghezza del riquadro
  static HEAD_ANGLE = 0.5;             // apertura della punta della freccia, in radianti
  static CORNER = 12;                  // raggio degli angoli del riquadro, in pixel di un'immagine larga BASE_WIDTH
  static BASE_WIDTH = 480;             // le misure valgono per un'immagine larga così; più larga, crescono in proporzione

  /**
   * Disegna una direzione sul canvas, sopra il fotogramma appena disegnato.
   * @param {OffscreenCanvasRenderingContext2D} ctx - contesto del canvas
   * @param {{yaw: number, pitch: number, box: Array<number>, eyes: Array<Array<number>>}} direction - la direzione
   *   (GazeDecoder), in pixel dell'immagine
   * @param {number} width - larghezza dell'immagine in pixel
   * @param {string} color - colore CSS delle frecce
   * @param {{box: boolean, eyes: boolean, arrows: boolean}} [layers] - livelli da disegnare (LAYERS se manca)
   */

  static draw(ctx, direction, width, color, layers = GazeDirectionDrawing.LAYERS) {
    const stroke = Math.max(2, width / GazeDirectionDrawing.BASE_WIDTH);
    const [x1, y1, x2, y2] = direction.box;
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    if (layers.box) GazeDirectionDrawing.#drawBox(ctx, x1, y1, x2, y2, stroke);
    if (layers.eyes) GazeDirectionDrawing.#drawEyes(ctx, direction.eyes, stroke);
    if (layers.arrows) {
      const starts = direction.eyes.length ? direction.eyes : [[(x1 + x2) / 2, (y1 + y2) / 2]];
      const length = (x2 - x1) * GazeDirectionDrawing.ARROW_FACTOR;
      const dx = Math.sin(direction.yaw) * length;
      const dy = -Math.sin(direction.pitch) * length;
      ctx.strokeStyle = color;
      ctx.lineWidth = stroke * 1.4;
      for (const [x, y] of starts) GazeDirectionDrawing.#drawArrow(ctx, x, y, x + dx, y + dy, stroke * 4);
    }
    ctx.restore();
  }

  /**
   * Il riquadro del volto, con gli angoli arrotondati.
   * @param {OffscreenCanvasRenderingContext2D} ctx - contesto del canvas
   * @param {number} x1 - sinistra
   * @param {number} y1 - alto
   * @param {number} x2 - destra
   * @param {number} y2 - basso
   * @param {number} stroke - spessore della linea
   */

  static #drawBox(ctx, x1, y1, x2, y2, stroke) {
    ctx.strokeStyle = GazeDirectionDrawing.FACE_COLOR;
    ctx.lineWidth = stroke;
    ctx.beginPath();
    ctx.roundRect(x1, y1, x2 - x1, y2 - y1, GazeDirectionDrawing.CORNER * stroke / 2);
    ctx.stroke();
  }

  /**
   * Un pallino su ogni centro dell'iride.
   * @param {OffscreenCanvasRenderingContext2D} ctx - contesto del canvas
   * @param {Array<Array<number>>} eyes - centri delle iridi, [x, y]
   * @param {number} stroke - spessore della linea (il raggio è il doppio)
   */

  static #drawEyes(ctx, eyes, stroke) {
    ctx.fillStyle = GazeDirectionDrawing.FACE_COLOR;
    for (const [x, y] of eyes) {
      ctx.beginPath();
      ctx.arc(x, y, stroke * 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  /**
   * Una freccia con la punta.
   * @param {OffscreenCanvasRenderingContext2D} ctx - contesto del canvas (colore e spessore già scelti)
   * @param {number} x - partenza x
   * @param {number} y - partenza y
   * @param {number} ex - arrivo x
   * @param {number} ey - arrivo y
   * @param {number} head - lunghezza dei due tratti della punta
   */

  static #drawArrow(ctx, x, y, ex, ey, head) {
    const angle = Math.atan2(ey - y, ex - x);
    const open = GazeDirectionDrawing.HEAD_ANGLE;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(ex, ey);
    ctx.moveTo(ex - head * Math.cos(angle - open), ey - head * Math.sin(angle - open));
    ctx.lineTo(ex, ey);
    ctx.lineTo(ex - head * Math.cos(angle + open), ey - head * Math.sin(angle + open));
    ctx.stroke();
  }
}
