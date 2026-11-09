/* iride — app/run/panels/browser-viewport.js
 *
 * BrowserViewport: le misure della finestra del browser che contano per la calibrazione dello sguardo sullo
 * schermo (ScreenPointOfGaze): larghezza e altezza della pagina in pixel CSS, posizione della finestra sullo schermo
 * e densità dei pixel. Se una cambia, la calibrazione non vale più (il punto guardato è nelle coordinate della
 * pagina calibrata).
 * Da viewport e sameViewport di gaze-ui.js di eyegaze-inference (integrations/n2ucore, del collega).
 * Nessun timer.
 */

/**
 * Misure della finestra per la calibrazione.
 */

export class BrowserViewport {
  static KEYS = ["width", "height", "x", "y", "dpr"];

  /**
   * @returns {{width: number, height: number, x: number, y: number, dpr: number}} le misure di adesso
   */

  static current() {
    return {
      width: window.innerWidth,
      height: window.innerHeight,
      x: window.screenX ?? 0,
      y: window.screenY ?? 0,
      dpr: window.devicePixelRatio ?? 1
    };
  }

  /**
   * @param {object|null} a - misure (current)
   * @param {object|null} b - altre misure
   * @returns {boolean} true se ci sono tutte e due e sono uguali
   */

  static same(a, b) {
    return !!a && !!b && BrowserViewport.KEYS.every(key => a[key] === b[key]);
  }
}
