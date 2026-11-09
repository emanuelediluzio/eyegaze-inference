/* iride — app/run/panels/panel-dom.js
 *
 * PanelDom: i piccoli aiuti DOM dei pannelli dei nodi (NodePanels). Crea un elemento con classe e testo, e un
 * testo può ricordare la sua chiave di Texts (data-text, con i segnaposto in data-text-params), così al cambio di
 * lingua refreshTexts lo riscrive senza ricreare il pannello.
 * Nuovo in iride (2026-11-04): in eyegaze-inference (integrations/n2ucore, del collega) era la funzione el di
 * gaze-ui.js; qui una classe, e i testi si rinnovano con la lingua.
 * Dipende da Texts (lib/core/texts.js).
 */

import { Texts } from "../../../lib/core/texts.js";

/**
 * Elementi e testi dei pannelli dei nodi.
 */

export class PanelDom {
  /**
   * @param {string} tag - nome del tag
   * @param {string} [className=""] - classi CSS
   * @param {string} [text=""] - testo già pronto (non tradotto)
   * @returns {HTMLElement} l'elemento nuovo
   */

  static element(tag, className = "", text = "") {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
  }

  /**
   * Un elemento il cui testo è una chiave di Texts: lo scrive nella lingua attuale e se la ricorda.
   * @param {string} tag - nome del tag
   * @param {string} className - classi CSS
   * @param {string} key - chiave di Texts
   * @param {object} [params={}] - valori dei segnaposto
   * @returns {HTMLElement} l'elemento nuovo
   */

  static translated(tag, className, key, params = {}) {
    const node = PanelDom.element(tag, className);
    PanelDom.setText(node, key, params);
    return node;
  }

  /**
   * Scrive in un elemento il testo di una chiave di Texts e se la ricorda (refreshTexts).
   * @param {HTMLElement} node - l'elemento
   * @param {string} key - chiave di Texts
   * @param {object} [params={}] - valori dei segnaposto
   */

  static setText(node, key, params = {}) {
    node.dataset.text = key;
    node.dataset.textParams = JSON.stringify(params);
    const text = Texts.get(key, params);
    if (node.textContent !== text) node.textContent = text;
  }

  /**
   * Riscrive nella lingua attuale tutti i testi ricordati sotto un elemento (lui compreso).
   * @param {HTMLElement} root - l'elemento
   */

  static refreshTexts(root) {
    const nodes = root.matches?.("[data-text]") ? [root, ...root.querySelectorAll("[data-text]")] : root.querySelectorAll("[data-text]");
    for (const node of nodes) PanelDom.setText(node, node.dataset.text, JSON.parse(node.dataset.textParams || "{}"));
  }

  /**
   * Una casella da spuntare con la sua scritta, nello stile dei campi delle schede (label.field).
   * @param {string} key - chiave di Texts della scritta
   * @param {boolean} checked - spuntata all'inizio
   * @param {function(boolean): void} onChange - chiamata a ogni cambio, con lo stato nuovo
   * @returns {{row: HTMLElement, input: HTMLInputElement}} la riga e la casella
   */

  static checkbox(key, checked, onChange) {
    const row = PanelDom.element("label", "field");
    const input = PanelDom.element("input");
    input.type = "checkbox";
    input.checked = checked;
    input.addEventListener("change", () => onChange(input.checked));
    row.append(input, PanelDom.translated("span", "", key));
    return { row, input };
  }

  /**
   * Un bottone dei comandi delle schede (command-run).
   * @param {string} key - chiave di Texts della scritta
   * @param {object} params - valori dei segnaposto
   * @param {function(): void} onClick - chiamata al clic
   * @returns {HTMLButtonElement} il bottone
   */

  static button(key, params, onClick) {
    const button = PanelDom.translated("button", "command-run", key, params);
    button.type = "button";
    button.addEventListener("click", onClick);
    return button;
  }
}
