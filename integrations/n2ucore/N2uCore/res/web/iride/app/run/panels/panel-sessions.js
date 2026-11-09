/* iride — app/run/panels/panel-sessions.js
 *
 * PanelSessions: lo stato dei pannelli dei nodi che vive con la pipeline e non con la scheda. Chiudere una scheda
 * rilascia solo il suo pannello; la sessione del nodo resta (es. ScreenGazeSession: la calibrazione mostrata e il
 * punto sulla pagina, che restano accesi anche a scheda chiusa). Le sessioni si dimenticano tutte quando le schede
 * si svuotano (CardBoard.clear: pipeline ferma o nuova).
 * updated (2026-11-06): la chiave può essere diversa dal nome del nodo di cui la sessione segue il "vivo" (nel lettore
 * più nodi registrati arrivano tutti dal Player: chiave = nodo + canale, nodo = il Player).
 * Una sessione può avere setAlive(boolean) (il suo nodo vivo o morto) e deve avere release().
 * Da PanelSessions di eyegaze-inference (integrations/n2ucore, del collega). Nessun DOM, nessun timer.
 */

/**
 * Sessioni dei pannelli, una per chiave (il nome del nodo).
 */

export class PanelSessions {
  #sessions = new Map();   // chiave -> { session, nodeName }

  /**
   * @param {string} key - chiave della sessione (di solito il nome del nodo)
   * @param {function(): object} create - crea la sessione se non c'è
   * @param {string} [nodeName=key] - updated (2026-11-06): il nodo di cui la sessione segue il "vivo"
   * @returns {object} la sessione, quella che c'era o quella appena creata
   */

  get(key, create, nodeName = key) {
    if (!this.#sessions.has(key)) this.#sessions.set(key, { session: create(), nodeName });
    return this.#sessions.get(key).session;
  }

  /**
   * Dice a ogni sessione se il suo nodo è vivo.
   * @param {function(string): boolean} isAlive - nome del nodo -> true se vivo
   */

  refreshAlive(isAlive) {
    for (const { session, nodeName } of this.#sessions.values()) session.setAlive?.(isAlive(nodeName));
  }

  /**
   * Rilascia e dimentica tutte le sessioni (pipeline ferma o nuova).
   */

  clear() {
    for (const { session } of this.#sessions.values()) session.release();
    this.#sessions.clear();
  }
}
