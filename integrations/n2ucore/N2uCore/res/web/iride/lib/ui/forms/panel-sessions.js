/** Pipeline-owned panel state. Closing a card only releases its observer. */
export class PanelSessions {
  #sessions = new Map();
  get(key, create) {
    if (!this.#sessions.has(key)) this.#sessions.set(key, create());
    return this.#sessions.get(key);
  }
  refreshAlive(isAlive) {
    for (const [nodeName, session] of this.#sessions) session.setAlive?.(isAlive(nodeName));
  }
  clear() {
    for (const session of this.#sessions.values()) session.release();
    this.#sessions.clear();
  }
}
