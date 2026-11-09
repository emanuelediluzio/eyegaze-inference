/* iride — app/run/panels/node-panels.js
 *
 * NodePanels: i pannelli propri di alcuni tipi di nodo. La scheda di un nodo di quei tipi (Card) mostra il suo
 * pannello al posto dei mattoncini scelti dalle uscite: è un'interfaccia fatta per quel nodo (video con lo sguardo,
 * calibrazione), non un grafico per canale. Le card dedicate (solo) restano quelle di sempre.
 * Un pannello è una classe con:
 *   - static CHANNELS: i canali del nodo che mostra (per la filmstrip e Card.setVisible / shownChannels);
 *   - static PREFERRED_WIDTH: la larghezza della scheda che chiede;
 *   - constructor(options): { nodeName, nodeType, renderer (DataSocket), alive, inputSources (nome dell'ingresso ->
 *     { nodeName, chanName } della sorgente collegata nel Design), send (comando, argomenti) -> Promise della
 *     risposta di nodeCommand, sessionFor (chiave, crea, nodo) -> sessione della pipeline (PanelSessions),
 *     updated (2026-11-06): channelNames (nome di dyndefs.json -> nome vero del canale: nel lettore quello del Player)
 *     e replay (true nella scheda di un nodo della registrazione rivista: niente comandi al nodo) };
 *   - render(container), attach(), setVisible(visible), setPaused(paused), setAlive(alive), refreshTexts(),
 *     release(): come i mattoncini (CardSection), più il nodo vivo o morto.
 * Nuovo in iride (2026-11-04): da NodePanels.PANELS e dai ganci di Card e CardBoard di eyegaze-inference
 * (integrations/n2ucore, framework-hooks.json, del collega), che li aggiungeva a una versione di iride che li aveva.
 * Dipende da GazeDirectionPanel (gaze-direction-panel.js) e ScreenGazePanel (screen-gaze-panel.js).
 * Nessun DOM, nessun timer.
 */

import { GazeDirectionPanel } from "./gaze-direction-panel.js";
import { ScreenGazePanel } from "./screen-gaze-panel.js";

/**
 * Tipo di nodo -> pannello della sua scheda.
 */

export class NodePanels {
  static PANELS = new Map([
    ["N2uCore.dyn.eyegaze.GazeDirection", GazeDirectionPanel],
    ["N2uCore.dyn.screenpointofgaze.ScreenPointOfGaze", ScreenGazePanel]
  ]);

  /**
   * @param {string|undefined} nodeType - tipo del nodo, come in dyndefs.json (es. "N2uCore.dyn.eyegaze.GazeDirection")
   * @returns {function|null} la classe del suo pannello, o null se il tipo non ne ha uno
   */

  static panelFor(nodeType) {
    return NodePanels.PANELS.get(nodeType) ?? null;
  }
}
