# Gaze Direction → Screen Point of Gaze

Due nodi separati, collegati tramite Flow:

```text
CvCamera.VideoStream → GazeDirection.VideoStream
GazeDirection.GazeDirection → ScreenPointOfGaze.GazeDirection
```

| Nodo | Classe | Ingresso | Uscita |
|---|---|---|---|
| Gaze Direction | `N2uCore.dyn.eyegaze.GazeDirection` (`MlDevice`) | Video RGB del volto | `GazeDirection`: yaw/pitch in radianti e gradi, vettore 3D unitario, stato e timestamp |
| Screen Point of Gaze | `N2uCore.dyn.screenpointofgaze.ScreenPointOfGaze` (`SwDevice`) | Direzione dello sguardo | `ScreenPoint`: X/Y in pixel, dimensioni dello schermo calibrato e stato |

Il modello gira **solo nel primo nodo**. Il secondo non carica PyTorch, MediaPipe o una telecamera: applica una trasformazione polinomiale leggera e gestisce la calibrazione. Puoi utilizzare Gaze Direction da solo. Screen Point of Gaze richiede l'uscita del primo nodo (contratto `gaze-direction.v1`).

Entrambi i canali sono TEXT con JSON UTF-8, già leggibili nelle viste testo Iride. Non usano il tipo GAZE Tobii: quel formato rappresenta pupille in mm e coordinate del video della scena, non gli angoli della webcam o i pixel dello schermo.

## Configurazione sul computer di destinazione

Installare l’integrazione seguendo il README nella radice di `integrations/n2ucore`.
Assegnare i tre nodi a un SysInit con le dipendenze ML disponibili. Importare
`gazedino.iride.json` e scegliere il SysInit corretto; quello di esempio è `SysInit`.
Il campo webcam ID `0` seleziona la prima camera del computer che esegue il nodo;
`1`, `2`, ecc. selezionano altre camere. L’ordine può cambiare tra macchine.
Windows usa DirectShow, Linux V4L2; verificare i permessi della camera.

Il checkpoint predefinito è `N2uCore/res/models/gazedino/best_run3_3.24deg.pt`
nel checkout N2uCore di destinazione. `hubDir` può rimanere vuoto: vengono usate
le sorgenti DINOv2 nella cache Torch, oppure scaricate da GitHub se assenti.
Per funzionare offline, predisporre anche quelle sorgenti e indicarne il percorso.
`calibrationPath` è facoltativo e appartiene solo a Screen Point of Gaze.
Senza calibrazione il nodo pubblica `uncalibrated`, con `screen_xy_px: null`.

Riavviare i servizi con la procedura dell’apparato dopo l’installazione e ricaricare
Iride senza cache. Non interrompere pipeline attive.

## Calibrazione guidata in Iride

Aprire la card del nodo **Screen Point of Gaze** a pipeline avviata:

- **Calibra 16 punti**: mostra una griglia 4×4 sull'area del browser Iride; seguire i punti tenendo ferma la testa. La raccolta dura 2,8 s per punto e scarta i primi 0,98 s. Il pannello avanza ogni 3 s.
- Il backend richiede almeno 3 predizioni fresche per punto, filtra gli outlier e calcola un fit cubico con almeno 10 punti e matrice di rango pieno. Il percorso guidato raccoglie tutti i 16 punti. Se il volto manca o il fit è degenere, il comando fallisce senza sostituire la calibrazione precedente.
- **Carica calibrazione / Salva calibrazione**: trasferiscono i coefficienti tramite `nodeCommand → uinode → Flow`, senza accedere direttamente al nodo o al filesystem della BlackBox.
- **Mostra punto nella pagina**: abilita il pallino nella pagina Iride dopo la calibrazione.
- **Prova a schermo intero**: estende Iride al monitor; calibrare dopo il cambio di viewport.
- **Esc / Annulla**, cambio di dimensioni della finestra, pagina nascosta o perdita del nodo annullano la procedura. La perdita del flusso direzionale annulla anche la raccolta sul backend.

I punti guidati sono espressi in **pixel CSS dell'area di Iride**. Le coordinate risultanti hanno lo stesso riferimento; non sono coordinate globali del desktop. Per applicare il risultato a un intero monitor occorre un viewport a schermo intero, stabile durante calibrazione e utilizzo, o una trasformazione esplicita di origine/scala. Non spostare la telecamera o cambiare seduta dopo aver calibrato.

La card Gaze Direction mostra video, riquadro del volto, centri degli occhi e frecce dello sguardo. La card Screen Point of Gaze gestisce calibrazione e pallino nell’area di Iride, anche a schermo intero. I dati tecnici restano disponibili in una sezione richiudibile e come stream JSON. Il pallino non può sovrapporsi alle altre applicazioni del desktop.

### Compatibilità con le calibrazioni esistenti

Sono accettati JSON con `cx`, `cy`, `screen_w`, `screen_h`: 6 coefficienti (quadratico) o 10 (cubico), nell'ordine `[1,yaw,pitch,yaw²,yaw·pitch,pitch²,yaw³,yaw²·pitch,yaw·pitch²,pitch³]`. Per trasferire una propria calibrazione pickle fidata della GUI:

```python
import pickle, json
with open('calibration.pkl', 'rb') as f:
    c = pickle.load(f)
with open('screen-calibration.json', 'w') as f:
    json.dump({k: c[k].tolist() if k in ('cx','cy') else c[k]
               for k in ('cx','cy','screen_w','screen_h')}, f)
```

Non usare pickle ricevuti da terzi. Una calibrazione desktop importata mantiene il riferimento/dimensioni originali: non applicarla all'area del browser senza verificarne la corrispondenza.

## Contratti dei dati

Direzione (`GazeDirection`, TEXT):

```json
{"schema":"gaze-direction.v1","status":"ok","yaw_rad":0.2,"pitch_rad":-0.1,"yaw_deg":11.459,"pitch_deg":-5.730,"direction_xyz":[0.197677,0.099833,-0.975170],"coordinate_space":"camera","bbox_xyxy_px":[100,80,300,320],"confidence":null,"source_ts_us":1792762000000000,"dropped_frames":12}
```

Vettore: `[sin(yaw)·cos(pitch), -sin(pitch), -cos(yaw)·cos(pitch)]`, stessa convenzione del modello addestrato. Non inventiamo una probabilità: è una regressione, non un classificatore.

Punto sullo schermo (`ScreenPoint`, TEXT):

```json
{"schema":"screen-point-of-gaze.v1","coordinate_space":"screen_pixels","source_ts_us":1792762000000000,"status":"ok","screen_xy_px":[520,290],"screen_size_px":[1000,600]}
```

Entrambi mantengono il timestamp della cattura. Il primo fa inferenza su thread ML con un solo frame in elaborazione; scarta i frame eccedenti e il frame che ha avviato il caricamento dei pesi. Il secondo rifiuta timestamp ripetuti, fuori ordine o troppo vecchi (`maxSampleAgeSecs`, default 2 s), valori non finiti e formati incompatibili.

Gli stati `no_face`, `model_ready`, `model_unavailable`, `error`, `input_disconnected`, `stale_input`, `invalid_input` o `uncalibrated` non contengono coordinate utilizzabili. Non ripetere l'ultimo punto in un consumer quando il flusso si interrompe: anche il consumer deve verificare il timestamp.

## Verifiche

```bash
python -m pytest tests/dyn/test_eyegaze.py tests/dyn/test_screenpointofgaze.py tests/dyn/wrk/test_mldevice.py -q
node --test tests/web/screen-gaze-panel.test.mjs
python tools/verify_eyegaze_iride.py --fixture /path/to/face.png \
  --checkpoint /path/to/best_run3_3.24deg.pt --hub-dir /path/to/dinov2
```

Il test d'integrazione avvia servizi isolati e un video di prova; controlla la catena `Iride WS → uinode → sysinit → telecamera → direzione → proiezione → Iride WS`, i timestamp, lo stato non calibrato, caricamento/cancellazione della calibrazione e la corrispondenza numerica dei pixel agli angoli originali. Arresta la pipeline e attende IDLE. Non usa la webcam né modifica le pipeline esistenti. La calibrazione numerica del test è sintetica: verifica il software, non la precisione di un soggetto reale.

Le prove locali precedenti hanno verificato anche il trasporto attraverso uinode
e sysinit, con video di prova e calibrazione sintetica. Ripetere il test di
integrazione sull’apparato di destinazione; non equivale a validare l’accuratezza
su persone reali.

## Validazione e revisione ancora necessarie

Il 3,24° del checkpoint è l'errore di validazione **per frame** dichiarato nel repository di training, non una misura indipendente su soggetti nuovi o in pixel sul monitor corrente. Il residuo del fit di calibrazione non è un test su punti nuovi. Restano la verifica del puntamento con persone reali e target non usati nel fit, e la misura della latenza sull'apparato finale.

Con Daniele: revisione dei contratti e integrazione nel runtime aziendale; verificare il keep-alive WebSocket durante l'avvio delle connessioni Flow, che ha richiesto una riconnessione in alcune prove precedenti. Con Sergio: obiettivo applicativo, hardware, inquadratura e requisiti di accuratezza. Nessun messaggio è stato inviato loro.

I repository `eye-gaze` (training) e `eyegaze-inference` (demo desktop e integrazione N2uCore) usano lo stesso modello. I nuovi due nodi dividono invece **due fasi della medesima inferenza**. `StressGazeInference`, Tobii e BioNormWear riguardano acquisizione strumentale o stima fisiologica: non sono questa coppia di nodi.
