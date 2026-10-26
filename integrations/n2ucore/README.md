# Eye Gaze — integrazione in Next2U Core / Iride

Questa guida è per chi deve integrare i due nodi in un checkout esistente di **N2uCore** e usarli da **Iride**. Il [README principale](../../README.md) descrive anche l’applicazione Python desktop; qui trovi il pacchetto per il framework.

## Subito: joblib e sorgenti dei nodi

| Cosa serve | Dove si trova | Destinazione / funzione |
|---|---|---|
| **Modello minimale `gazedino_inference.joblib`** | **[Download diretto GitHub](https://github.com/emanuelediluzio/eyegaze-inference/releases/download/inference-v1/gazedino_inference.joblib)** · **[Download diretto Gitea](http://192.168.70.200:3000/attachments/8a538e7d-0727-4b6b-81d3-77c27c6deb06)** · [dettagli release](http://192.168.70.200:3000/ediluzio/eyegaze-inference/releases/tag/inference-v1) | Copiare e rinominare in **`N2uCore/res/models/gazedino/model.joblib`** nel framework di destinazione |
| **Nodo Gaze Direction** | [N2uCore/dyn/eyegaze.py](N2uCore/dyn/eyegaze.py) | `GazeDirection(MlDevice)` e `EyeGazeTicker(MlTicker)`: dal video a yaw/pitch e vettore 3D |
| **Nodo Screen Point of Gaze** | [N2uCore/dyn/screenpointofgaze.py](N2uCore/dyn/screenpointofgaze.py) | `ScreenPointOfGaze(SwDevice)`: dalla direzione al punto sullo schermo, dopo calibrazione |
| **Installazione completa** | [install.py](install.py) · [framework-hooks.json](framework-hooks.json) | Installa motori, catalogo, pannelli Iride e collegamenti necessari nel framework |
| **Pipeline di esempio** | [examples/eyegaze/gazedino.iride.json](examples/eyegaze/gazedino.iride.json) | Importare in Iride e scegliere il SysInit e la webcam del dispositivo di destinazione |

**Il joblib è un allegato della release, non è incluso nel clone Git.** Pesa **324.174.954 byte** (circa 324 MB / 309 MiB), circa il **69% in meno** del checkpoint di training da 1.044.671.152 byte.

SHA-256 del joblib della release `inference-v1`:

```text
d265ced7c0840c5f8f7d8880b2876d52b27313f337651e575f3975f8cd319cf4
```

Per un’integrazione funzionante prendere **l’intero contenuto di questa cartella**, non soltanto i due file `dyn/*.py`: i nodi dipendono dai motori di inferenza, dalle voci di catalogo e dai pannelli Iride elencati sotto. L’installer applica questi componenti mantenendo la gerarchia del framework.

## 1. Installare nel framework esistente

Prerequisiti: checkout N2uCore con `MlDevice`, `MlTicker`, `WorkerFlowInput`, catalogo dinamico e Iride. Il Python da usare è quello configurato nel **SysInit che eseguirà i nodi**.

Dalla radice del repository `eyegaze-inference`, a pipeline ferma:

```sh
python -m pip install -r integrations/n2ucore/requirements.txt
python integrations/n2ucore/install.py "/percorso/al/checkout/n2ucore"
python integrations/n2ucore/install.py "/percorso/al/checkout/n2ucore" --apply
```

Su Windows il percorso può essere, per esempio, `C:/repos/n2ucore`. La chiamata senza `--apply` mostra l’anteprima. Quella con `--apply` verifica prima tutti i punti di integrazione, poi scrive e conserva i file precedenti in `.gaze-integration-backups/`.

L’installer unisce le voci dei due nodi nei cataloghi esistenti, preservando le altre voci. I JSON distribuiti sotto `N2uCore/res/` sono **frammenti da unire**, non cataloghi completi da sostituire. Gli adattamenti condivisi di Card/CardBoard, pannelli e campi facoltativi sono dichiarati in `framework-hooks.json` e `install.py`. Se il framework è incompatibile, il preflight termina senza scrivere. Ripetere l’installazione sullo stesso codice non applica modifiche ulteriori.

L’installer non avvia servizi, non installa pesi e non modifica la configurazione della webcam. Questa cartella è un overlay del framework: non va usata come framework alternativo aggiungendola al `PYTHONPATH`.

## 2. Posizionare il joblib

Scaricare il modello dai link iniziali. Nel checkout **N2uCore di destinazione** creare la cartella e collocare il file così:

```text
N2uCore/res/models/gazedino/model.joblib
```

Il parametro **Checkpoint path** di Gaze Direction deve essere:

```text
models/gazedino/model.joblib
```

Questo è il valore predefinito dei nuovi nodi. I progetti Iride già salvati conservano i propri valori: se puntano ancora al `.pt` originale, aggiornare questo parametro. È possibile usare anche un percorso assoluto, riferito al computer che esegue il nodo.

Servono inoltre le **sorgenti DINOv2**. Con **Local DINOv2 source directory (offline)** vuoto il motore usa la cache Torch `facebookresearch_dinov2_main`, oppure scarica le sorgenti se assenti. Per un apparato offline predisporre la cache o indicare esplicitamente la cartella locale contenente `hubconf.py`. Il caricamento usa `pretrained=False`: tutti i pesi del backbone sono già nel joblib.

## 3. Avviare in Iride

La pipeline è:

```text
WebCam.VideoStream → GazeDirection.VideoStream
GazeDirection.GazeDirection → ScreenPointOfGaze.GazeDirection
```

Importare la [pipeline portabile](examples/eyegaze/gazedino.iride.json), scegliere il SysInit corretto e verificare l’ID della webcam **sul computer che esegue WebCam**. L’ID dipende dall’ordine dei dispositivi su quella macchina.

| Impostazione | Valore / significato |
|---|---|
| Gaze Direction — Checkpoint path | `models/gazedino/model.joblib` |
| Compute device | `auto`: CUDA, MPS o CPU secondo disponibilità |
| Maximum inference FPS | `15`: limite massimo, non frequenza garantita |
| Local DINOv2 source directory (offline) | Facoltativo; cache o sorgenti locali come descritto sopra |
| Screen Point of Gaze — Screen calibration JSON | Facoltativo; senza calibrazione il nodo emette `uncalibrated` |

Dopo l’installazione riavviare i servizi con la procedura prevista dall’apparato, a pipeline ferma, e ricaricare Iride senza cache. Avviare quindi la pipeline.

Aprire Gaze Direction per vedere video, volto, occhi e frecce; i dettagli tecnici sono richiudibili. Aprire Screen Point of Gaze e usare **Calibra 16 punti**. Per usare tutto il monitor entrare prima a schermo intero e calibrare in quella modalità.

Il check **Mostra punto nella pagina** appartiene alla sessione della pipeline: chiudere, nascondere o ridurre la card lascia attivo il pallino; riaprendola il check resta selezionato. Togliere il check lo nasconde. Lo stop della pipeline elimina la sessione. Volto assente, dati non recenti, disconnessione e pagina nascosta nascondono temporaneamente il punto senza cambiare la preferenza. Cambiare dimensioni, posizione della finestra o densità del display richiede una nuova calibrazione.

**Carica/Salva calibrazione** sono nelle Opzioni avanzate. Il relativo JSON contiene coefficienti e dimensioni dell’area calibrata; è distinto dal joblib dei pesi. Annullare una nuova calibrazione conserva quella precedente. Il pallino viene disegnato nella pagina Iride e non può restare visibile sopra altre applicazioni o dopo la chiusura del browser.

Per contratti dei canali, dettagli della calibrazione e compatibilità dei JSON vedere [examples/eyegaze/README.md](examples/eyegaze/README.md).

## File da integrare e gerarchia

I percorsi seguenti sono relativi a questa cartella e corrispondono ai percorsi nel checkout N2uCore:

```text
N2uCore/
  dyn/eyegaze.py                         Nodo ML e ticker dedicato
  dyn/screenpointofgaze.py               Nodo di proiezione e calibrazione
  inference/gazedino.py                  Rete PyTorch e rilevamento del volto
  inference/gazeartifact.py              Contratto, caricamento ed export joblib
  inference/screengaze.py                Fit e proiezione della calibrazione
  res/dyndefs.json                       Frammento del catalogo dei due nodi
  res/web/iride/schema-ui.json           Frammento dei parametri Iride
  res/web/iride/lib/ui/forms/
    gaze-direction-panel.js             Card video e direzione
    screen-gaze-panel.js                 Card e controlli di calibrazione
    screen-gaze-session.js               Pallino indipendente dalla card
    panel-sessions.js                    Stato dei pannelli per la pipeline
    gaze-ui.js                          Utilità e viste DataSocket condivise
  res/web/iride/css/app/gaze-panels.css   Stili delle card e del pallino
examples/eyegaze/                        Pipeline, sketch e documentazione
tests/dyn/                             Test Python
tests/web/                             Test dell’interfaccia
tools/export_eyegaze.py                 Export del checkpoint originale
tools/verify_eyegaze_iride.py            Test completo del trasporto
```

Gaze Direction mantiene il pattern **`MlDevice` / `MlTicker`**: decodifica, rilevamento del volto e PyTorch girano sul thread ML, con un solo frame in elaborazione e scarto dei frame superati. Il formato joblib non cambia la rete in un modello tabellare. Screen Point of Gaze usa **`SwDevice`** e NumPy; non carica il modello neurale. Iride comunica solo con **uinode**, tramite i canali WebSocket del framework; tra i nodi si usa Flow.

## Contenuto del joblib ed export

Il file è un dizionario con schema `n2u.gazedino.inference.v1`:

| Campo | Contenuto |
|---|---|
| `format` | Versione del contratto dell’artifact |
| `architecture` | GazeDINO, backbone DINOv2 ViT-B/14 e dimensioni della testa |
| `preprocessing` | RGB, resize 224×224 e normalizzazione ImageNet |
| `output` | Ordine yaw/pitch, in radianti |
| `state_dict` | 183 tensori conservati come array NumPy float32 |
| `metadata` | SHA-256 del checkpoint sorgente ed errore di validazione riportato dal checkpoint |

Non contiene ottimizzatore, dataset, cronologia di training o un’istanza serializzata della classe del modello. Il loader ricostruisce la rete PyTorch e carica i pesi con `strict=True`. Usare artifact provenienti da fonti fidate.

Per creare una nuova copia minimale, dalla radice del checkout **N2uCore dopo l’installazione**:

```sh
python tools/export_eyegaze.py --source /percorso/best_run3_3.24deg.pt --output /percorso/model.joblib
```

L’export rilegge il file generato e confronta tutti i pesi con gli originali; non sovrascrive una destinazione esistente. Il `.pt` originale resta supportato dal nodo per compatibilità ed è il formato usato dalla GUI desktop.

## Verifiche e limiti misurati

- **183 tensori float32 identici** al checkpoint originale; nessuna quantizzazione o modifica dei pesi.
- **Predizioni identiche su cinque crop** di un’immagine pubblica, con differenza massima di 0 radianti su MPS, sia localmente sia sul Mac mini. [Report di parità e checksum](examples/eyegaze/inference-export-verification.json).
- **Test completo sul Mac mini** con video di prova: WebSocket → uinode → sysinit → WebCam → Gaze Direction/MlTicker → Screen Point of Gaze → WebSocket, verifica della proiezione e arresto a IDLE. [Report del test](examples/eyegaze/minimal-node-integration-test.json).
- Ambiente verificato: Python 3.11, MediaPipe 0.10.21, NumPy 1.26.4 e MPS. Scegliere build torch/torchvision adatte all’hardware. Windows e Linux non sono ancora certificati con prove equivalenti.

La riduzione del file elimina lo stato di addestramento; il modello rimane DINOv2 ViT-B/14. Non è una misura di minore costo computazionale. Una prova sul Mac mini M1 ha misurato circa 100–150 ms per analisi completa; `maxFps=15` è un tetto. La parità dell’export non costituisce una nuova validazione dell’accuratezza: il puntamento va verificato sulla persona e sul monitor reali, su target non usati per calibrare.

Per ripetere i controlli, dalla radice del checkout **N2uCore** e nel suo ambiente Python:

```sh
python -m pip install -r /percorso/eyegaze-inference/integrations/n2ucore/requirements-dev.txt
python -m pytest tests/dyn/test_gazeartifact.py tests/dyn/test_eyegaze.py tests/dyn/test_screenpointofgaze.py -q
node --test tests/web/gaze-direction-panel.test.mjs tests/web/screen-gaze-panel.test.mjs tests/web/optional-gaze-fields.test.mjs
python tools/verify_eyegaze_iride.py --fixture /percorso/volto.png --checkpoint /percorso/model.joblib --hub-dir /percorso/dinov2
```

Il test completo usa un’immagine pubblica ripetuta in un video di prova; non richiede una webcam fisica. Le dipendenze di test sono separate da quelle di esecuzione.
