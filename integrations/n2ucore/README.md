# Integrazione N2uCore / Iride

Questa cartella distribuisce i due nodi mantenendo i percorsi del framework:

```text
N2uCore/
  dyn/eyegaze.py                  GazeDirection(MlDevice), EyeGazeTicker(MlTicker)
  dyn/screenpointofgaze.py         ScreenPointOfGaze(SwDevice)
  inference/gazedino.py           Motore PyTorch e rilevamento del volto
  inference/gazeartifact.py       Contratto e export minimale dei pesi
  inference/screengaze.py         Proiezione e calibrazione
  res/dyndefs.json                Frammento del catalogo: solo i due nodi
  res/web/iride/schema-ui.json    Frammento delle impostazioni UI
  res/web/iride/lib/ui/forms/screen-gaze-panel.js
examples/eyegaze/                 Pipeline portabile, sketch e contratti
tests/dyn/                       Test dei nodi e della calibrazione
tests/web/                       Test del pannello e dei campi facoltativi
tools/verify_eyegaze_iride.py      Verifica del trasporto completo
install.py                       Installazione con anteprima e backup
```

Non è una copia completa di N2uCore e non va aggiunta al `PYTHONPATH` come
framework alternativo. Richiede il checkout N2uCore/Iride con `MlDevice`,
`MlTicker`, `WorkerFlowInput`, catalogo dinamico e pannelli Iride. I sorgenti
framework condivisi, gli altri modelli e i servizi dell'apparato restano nel
repository N2uCore.

## Installazione

Dalla radice di `eyegaze-inference`, su Windows, Linux o macOS:

```sh
python integrations/n2ucore/install.py /percorso/al/checkout/n2ucore
python integrations/n2ucore/install.py /percorso/al/checkout/n2ucore --apply
```

Su Windows usare ad esempio `C:/repos/n2ucore`. La prima chiamata mostra i file;
la seconda scrive dopo aver verificato tutti i punti di integrazione. Conserva
un backup dei file sostituiti in `.gaze-integration-backups/` nel checkout di
destinazione. Non avvia né arresta servizi e non modifica le dipendenze Python.
Eseguire a pipeline ferma e verificare il diff prima di riavviare i servizi.

**I due JSON sotto `res/` sono frammenti da unire**, non sostituzioni dei cataloghi
completi. L'installer li unisce per tipo di nodo, preservando le altre voci.
Registra il pannello di calibrazione e abilita la metainformazione `optional`
nel validatore Iride, così `hubDir` e `calibrationPath` possono rimanere vuoti.
Se il framework ha una struttura incompatibile, si ferma prima di scrivere.

Nel Python usato dal SysInit che esegue i nodi installare le dipendenze:

```sh
python -m pip install -r integrations/n2ucore/requirements.txt
```

Scegliere una coppia torch/torchvision compatibile con l'hardware di destinazione.
L'ambiente verificato sul Mac usa Python 3.11, MediaPipe 0.10.21 e NumPy 1.26.4;
la distribuzione in questa cartella non certifica ancora l'esecuzione su Windows
o Linux. Il secondo nodo richiede NumPy e il framework, senza usare PyTorch.

Il nodo usa l’export minimale `gazedino_inference.joblib` dalla
[release inference-v1](https://github.com/emanuelediluzio/eyegaze-inference/releases/tag/inference-v1).
Copiarlo in `N2uCore/res/models/gazedino/model.joblib`, oppure indicare un
percorso assoluto nel parametro Checkpoint. Il formato è un dizionario versionato
con pesi NumPy float32, architettura, preprocessing, convenzione delle uscite e
provenienza. Non contiene ottimizzatore, dataset o oggetti del modello Python.
Il nodo resta `MlDevice`/`MlTicker` e ricostruisce la rete PyTorch con caricamento
rigoroso dei pesi. Caricare solo artifact provenienti da fonti fidate.

Per creare lo stesso export dal checkpoint originale di training:

```sh
python tools/export_eyegaze.py --source /percorso/best_run3_3.24deg.pt --output /percorso/model.joblib
```

L’export verifica tutti i tensori dopo la scrittura e non sovrascrive file già
esistenti. Il checkpoint originale `.pt` resta supportato per compatibilità e
rimane quello usato dalla GUI desktop originale; la nuova variante non cambia
i pesi, la precisione numerica o la calibrazione.
Per operare offline servono anche le sorgenti DINOv2: predisporle e impostare
`hubDir` sul percorso locale. I pesi sono allegati alla release, non al Git dei sorgenti.

## Utilizzo e verifica

La sequenza è `CvCamera → GazeDirection → ScreenPointOfGaze`. Il primo nodo
esegue la rete su thread ML dedicato; il secondo converte gli angoli in pixel
dopo calibrazione. Iride comunica tramite uinode e Flow.

Vedere [configurazione, calibrazione e contratti](examples/eyegaze/README.md).
Dopo l'installazione, dalla radice del checkout **N2uCore**:

```sh
python -m pytest tests/dyn/test_eyegaze.py tests/dyn/test_screenpointofgaze.py -q
node --test tests/web/optional-gaze-fields.test.mjs tests/web/screen-gaze-panel.test.mjs
python tools/verify_eyegaze_iride.py --fixture /percorso/volto.png --checkpoint /percorso/model.joblib --hub-dir /percorso/dinov2
```

I test Python richiedono pytest e quelli di trasporto websocket-client; sono
elencati in `requirements-dev.txt`. Il test di trasporto avvia servizi isolati e
un video di prova; non misura l'accuratezza su soggetti reali. La precisione del
puntamento va validata su target nuovi dopo la calibrazione sull'apparato finale.

## Schede video e punto nella pagina

Gaze Direction usa una vista video del Worker dati Iride e una vista TEXT della
stessa DataSocket: la webcam non viene riaperta dal browser. Il motore ML emette
anche `image_size_px` e due centri iride (`eyes_px`) per allineare riquadro e
frecce. I dettagli JSON restano chiusi nella scheda.

Screen Point of Gaze mostra un’anteprima e il punto sopra la pagina Iride,
con `pointer-events: none`. Il punto sparisce su perdita del volto, campioni
fermi, disconnessione, pagina nascosta; chiudere o ridurre la scheda conserva il check e il pallino. Cambi
di dimensioni, posizione della finestra o densità del display richiedono una
nuova calibrazione; non vengono riscalate silenziosamente coordinate vecchie.
La modalità schermo intero va calibrata dopo l’ingresso nel nuovo viewport.

Carica/Salva calibrazione sono nelle Opzioni avanzate; il pulsante di cancellazione
non compare. Annullare una nuova raccolta conserva il fit precedente. Una nuova
calibrazione viene applicata soltanto dopo un fit riuscito.

I punti sono limitati alla pagina Iride, non al desktop o ad altre schede.
La precisione va verificata con la persona e il monitor reali.

### Frequenza e latenza

Il limite predefinito di inferenza è 15 FPS, senza coda: se il modello è più lento, il nodo continua a usare il frame più recente disponibile. I progetti già salvati mantengono il loro valore esplicito di `maxFps`: impostarlo a 15 per eliminare il precedente tetto conservativo di 4 FPS. Il valore è un massimo, non una frequenza garantita; dipende da dispositivo, modello e carico. Lo smussamento EMA resta invariato (alpha 0,25).

Su Mac mini M1, una prova ripetuta con immagine pubblica a 1920×1080 ha misurato circa 100–150 ms per analisi completa su MPS; ridurre a 1280×720 non ha dato un vantaggio. Questi dati descrivono il calcolo, non la latenza completa della webcam o della rete. Nessun cambio dei pesi, della precisione numerica o del nodo webcam. La scheda aggiorna le metriche esistenti e formatta i dettagli JSON solo quando sono aperti.

Il check **Mostra punto nella pagina** appartiene alla sessione della pipeline. Chiudere, nascondere o ridurre la card Screen Point of Gaze lascia attivi pallino e sottoscrizione se il check è selezionato. Riaprire la card ripristina la selezione; togliere il check nasconde il pallino. Fermare la pipeline elimina la sessione. Dati non recenti, volto assente, disconnessione o geometria non valida nascondono temporaneamente il punto senza cambiare la preferenza. Dopo un cambio di geometria occorre ricalibrare.

### Artifact minimale verificato

`gazedino_inference.joblib`: 324.174.954 byte (circa 309 MiB), contro
1.044.671.152 byte del checkpoint di training. SHA-256:
`d265ced7c0840c5f8f7d8880b2876d52b27313f337651e575f3975f8cd319cf4`.
Tutti i 183 tensori float32 sono identici. Il confronto su cinque crop di
un’immagine pubblica, con lo stesso preprocessing su MPS, ha dato differenza
massima delle predizioni di 0 radianti. Questo verifica la parità dell’export,
non una nuova misura dell’accuratezza. Il [report](examples/eyegaze/inference-export-verification.json) conserva dimensioni, hash ed esito.
