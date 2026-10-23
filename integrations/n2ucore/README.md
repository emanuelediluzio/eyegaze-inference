# Integrazione N2uCore / Iride

Questa cartella distribuisce i due nodi mantenendo i percorsi del framework:

```text
N2uCore/
  dyn/eyegaze.py                  GazeDirection(MlDevice), EyeGazeTicker(MlTicker)
  dyn/screenpointofgaze.py         ScreenPointOfGaze(SwDevice)
  inference/gazedino.py           Motore PyTorch e rilevamento del volto
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

Scaricare `best_run3_3.24deg.pt` dalla release di
[eye-gaze su GitHub](https://github.com/emanuelediluzio/eye-gaze/releases/tag/v1.0-weights)
oppure dalla [release interna](http://192.168.70.200:3000/ediluzio/eye-gaze/releases/tag/v1.0-weights).
Copiarlo nel checkout di destinazione in
`N2uCore/res/models/gazedino/best_run3_3.24deg.pt`, oppure indicare nel nodo
un percorso assoluto sul computer che lo esegue.

SHA-256: `b68131063b6475f85e1ca152113b45d96b413a25c5e2dd517276b2f53cb796a0`.
I pesi non sono inclusi nei sorgenti Git. Per operare offline servono anche le
sorgenti DINOv2: predisporle e impostare `hubDir` sul percorso locale.

## Utilizzo e verifica

La sequenza è `CvCamera → GazeDirection → ScreenPointOfGaze`. Il primo nodo
esegue la rete su thread ML dedicato; il secondo converte gli angoli in pixel
dopo calibrazione. Iride comunica tramite uinode e Flow.

Vedere [configurazione, calibrazione e contratti](examples/eyegaze/README.md).
Dopo l'installazione, dalla radice del checkout **N2uCore**:

```sh
python -m pytest tests/dyn/test_eyegaze.py tests/dyn/test_screenpointofgaze.py -q
node --test tests/web/optional-gaze-fields.test.mjs tests/web/screen-gaze-panel.test.mjs
python tools/verify_eyegaze_iride.py --fixture /percorso/volto.png --checkpoint /percorso/best_run3_3.24deg.pt --hub-dir /percorso/dinov2
```

I test Python richiedono pytest e quelli di trasporto websocket-client; sono
elencati in `requirements-dev.txt`. Il test di trasporto avvia servizi isolati e
un video di prova; non misura l'accuratezza su soggetti reali. La precisione del
puntamento va validata su target nuovi dopo la calibrazione sull'apparato finale.
