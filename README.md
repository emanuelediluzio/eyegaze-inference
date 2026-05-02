# eyegaze-inference — GazeDINO Real-Time Inference

Inferenza in tempo reale con **GazeDINO** (DINOv2 ViT-B/14 + MLP head).  
Nessun codice di training, nessuna dipendenza dal dataset. Plug-in del checkpoint e via.

Best model: **3.24° di errore angolare medio**.

---

## Architettura Modello

```
Input: (B, 3, 224, 224)  ← crop volto, normalizzato ImageNet
          │
  DINOv2 ViT-B/14  →  CLS token  →  [768 dim]
  (86M params, pre-trainato self-supervised su 142M immagini)
          │
  LayerNorm(768)
  → Linear(768 → 512) → GELU → Dropout(0.30)
  → Linear(512 → 128) → GELU → Dropout(0.15)
  → Linear(128 → 2)
          │
  Output: (yaw, pitch) in radianti
```

Il modello è trainato con **angular loss** (errore angolare medio tra vettori gaze 3D) su GazeGene (56 soggetti, 9 camere, ~90K campioni). Repo training: [eye-gaze](https://github.com/emanuelediluzio/eye-gaze).

---

## Quick Start

```bash
# 1. Clona e installa
git clone https://github.com/emanuelediluzio/eyegaze-inference.git
cd eyegaze-inference
pip install -r requirements.txt

# 2. Scarica il checkpoint migliore da GitHub Releases
# https://github.com/emanuelediluzio/eye-gaze/releases/tag/v1.0-weights
mkdir checkpoints
# Rinomina il file scaricato in best.pt
mv best_run3_3.24deg.pt checkpoints/best.pt

# 3. Avvia la GUI (webcam live)
python gui.py

# 4. Oppure su un video
python gui.py --video path/to/video.mp4
```

> **macOS:** se la camera non si apre, vai in Preferenze di Sistema → Privacy & Sicurezza → Fotocamera e autorizza il Terminale.

---

## Pipeline di Inferenza

```
Frame webcam (BGR)
      │
      ▼
MediaPipe FaceMesh (468 landmark + iris)
      │
      ├─ Crop volto → 224×224 → GazeDINO → (yaw, pitch) raw
      │         └─ EMA smoothing (alpha=0.3)
      │
      ├─ Iris landmarks → pupillometry (iris_ratio = iris_px / eye_width_px)
      │
      └─ Cheekbone landmarks → distanza (pinhole camera model, cm)
                │
                ▼
        Senza calibrazione: freccia gaze sul video
        Con calibrazione:   punto (x, y) sullo schermo
```

---

## GUI

Interfaccia CustomTkinter dark-mode con:

| Sezione | Contenuto |
|---------|-----------|
| **Sidebar — MODEL** | Val error del checkpoint caricato |
| **Sidebar — OVERLAYS** | Toggle: gaze arrow, distance, pupillometry |
| **Sidebar — CALIBRATION** | Calibrate / Load / Clear |
| **Sidebar — VIDEO** | Open file, Play/Pause, seek slider |
| **Video principale** | Feed live con overlay attivi |
| **Metriche** | Yaw/Pitch°, distanza cm (verde/giallo/rosso), screen XY se calibrato |

**Scorciatoie tastiera:**
| Tasto | Azione |
|-------|--------|
| `Q` / `ESC` | Esci |
| `C` | Avvia calibrazione |
| `Space` | Play/Pause video |

---

## Calibrazione 9 Punti

Mappa `(yaw, pitch)` → `(screen_x, screen_y)` tramite fit polinomiale di grado 2.

1. Finestra a schermo intero con 9 punti in griglia 3×3
2. Segui ogni punto con gli occhi (2.5 s per punto), testa ferma
3. Fit con PolynomialFeatures(degree=2) + Ridge regression
4. Salvata in `calibration.pkl` per le sessioni successive

Richiede almeno 6 punti validi (skip automatico dei punti con pochi sample).

---

## CLI Options

```bash
python gui.py [--model PATH] [--camera INT] [--video PATH] [--no-loop]
```

| Flag | Default | Descrizione |
|------|---------|-------------|
| `--model` | `checkpoints/best.pt` | Path al checkpoint `.pt` |
| `--camera` | 0 | Indice webcam |
| `--video` | — | Path a file video |
| `--no-loop` | off | Non ripetere il video |

---

## Struttura Progetto

```
eyegaze-inference/
  models/
    gaze_model.py       # GazeDINO: DINOv2 + MLP head (768→512→128→2)
  infer_gaze.py         # Core: load_model, predict, FaceMeshAnalyzer, EMA, draw helpers
  gui.py                # Entry point — GUI CustomTkinter
  calibration.py        # Calibrazione 9 punti, GazeCalibrator
  checkpoints/
    best.pt             # Checkpoint da inserire manualmente
  requirements.txt
```

---

## Metriche Live

| Metrica | Come calcolata |
|---------|----------------|
| **Yaw / Pitch** | Output diretto del modello, convertito in gradi |
| **Distanza (cm)** | `d = (face_real_width × focal_length) / face_px_width`, zona comfort 40-90 cm |
| **Iris ratio** | `iris_diameter_px / eye_width_px` — proxy dilatazione pupillare |
| **Screen XY** | Solo dopo calibrazione: polinomiale (yaw,pitch) → pixel schermo |

---

## Requisiti

- Python 3.10+
- Apple Silicon (MPS), CUDA, o CPU
- Webcam (per inferenza live)

```
torch torchvision
mediapipe
opencv-python
customtkinter
scikit-learn
numpy
```

---

## Troubleshooting

**Camera non autorizzata (macOS):** Sistema → Privacy → Fotocamera → abilita Terminale.

**Size mismatch al caricamento:** Assicurati di usare il checkpoint da [Releases v1.0-weights](https://github.com/emanuelediluzio/eye-gaze/releases/tag/v1.0-weights) — l'architettura è 768→512→128→2.

**Calibrazione imprecisa:** Illuminazione uniforme, entrambi gli occhi visibili, testa ferma.

**Warning xFormers:** Cosmético, non influenza le performance.

---

## Riferimenti

- Training repo: [eye-gaze](https://github.com/emanuelediluzio/eye-gaze)
- [DINOv2](https://github.com/facebookresearch/dinov2) — Meta AI, Apache 2.0
- [MediaPipe](https://github.com/google/mediapipe) — Google, Apache 2.0
- [GazeGene Dataset](https://huggingface.co/datasets/vigil1917/GazeGene)

## License

MIT
