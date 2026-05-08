# EyeGaze — Pipeline Completa

Documento tecnico di riferimento. Descrive il sistema **GazeDINO** dal modello al training al dataset, con schemi visuali e dati essenziali.

---

## Indice

- [Introduzione](#introduzione)
- [1. Architettura del modello](#1-architettura-del-modello)
- [2. Dataset — GazeGene](#2-dataset--gazegene)
- [3. Pipeline di training](#3-pipeline-di-training)
- [4. Risultati](#4-risultati)

---

## Introduzione

Il sistema stima la direzione dello sguardo `(yaw, pitch)` da una singola immagine RGB del volto. È composto da due moduli:

- un **backbone DINOv2 ViT-B/14** (~86 M parametri) pre-addestrato self-supervised da Meta AI su 142 M immagini, scaricato già pronto e usato come estrattore di feature
- una **head MLP** (~0.5 M parametri) che noi abbiamo addestrato per tradurre le feature DINOv2 in `(yaw, pitch)`

Il training è stato fatto sul dataset **GazeGene** (50 soggetti, 9 camere, ~90 K campioni) in due fasi: warm-up della head con backbone congelato, poi full fine-tune con learning rate molto basso. Il modello migliore raggiunge **3.24° di errore angolare medio** sul validation set.

Questo repo contiene **solo l'inferenza**: si carica il checkpoint, si fa la GUI con webcam o video. Il training avviene nel repo separato [eye-gaze](https://github.com/emanuelediluzio/eye-gaze).

---

## 1. Architettura del modello

```
                      ┌─────────────────────────────────────┐
                      │     INPUT IMAGE                     │
                      │     face crop RGB                   │
                      │     normalized ImageNet stats       │
                      │     mean = [0.485, 0.456, 0.406]    │
                      │     std  = [0.229, 0.224, 0.225]    │
                      └─────────────────┬───────────────────┘
                                        │
                                shape: (B, 3, 224, 224)
                                        │
                                        ▼
   ╔════════════════════════════════════════════════════════════════════╗
   ║                                                                    ║
   ║                  BACKBONE — DINOv2 ViT-B/14                        ║
   ║                                                                    ║
   ║   ┌────────────────────────────────────────────────────────────┐   ║
   ║   │  Patch Embedding 14×14                                     │   ║
   ║   │  → (16×16) = 256 patch token + 1 CLS token  +  pos.embed   │   ║
   ║   └────────────────────────────────────────────────────────────┘   ║
   ║                                │                                   ║
   ║                                ▼                                   ║
   ║   ┌────────────────────────────────────────────────────────────┐   ║
   ║   │  12 × Transformer Block                                    │   ║
   ║   │     ├─ MHSA (12 heads, dim 64)                             │   ║
   ║   │     ├─ LayerNorm + Residual                                │   ║
   ║   │     ├─ MLP (3072 hidden)                                   │   ║
   ║   │     └─ LayerNorm + Residual                                │   ║
   ║   └────────────────────────────────────────────────────────────┘   ║
   ║                                │                                   ║
   ║          (estrai SOLO il CLS token, scarta i 256 patch token)      ║
   ║                                │                                   ║
   ║                                ▼                                   ║
   ║                       CLS token: (B, 768)                          ║
   ║                                                                    ║
   ║              86 M parametri · pre-trained self-supervised          ║
   ║              (Meta AI, su 142 M immagini, 0 label)                 ║
   ╚════════════════════════════════════════════════════════════════════╝
                                        │
                                  (B, 768)
                                        │
                                        ▼
   ╔════════════════════════════════════════════════════════════════════╗
   ║                                                                    ║
   ║                    REGRESSION HEAD (MLP)                           ║
   ║                                                                    ║
   ║       ┌──────────────────────────────────┐                         ║
   ║       │  LayerNorm(768)                  │                         ║
   ║       └─────────────────┬────────────────┘                         ║
   ║                         ▼                                          ║
   ║       ┌──────────────────────────────────┐                         ║
   ║       │  Linear(768 → 512)               │                         ║
   ║       │  GELU                            │                         ║
   ║       │  Dropout(p = 0.30)               │                         ║
   ║       └─────────────────┬────────────────┘                         ║
   ║                         ▼                                          ║
   ║       ┌──────────────────────────────────┐                         ║
   ║       │  Linear(512 → 128)               │                         ║
   ║       │  GELU                            │                         ║
   ║       │  Dropout(p = 0.15)               │                         ║
   ║       └─────────────────┬────────────────┘                         ║
   ║                         ▼                                          ║
   ║       ┌──────────────────────────────────┐                         ║
   ║       │  Linear(128 → 2)                 │                         ║
   ║       └─────────────────┬────────────────┘                         ║
   ║                                                                    ║
   ║          ~ 0.5 M parametri · init Xavier uniform · bias = 0        ║
   ╚════════════════════════════════════════════════════════════════════╝
                                        │
                                  (B, 2)
                                        │
                                        ▼
                           ┌────────────────────────┐
                           │     OUTPUT             │
                           │  (yaw, pitch) radianti │
                           └────────────────────────┘

                       TOTALE: ~ 87 M parametri
```

### Conteggio parametri

| Componente | Parametri |
|------------|----------:|
| Backbone DINOv2 ViT-B/14 | 86 580 480 |
| Head MLP (768→512→128→2) | 461 186 |
| **Totale** | **87 041 666** |

### Output

- `yaw` ∈ ℝ — angolo orizzontale (positivo = sguardo a destra)
- `pitch` ∈ ℝ — angolo verticale (positivo = sguardo verso l'alto)
- entrambi in **radianti** (conversione in gradi solo per display)

---

## 2. Dataset — GazeGene

**Fonte:** [HuggingFace `vigil1917/GazeGene`](https://huggingface.co/datasets/vigil1917/GazeGene), ~27 GB. Pubblicato sotto licenza permissiva.

### Composizione

| | |
|---|---|
| Soggetti | 56 |
| Camere per soggetto | 9 (angolazioni diverse) |
| Combinazioni totali | 56 × 9 = 504 |
| Frame per camera | fino a 200 (configurabile) |
| Run 3 (best model) | 50 soggetti × 9 × 200 = **90 000 campioni** |
| Split train / val | 80 / 20 frame-level (seed 42) |
| Range yaw osservato | −134° ↔ +124° |
| Range pitch osservato | −87° ↔ +77° |

### Struttura su disco

```
gazegene_data/GazeGene_FaceCrops/
├── subject1/
│   ├── imgs/
│   │   ├── camera0/   subject1_0000.jpg … subject1_0199.jpg
│   │   ├── camera1/   …
│   │   └── camera8/
│   └── labels/
│       ├── gaze_label_camera0.pkl    ← gaze_C: (N, 3) vettori 3D unitari
│       ├── gaze_label_camera1.pkl
│       └── …
├── subject2/
└── …
└── subject56/
```

### Conversione label 3D → angoli

Il pickle contiene `gaze_C` come array `(N, 3)` di vettori direzione 3D. Conversione (`data/gazegene_loader.py`):

```python
# convenzione: x=destra, y=basso, z=avanti
yaw   = arctan2(x, -z)
pitch = arctan2(-y, sqrt(x² + z²))
```

### Augmentation (solo train)

| Trasformazione | Probabilità | Note |
|----------------|------------:|------|
| Horizontal flip + yaw mirror | 0.5 | `label[0] = -label[0]` (critico) |
| ColorJitter | 0.7 | brightness 0.3, contrast 0.3, saturation 0.2, hue 0.05 |
| GaussianBlur | 0.3 | kernel 3, sigma 0.1–1.5 |
| Gaussian noise | 0.3 | σ = 0.02 |

Le immagini vengono caricate **on-demand** da disco con `cv2.imread` (lazy loading) per non saturare la RAM.

---

## 3. Pipeline di training

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              PREPARAZIONE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  GazeGene HuggingFace (~27 GB)                                          │
│         │                                                               │
│         ▼                                                               │
│  data/download_gazegene.py                                              │
│         │  curl + zip merge + extract                                   │
│         ▼                                                               │
│  gazegene_data/GazeGene_FaceCrops/                                      │
│    └── subjectN/imgs/cameraK/*.jpg                                      │
│    └── subjectN/labels/gaze_label_cameraK.pkl                           │
│         │                                                               │
│         ▼                                                               │
│  data/gazegene_loader.py · load_gazegene()                              │
│    ├─ legge i .pkl (vettori 3D)                                         │
│    ├─ converti 3D → (yaw, pitch) rad: arctan2(...)                      │
│    ├─ split frame-level 80/20 (seed=42)                                 │
│    └─ ritorna (paths_train, paths_val, y_train, y_val)                  │
│         │                                                               │
│         ▼                                                               │
│  data/gaze_dataset.py · LazyGazeDataset                                 │
│    ├─ legge immagini on-demand (cv2.imread)                             │
│    ├─ resize 224 + normalize ImageNet                                   │
│    └─ augmentation (solo train): hflip+yaw mirror, jitter, blur, noise  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                              DataLoader (batch_size=32, num_workers=0)
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                               MODEL INIT                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   GazeDINO(backbone='dinov2_vitb14', dropout=0.3, freeze_backbone=True) │
│         │                                                               │
│         ▼                                                               │
│   torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')            │
│         │                                                               │
│         ▼                                                               │
│   Head MLP (init Xavier, bias=0)                                        │
│         │                                                               │
│         ▼                                                               │
│   .to(device)  · device = MPS / CUDA / CPU (auto)                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
╔═════════════════════════════════════════════════════════════════════════╗
║                  PHASE 1 · HEAD WARM-UP · 10 epoche                     ║
╠═════════════════════════════════════════════════════════════════════════╣
║                                                                         ║
║  backbone:  ❄ FROZEN  (pesi DINOv2 intatti)                             ║
║  head:      🟢 trainable  (~ 0.5 M params)                              ║
║                                                                         ║
║  optimizer  AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)        ║
║  scheduler  CosineAnnealingLR(T_max=10, eta_min=1e-5)                   ║
║                                                                         ║
║          LR:   1e-3  ─────────cosine─────────►  1e-5                    ║
║                                                                         ║
║  per ogni epoca:                                                        ║
║    ┌─────────────────────────────────────────────────────────────┐      ║
║    │ for batch in train_loader:                                  │      ║
║    │   pred = model(imgs)                                        │      ║
║    │   loss = angular_loss(pred, labels)                         │      ║
║    │   loss.backward()                                           │      ║
║    │   clip_grad_norm_(1.0)                                      │      ║
║    │   optimizer.step()                                          │      ║
║    │ # validazione → val_err in gradi                            │      ║
║    │ if val_err < best: save_checkpoint(best.pt)                 │      ║
║    └─────────────────────────────────────────────────────────────┘      ║
║                                                                         ║
║  output Run 1 fine fase 1: ~62.5° (head ancora "fredda" ma stabile)     ║
╚═════════════════════════════════════════════════════════════════════════╝
                                    │
                              .unfreeze_backbone()
                                    │
                                    ▼
╔═════════════════════════════════════════════════════════════════════════╗
║              PHASE 2 · FULL FINE-TUNE · 50 epoche                       ║
╠═════════════════════════════════════════════════════════════════════════╣
║                                                                         ║
║  backbone:  🟢 trainable  (86 M params, gradienti attivi)               ║
║  head:      🟢 trainable                                                ║
║                                                                         ║
║  optimizer  AdamW(model.parameters(), lr=5e-6, weight_decay=1e-4)       ║
║  scheduler  CosineAnnealingLR(T_max=50, eta_min=5e-8)                   ║
║                                                                         ║
║          LR:   5e-6  ─────────cosine─────────►  5e-8                    ║
║                                                                         ║
║          ↑                                                              ║
║          200× più piccolo della Phase 1 — backbone già "ottimo",        ║
║          basta un nudge per non distruggere DINOv2                      ║
║                                                                         ║
║  early stopping: patience=12 epoche                                     ║
║                                                                         ║
║  Run 1 (10 subj): best val 4.44° @ ep 47                                ║
║  Run 2 (50 subj): interrotta a 43/50                                    ║
║  Run 3 (50 subj, --resume da Run 2): best val 3.24° @ ep 49 ✅          ║
║                                                                         ║
╚═════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              LOSS & METRICA                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  angular_loss(pred, target)                                             │
│                                                                         │
│      converti (yaw, pitch) → vettore 3D unitario                        │
│      v = [sin(yaw)·cos(pitch), -sin(pitch), -cos(yaw)·cos(pitch)]       │
│                                                                         │
│      cos_sim = (v_pred · v_true).clamp(-1+ε, 1-ε)                       │
│      loss    = acos(cos_sim).mean()      ← in radianti                  │
│                                                                         │
│      val_err_deg = rad2deg(loss)         ← per logging                  │
│                                                                         │
│  Geometricamente corretta: misura l'angolo reale fra i vettori 3D       │
│  sulla sfera unitaria, non l'errore euclideo nel piano (yaw, pitch).    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                                OUTPUT                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  checkpoints/gaze_YYYYMMDD_HHMMSS/best.pt                               │
│    {                                                                    │
│      "epoch":       49,                                                 │
│      "val_err_deg": 3.24,                                               │
│      "model":       state_dict,                                         │
│      "optimizer":   state_dict                                          │
│    }                                                                    │
│                                                                         │
│  logs/gaze_YYYYMMDD_HHMMSS/training_log.csv                             │
│    phase, epoch, loss, train_err, val_loss, val_err, lr                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Cheat sheet iperparametri

| | Phase 1 | Phase 2 |
|---|--------:|--------:|
| Epoche | 10 | 50 |
| Backbone | ❄ frozen | 🟢 trainable |
| Trainable params | 0.5 M | 87 M |
| LR iniziale | `1e-3` | `5e-6` |
| LR finale | `1e-5` | `5e-8` |
| Optimizer | AdamW | AdamW |
| Weight decay | `1e-4` | `1e-4` |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Gradient clip | `max_norm=1.0` | `max_norm=1.0` |
| Early stopping | patience 12 | patience 12 |
| Batch size | 32 | 32 |
| Loss | angular_loss | angular_loss |
| Augmentation | sì (solo train) | sì (solo train) |

### Hardware

Tutti i run sono stati eseguiti su **Apple Silicon (MPS backend)**. Tempo full Run 3: ~5 giorni su M1.

---

## 4. Risultati

| Run | Data | Soggetti | Phase | Best Val Error | Note |
|-----|------|---------:|-------|---------------:|------|
| Run 1 | 2026-03-14 | 10 | 1 + 2 | **4.44°** @ ep 47 | from scratch |
| Run 2 | 2026-04-11 | 50 | 2 only | ~3.94° | interrotto a 43/50 |
| **Run 3** | **2026-04-16** | **50** | 2 only (resume da Run 2) | **3.24°** @ ep 49 | **best** |

**Improvement Run 1 → Run 3:** −1.20° (driver: 5× più dati + resume iterativo).

I checkpoint sono pubblicati su [Releases v1.0-weights](https://github.com/emanuelediluzio/eye-gaze/releases/tag/v1.0-weights):
- `best_run3_3.24deg.pt` — best (Run 3)
- `best_run1_4.44deg.pt` — baseline (Run 1)

---

## Riferimenti

- DINOv2 — Oquab et al., 2023 — [paper](https://arxiv.org/abs/2304.07193) · [repo](https://github.com/facebookresearch/dinov2) · Apache 2.0
- GazeGene Dataset — [HuggingFace](https://huggingface.co/datasets/vigil1917/GazeGene)
- Vision Transformers (ViT) — Dosovitskiy et al., 2020 — [paper](https://arxiv.org/abs/2010.11929)
- Training repo: [eye-gaze](https://github.com/emanuelediluzio/eye-gaze)
