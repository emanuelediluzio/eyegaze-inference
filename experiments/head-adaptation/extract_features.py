"""GazeDINO features of the GazeGene subjects held out of training, for the head adaptation experiment.

    python extract_features.py DATASET_DIR CHECKPOINT OUT.npz

DATASET_DIR has subjectN/imgs/cameraC/subjectN_IIII.jpg and subjectN/labels/gaze_label_cameraC.pkl (GazeGene face
crops). For every frame: the DINOv2 CLS token (768), the input of the last head layer (128), GazeDINO's angles and
the true angles (yaw, pitch in radians, pitch positive up, as in training), with subject and camera.
"""
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.gaze_model import GazeDINO  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def angles(vec):
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    return np.stack([np.arctan2(x, -z), np.arctan2(-y, np.sqrt(x**2 + z**2))], axis=-1)


def main(dataDir, checkpoint, out):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = GazeDINO(dropout=0.0)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])
    model.eval().to(device)
    trunk = model.head[:-1]          # LayerNorm ... GELU: the 128 inputs of the last Linear

    rows = {k: [] for k in ("cls", "h128", "pred", "true", "subject", "camera", "frame")}
    for subj in sorted(Path(dataDir).glob("subject*")):
        for cam in range(9):
            label = subj / "labels" / f"gaze_label_camera{cam}.pkl"
            if not label.exists():
                continue
            true = angles(np.asarray(pickle.load(open(label, "rb"))["gaze_C"]))
            paths = [(i, subj / "imgs" / f"camera{cam}" / f"{subj.name}_{i:04d}.jpg") for i in range(min(200, len(true)))]
            paths = [(i, p) for i, p in paths if p.exists()]
            for start in range(0, len(paths), 32):
                batch = paths[start:start + 32]
                imgs = [cv2.cvtColor(cv2.resize(cv2.imread(str(p)), (224, 224)), cv2.COLOR_BGR2RGB) for _, p in batch]
                x = (torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0 - MEAN) / STD
                with torch.no_grad():
                    cls = model.backbone(x.to(device))
                    h = trunk(cls)
                    pred = model.head[-1](h)
                rows["cls"].append(cls.cpu().numpy()); rows["h128"].append(h.cpu().numpy())
                rows["pred"].append(pred.cpu().numpy())
                rows["true"].append(true[[i for i, _ in batch]])
                rows["subject"] += [subj.name] * len(batch); rows["camera"] += [cam] * len(batch)
                rows["frame"] += [i for i, _ in batch]
            print(subj.name, cam, flush=True)
    np.savez_compressed(out, **{k: (np.concatenate(v) if k in ("cls", "h128", "pred", "true") else np.array(v))
                                for k, v in rows.items()},
                        last_w=model.head[-1].weight.detach().cpu().numpy(), last_b=model.head[-1].bias.detach().cpu().numpy())


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
