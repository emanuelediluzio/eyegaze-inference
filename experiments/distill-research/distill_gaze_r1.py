"""
Gaze Estimation — distillation R1 (RULE-R1.md): distillation v3 (../distill-v3/distill_gaze_v3.py) with many more
people, from shards of export_crops_video.py.

What changes from v3:
  - real crops come from v3's folders (--real_dir: MPIIFaceGaze p00-p09, the training session), RT-GENE
    distillation shards (--real_shards) and the AVCAffe people of --avc_train (--avc_dir shards); each real batch
    is --avc_share AVCAffe, drawn so that every AVCAffe person weighs the same, the rest from the other sources;
  - the checkpoint is chosen on AVCAffe --avc_val only: per person the mean gap to the teacher over at most
    --val_per_person crops evenly spaced in the shard, then the mean over people (RULE-R1.md); MPIIFaceGaze p10-p14,
    the RT-GENE test people and Joel are never looked at;
  - AVCAffe people with fewer than --min_crops crops are left out of both groups;
  - shards are read once into memory (the FAT32 disk is slow on small reads); their loaders run in the main process;
  - checkpoints and logs go to --out_dir (on the Mac mini's HARD DISK, with the data).

Runs from the eye-gaze training code (next to train_gaze.py):
    python distill_gaze_r1.py --teacher ../teacher/best_run3_3.24deg.pt --init V3_BEST.pt --dataset_dir GAZEGENE \
        --real_dir crops/mpii_p00-p09,crops/free_train --real_shards rtgene/s000.npz,... \
        --avc_dir AVCAFFE_SHARDS --avc_train info/train.txt --avc_val info/val.txt --out_dir OUT
    python distill_gaze_r1.py ... --resume OUT/checkpoints/distill_r1_<run>/last.pt   # same arguments
"""
from __future__ import annotations
import argparse
import copy
import csv
import itertools
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_gaze import get_device, save_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir",  default="gazegene_data/GazeGene_FaceCrops")
    p.add_argument("--teacher",      required=True, help="GazeDINO checkpoint (.pt) of the teacher")
    p.add_argument("--teacher_backbone", default="dinov2_vitb14")
    p.add_argument("--student",      default="dinov2_vits14", choices=["dinov2_vits14", "dinov2_vitb14"])
    p.add_argument("--init",         help="student checkpoint to start from (v1 best.pt); without it, v1's phase 1 runs")
    p.add_argument("--real_dir",     required=True, help="real face crops for training (export_crops.py), comma separated folders")
    p.add_argument("--real_shards",  default="", help="more training crops: export_crops_video.py shards, comma separated")
    p.add_argument("--avc_dir",      required=True, help="AVCAffe shards (export_crops_video.py)")
    p.add_argument("--avc_train",    required=True, help="AVCAffe people for training, one per line (info/train.txt)")
    p.add_argument("--avc_val",      required=True, help="AVCAffe people for checkpoint selection (info/val.txt)")
    p.add_argument("--avc_share",    type=float, default=0.5, help="share of AVCAffe crops in each real batch")
    p.add_argument("--min_crops",    type=int, default=300, help="AVCAffe people with fewer crops are left out")
    p.add_argument("--val_per_person", type=int, default=500)
    p.add_argument("--out_dir",      default=".", help="checkpoints/ and logs/ go here")
    p.add_argument("--resume",       help="last.pt of an interrupted run: continues it in the same folder")
    p.add_argument("--max_subjects", type=int, default=50)
    p.add_argument("--max_frames",   type=int, default=200)
    p.add_argument("--batch_size",   type=int, default=32, help="GazeGene images per batch")
    p.add_argument("--real_batch",   type=int, default=32, help="real crops per batch")
    p.add_argument("--epochs_p1",    type=int, default=10)
    p.add_argument("--epochs",       type=int, default=100)
    p.add_argument("--warmup",       type=int, default=3)
    p.add_argument("--lr_p1",        type=float, default=1e-3)
    p.add_argument("--lr",           type=float, default=2e-5)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--patience",     type=int,   default=25)
    p.add_argument("--alpha",        type=float, default=0.5, help="weight of the teacher's answers on GazeGene")
    p.add_argument("--beta",         type=float, default=1.0, help="weight of the teacher's CLS token")
    p.add_argument("--patch_beta",   type=float, default=1.0, help="weight of the teacher's patch tokens")
    p.add_argument("--real_weight",  type=float, default=1.0, help="weight of the real-crop losses")
    p.add_argument("--ema",          type=float, default=0.999)
    p.add_argument("--max_batches",  type=int,   default=None, help="cap per epoch (smoke test)")
    p.add_argument("--workers",      type=int,   default=4)
    return p.parse_args()


def load_teacher(path: str, backbone: str, device: torch.device) -> nn.Module:
    from models.gaze_model import GazeDINO
    teacher = GazeDINO(backbone=backbone)
    teacher.load_state_dict(torch.load(path, map_location="cpu")["model"])
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    teacher = teacher.eval().to(device, dtype)
    teacher.requires_grad_(False)
    return teacher


def tokens(backbone: nn.Module, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """CLS token and patch tokens after DINOv2's last norm (the CLS token is what GazeDINO's head reads)."""
    out = backbone.forward_features(imgs)
    return out["x_norm_clstoken"], out["x_norm_patchtokens"]


@torch.no_grad()
def teach(teacher: nn.Module, imgs: torch.Tensor):
    """Teacher yaw/pitch, CLS and patch tokens, back in fp32."""
    cls, patches = tokens(teacher.backbone, imgs.to(next(teacher.parameters()).dtype))
    return teacher.head(cls).float(), cls.float(), patches.float()


class Jpegs:
    """Shard crops kept as JPEG bytes in memory, decoded to RGB uint8 on access: the image list of GazeDataset."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import cv2
        return cv2.cvtColor(cv2.imdecode(self.items[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def shard(path, picks=None) -> list[np.ndarray]:
    z = np.load(path)
    jpeg, offsets = z["jpeg"], z["offsets"]
    picks = range(len(offsets) - 1) if picks is None else picks
    return [jpeg[offsets[i]:offsets[i + 1]].copy() for i in picks]


def people(path: str) -> list[str]:
    return [l.strip() for l in open(path) if l.strip()]


def crops(folders: str) -> list[str]:
    # not the hidden "._" files macOS adds when copying to another disk: they are no images
    return sorted(str(p) for f in folders.split(",") for p in Path(f).glob("*.jpg") if not p.name.startswith("."))


def main():
    args   = parse_args()
    device = get_device()
    torch.backends.cuda.matmul.allow_tf32 = True     # A6000: TF32 matmuls, as fast as fp16 for the student
    torch.backends.cudnn.allow_tf32 = True
    print(f"[distillR1] Device: {device}")

    from data.gazegene_loader import load_gazegene
    from data.gaze_dataset    import LazyGazeDataset
    from models.gaze_model    import GazeDINO, angular_loss

    from data.gaze_dataset    import GazeDataset
    from torch.utils.data     import ConcatDataset, WeightedRandomSampler
    import r1stats

    p_train, p_val, y_train, y_val = load_gazegene(
        args.dataset_dir, max_subjects=args.max_subjects, max_frames_per_camera=args.max_frames, lazy=True)
    loader = lambda paths, labels, augment, bs, shuffle: DataLoader(
        LazyGazeDataset(paths, labels, augment=augment), batch_size=bs, shuffle=shuffle,
        num_workers=args.workers, drop_last=augment, persistent_workers=args.workers > 0)
    zeros = lambda n: np.zeros((n, 2), np.float32)   # real crops have no label: never used (the flip negates them)
    train_loader    = loader(p_train, y_train, True, args.batch_size, True)
    val_loader      = loader(p_val, y_val, False, args.batch_size, False)

    def avc(person):
        path = Path(args.avc_dir) / f"{person}.npz"
        return path if path.exists() and len(np.load(path)["offsets"]) - 1 >= args.min_crops else None
    left_out = [p for p in people(args.avc_train) + people(args.avc_val) if avc(p) is None]
    print(f"[distillR1] AVCAffe left out (missing or under {args.min_crops} crops): {' '.join(left_out) or 'none'}")
    avc_items, avc_owner = [], []
    for person in people(args.avc_train):
        if avc(person):
            items = shard(avc(person))
            avc_items += items
            avc_owner += [person] * len(items)
    counts = {p: avc_owner.count(p) for p in set(avc_owner)}
    avc_batch = round(args.real_batch * args.avc_share)
    avc_loader = DataLoader(GazeDataset(Jpegs(avc_items), zeros(len(avc_items)), augment=True), batch_size=avc_batch,
                            sampler=WeightedRandomSampler([1 / counts[p] for p in avc_owner], len(avc_items)),
                            num_workers=0, drop_last=True)
    others = crops(args.real_dir)
    shard_items = [i for f in args.real_shards.split(",") if f for i in shard(f)]
    other_loader = DataLoader(ConcatDataset([LazyGazeDataset(others, zeros(len(others)), augment=True),
                                             GazeDataset(Jpegs(shard_items), zeros(len(shard_items)), augment=True)]),
                              batch_size=args.real_batch - avc_batch, shuffle=True, num_workers=0, drop_last=True)
    val_people = [p for p in people(args.avc_val) if avc(p)]
    real_val_loaders = []
    for person in val_people:
        n = len(np.load(avc(person))["offsets"]) - 1
        items = shard(avc(person), r1stats.evenlySpaced(n, args.val_per_person))
        real_val_loaders.append(DataLoader(GazeDataset(Jpegs(items), zeros(len(items)), augment=False),
                                           batch_size=args.batch_size, shuffle=False, num_workers=0))
    print(f"[distillR1] GazeGene train {len(p_train)}, val {len(p_val)}; real train: AVCAffe {len(avc_items)} "
          f"({len(counts)} people, {avc_batch} a batch), others {len(others)} + {len(shard_items)}; "
          f"selection: {len(val_people)} AVCAffe people")

    teacher = load_teacher(args.teacher, args.teacher_backbone, device)
    student = GazeDINO(backbone=args.student, dropout=args.dropout).to(device)
    if args.init:
        student.load_state_dict(torch.load(args.init, map_location="cpu")["model"])
        print(f"[distillR1] Student from {args.init}")
    s_dim, t_dim = student.backbone.embed_dim, teacher.backbone.embed_dim
    project       = nn.Linear(s_dim, t_dim).to(device)     # used only for the loss, not saved with the student
    project_patch = nn.Linear(s_dim, t_dim).to(device)
    ema = copy.deepcopy(student).eval().requires_grad_(False)

    state = torch.load(args.resume, map_location="cpu") if args.resume else None
    run_id   = state["run_id"] if state else datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(args.out_dir) / "checkpoints" / f"distill_r1_{run_id}"
    logs_dir = Path(args.out_dir) / "logs"        / f"distill_r1_{run_id}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = ckpt_dir / "best.pt", ckpt_dir / "last.pt"
    with open(logs_dir / ("args_resume.json" if state else "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    csv_file   = open(logs_dir / "training_log.csv", "a" if state else "w", newline="")
    csv_writer = csv.writer(csv_file)
    if not state:
        csv_writer.writerow(["phase", "epoch", "loss", "label_err_deg", "teacher_gap_deg", "real_gap_deg", "cls_cos",
                             "patch_cos", "val_err_deg", "real_val_gap_deg", "lr", "seconds"])
    avc_batches, other_batches = itertools.cycle(avc_loader), itertools.cycle(other_loader)

    def losses(imgs, t_preds, t_cls, t_patches):
        cls, patches = tokens(student.backbone, imgs)
        preds = student.head(cls)
        cls_cos   = F.cosine_similarity(project(cls), t_cls, dim=-1).mean()
        patch_cos = F.cosine_similarity(project_patch(patches), t_patches, dim=-1).mean()
        return preds, cls_cos, patch_cos

    def run_epoch(optimizer, scheduler) -> list[float]:
        student.train()
        totals = torch.zeros(6)
        n = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            r_imgs = torch.cat([next(avc_batches)[0], next(other_batches)[0]]).to(device)
            preds, cls_cos, patch_cos = losses(imgs, *(t := teach(teacher, imgs)))
            label_loss   = angular_loss(preds, labels)
            teacher_loss = angular_loss(preds, t[0])
            r_preds, r_cls_cos, r_patch_cos = losses(r_imgs, *(rt := teach(teacher, r_imgs)))
            real_loss = angular_loss(r_preds, rt[0])
            loss = ((1 - args.alpha) * label_loss + args.alpha * teacher_loss
                    + args.beta * (1 - cls_cos) + args.patch_beta * (1 - patch_cos)
                    + args.real_weight * (real_loss + args.beta * (1 - r_cls_cos) + args.patch_beta * (1 - r_patch_cos)))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            with torch.no_grad():
                for e, s in zip(ema.state_dict().values(), student.state_dict().values()):
                    if e.dtype.is_floating_point:
                        e.mul_(args.ema).add_(s, alpha=1 - args.ema)
                    else:
                        e.copy_(s)

            totals += torch.tensor([loss.item(), label_loss.item(), teacher_loss.item(), real_loss.item(),
                                    (cls_cos.item() + r_cls_cos.item()) / 2, (patch_cos.item() + r_patch_cos.item()) / 2])
            n += 1
            if n == args.max_batches:
                break
        out = (totals / n).tolist()
        return [out[0]] + [v * 180 / np.pi for v in out[1:4]] + out[4:]

    @torch.no_grad()
    def validate(model, forward) -> tuple[float, float, list[float]]:
        """Mean angular error (deg) to the labels on GazeGene's validation split, and to the teacher for each AVCAffe
        selection person: (GazeGene error, mean over people of their gaps, the gaps)."""
        model.eval()
        result = []
        for data, to_teacher in [(val_loader, False)] + [(v, True) for v in real_val_loaders]:
            err, n = 0.0, 0
            for imgs, labels in data:
                imgs = imgs.to(device)
                target = teach(teacher, imgs)[0] if to_teacher else labels.to(device)
                err += angular_loss(forward(imgs), target).item() * len(imgs)
                n += len(imgs)
                if args.max_batches and n >= args.max_batches * args.batch_size:
                    break
            result.append(float(err / n * 180 / np.pi))
        return result[0], float(np.mean(result[1:])), result[1:]

    gaps = lambda g: " / ".join(f"{v:.2f}°" for v in g)
    teacher_err, _, _ = validate(teacher, lambda imgs: teach(teacher, imgs)[0])
    best_gap = best_val = float("inf")
    best_epoch = 0
    if state:
        student.load_state_dict(state["student"]); ema.load_state_dict(state["ema"])
        project.load_state_dict(state["project"]); project_patch.load_state_dict(state["project_patch"])
        best_gap, best_val, best_epoch = state["best_gap"], state["best_val"], state["best_epoch"]
    start_err, start_gap, start_gaps = validate(ema if state else student, ema if state else student)
    if state:
        print(f"[distillR1] Resumed {args.resume} after epoch {state['epoch']} (best {best_gap:.2f}° at epoch {best_epoch})")
    print(f"[distillR1] Teacher on GazeGene val: {teacher_err:.2f}°; student at start: {start_err:.2f}°, "
          f"gap to the teacher on real val {start_gap:.2f}° ({gaps(start_gaps)})")

    def train_phase(phase: int, epochs: int, lr: float, frozen: bool, warmup: int):
        nonlocal best_gap, best_val, best_epoch, state
        student.backbone.requires_grad_(not frozen)
        # the projectors start from scratch (v1's is not saved): they learn at phase 1's rate, so the feature losses
        # make sense before the warm-up lets the student move
        optimizer = torch.optim.AdamW([
            {"params": [p for p in student.parameters() if p.requires_grad], "lr": lr},
            {"params": list(project.parameters()) + list(project_patch.parameters()), "lr": max(lr, args.lr_p1)},
        ], weight_decay=1e-4)
        steps = epochs * (args.max_batches or len(train_loader))
        warm = warmup * (args.max_batches or len(train_loader))
        # linear warm-up, then cosine down to 1% (a Python float: a numpy one would end up in the checkpoint)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, warm))
                                                      * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
        print(f"\n[distillR1] === Phase {phase} ({'frozen' if frozen else 'unfrozen'}) — {epochs} epochs, lr={lr} ===")
        no_improve, first = 0, 1
        if state and state["phase"] == phase:
            optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
            no_improve, first = state["no_improve"], state["epoch"] + 1
        state = None
        for epoch in range(first, epochs + 1):
            started = time.time()
            loss, label_err, teacher_gap, real_gap, cls_cos, patch_cos = run_epoch(optimizer, scheduler)
            val_err, real_val_gap, real_val_gaps = validate(ema, ema)
            cur_lr = scheduler.get_last_lr()[0]
            seconds = time.time() - started
            print(f"  ep {epoch:03d}/{epochs}  loss {loss:.4f}  err {label_err:.2f}°  gap {teacher_gap:.2f}°  "
                  f"real gap {real_gap:.2f}°  cos {cls_cos:.3f}/{patch_cos:.3f}  |  val {val_err:.2f}° "
                  f"(teacher {teacher_err:.2f}°)  real val gap {real_val_gap:.2f}° ({gaps(real_val_gaps)})  lr {cur_lr:.2e}  {seconds:.0f}s")
            csv_writer.writerow([phase, epoch, loss, label_err, teacher_gap, real_gap, cls_cos, patch_cos, val_err,
                                 real_val_gap, cur_lr, seconds])
            csv_file.flush()
            if real_val_gap < best_gap:
                best_gap, best_val, best_epoch = real_val_gap, val_err, epoch
                save_checkpoint(best_path, ema, optimizer, epoch, val_err)
                print(f"  ✓ best student saved (real val gap {real_val_gap:.2f}°, val {val_err:.2f}°)")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"  Early stopping (no improvement for {args.patience} epochs)")
                    break
            torch.save({"run_id": run_id, "phase": phase, "epoch": epoch, "student": student.state_dict(),
                        "ema": ema.state_dict(), "project": project.state_dict(),
                        "project_patch": project_patch.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "best_gap": best_gap, "best_val": best_val,
                        "best_epoch": best_epoch, "no_improve": no_improve}, str(last_path) + ".tmp")
            Path(str(last_path) + ".tmp").replace(last_path)

    if not args.init and not (state and state["phase"] == 2):
        train_phase(1, args.epochs_p1, args.lr_p1, frozen=True, warmup=0)
        ema.load_state_dict(student.state_dict())
    train_phase(2, args.epochs, args.lr, frozen=False, warmup=args.warmup)
    csv_file.close()

    print(f"\n[distillR1] Best student: real val gap {best_gap:.2f}°, val {best_val:.2f}° "
          f"(teacher {teacher_err:.2f}°) -> {best_path}")
    with open(logs_dir / "results.json", "w") as f:
        json.dump({"teacher": args.teacher, "student": args.student, "init": args.init,
                   "teacher_val_err_deg": round(teacher_err, 4), "start_val_err_deg": round(start_err, 4),
                   "start_real_val_gap_deg": round(start_gap, 4), "best_real_val_gap_deg": round(best_gap, 4),
                   "best_val_err_deg": round(best_val, 4), "best_epoch": best_epoch}, f, indent=2)


if __name__ == "__main__":
    main()
