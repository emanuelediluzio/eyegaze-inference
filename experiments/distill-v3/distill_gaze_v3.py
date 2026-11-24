"""
Gaze Estimation — distillation v3: distillation v2 (../model-size/distill_gaze_v2.py) with real faces in more head
poses, and a run that survives a restart.

What changes from v2:
  - --real_dir and --real_val_dir take several folders, comma separated: MPIIFaceGaze crops plus crops of free-pose
    recordings (record_free.py), where v2 lost when the head came closer or looked down;
  - the best checkpoint is chosen on the mean of the gaps to the teacher over the --real_val_dir folders, so frontal
    and free-pose faces count the same (v2, chosen on MPIIFaceGaze alone, stopped improving at epoch 2);
  - every epoch saves last.pt (student, EMA, projectors, optimizer, scheduler, counters); --resume continues from it.

Distillation v2, for reference:

What changes from v1 (see ../head-adaptation/README.md for v1):
  - real faces without labels: every batch also has --real_batch crops cut by Iride (export_crops.py);
    on them the student learns only from the teacher (answers and features), as in DistillGaze (2026);
  - patch tokens: besides the CLS token, the student's patch tokens (projected) follow the teacher's,
    1 - cosine, weight --patch_beta (the CLS cosine alone saturated at 0.998 in v1);
  - a longer schedule at a higher learning rate (v1 was still improving when its cosine schedule ended);
  - an EMA of the student's weights is what is validated and saved;
  - the best checkpoint is chosen on the gap to the teacher on held-out real faces (--real_val_dir), where v1 lost
    accuracy; the error on GazeGene's validation split is logged too.
The teacher sees the same augmented batch as the student ("consistent teaching", Beyer et al. 2022).

Runs from the eye-gaze training code (next to train_gaze.py), like distill_gaze.py:
    python distill_gaze_v3.py --teacher ../teacher/best_run3_3.24deg.pt --init checkpoints/distill_v2/best.pt \
        --real_dir crops/mpii_p00-p09,crops/free_train --real_val_dir crops/mpii_p10-p14,crops/free_val
    python distill_gaze_v3.py ... --resume checkpoints/distill3_<run>/last.pt      # after a restart, same arguments
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
    p.add_argument("--real_val_dir", required=True, help="real face crops not used for training, comma separated folders")
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


def crops(folders: str) -> list[str]:
    # not the hidden "._" files macOS adds when copying to another disk: they are no images
    return sorted(str(p) for f in folders.split(",") for p in Path(f).glob("*.jpg") if not p.name.startswith("."))


def main():
    args   = parse_args()
    device = get_device()
    torch.backends.cuda.matmul.allow_tf32 = True     # A6000: TF32 matmuls, as fast as fp16 for the student
    torch.backends.cudnn.allow_tf32 = True
    print(f"[distill3] Device: {device}")

    from data.gazegene_loader import load_gazegene
    from data.gaze_dataset    import LazyGazeDataset
    from models.gaze_model    import GazeDINO, angular_loss

    p_train, p_val, y_train, y_val = load_gazegene(
        args.dataset_dir, max_subjects=args.max_subjects, max_frames_per_camera=args.max_frames, lazy=True)
    real = crops(args.real_dir)
    real_vals = [crops(f) for f in args.real_val_dir.split(",")]
    loader = lambda paths, labels, augment, bs, shuffle: DataLoader(
        LazyGazeDataset(paths, labels, augment=augment), batch_size=bs, shuffle=shuffle,
        num_workers=args.workers, drop_last=augment, persistent_workers=args.workers > 0)
    train_loader    = loader(p_train, y_train, True, args.batch_size, True)
    val_loader      = loader(p_val, y_val, False, args.batch_size, False)
    # real crops have no label: zeros, never used (the flip negates them, harmlessly)
    real_loader     = loader(real, np.zeros((len(real), 2), np.float32), True, args.real_batch, True)
    real_val_loaders = [loader(v, np.zeros((len(v), 2), np.float32), False, args.batch_size, False) for v in real_vals]
    print(f"[distill3] GazeGene train {len(p_train)}, val {len(p_val)}; real train {len(real)}, "
          f"real val {' + '.join(str(len(v)) for v in real_vals)}")

    teacher = load_teacher(args.teacher, args.teacher_backbone, device)
    student = GazeDINO(backbone=args.student, dropout=args.dropout).to(device)
    if args.init:
        student.load_state_dict(torch.load(args.init, map_location="cpu")["model"])
        print(f"[distill3] Student from {args.init}")
    s_dim, t_dim = student.backbone.embed_dim, teacher.backbone.embed_dim
    project       = nn.Linear(s_dim, t_dim).to(device)     # used only for the loss, not saved with the student
    project_patch = nn.Linear(s_dim, t_dim).to(device)
    ema = copy.deepcopy(student).eval().requires_grad_(False)

    state = torch.load(args.resume, map_location="cpu") if args.resume else None
    run_id   = state["run_id"] if state else datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path("checkpoints") / f"distill3_{run_id}"
    logs_dir = Path("logs")        / f"distill3_{run_id}"
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
    real_batches = itertools.cycle(real_loader)

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
            r_imgs = next(real_batches)[0].to(device)
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
        """Mean angular error (deg) to the labels on GazeGene's validation split, and to the teacher on each folder of
        real crops: (GazeGene error, mean of the folder gaps, folder gaps)."""
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
        print(f"[distill3] Resumed {args.resume} after epoch {state['epoch']} (best {best_gap:.2f}° at epoch {best_epoch})")
    print(f"[distill3] Teacher on GazeGene val: {teacher_err:.2f}°; student at start: {start_err:.2f}°, "
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
        print(f"\n[distill3] === Phase {phase} ({'frozen' if frozen else 'unfrozen'}) — {epochs} epochs, lr={lr} ===")
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

    print(f"\n[distill3] Best student: real val gap {best_gap:.2f}°, val {best_val:.2f}° "
          f"(teacher {teacher_err:.2f}°) -> {best_path}")
    with open(logs_dir / "results.json", "w") as f:
        json.dump({"teacher": args.teacher, "student": args.student, "init": args.init,
                   "teacher_val_err_deg": round(teacher_err, 4), "start_val_err_deg": round(start_err, 4),
                   "start_real_val_gap_deg": round(start_gap, 4), "best_real_val_gap_deg": round(best_gap, 4),
                   "best_val_err_deg": round(best_val, 4), "best_epoch": best_epoch}, f, indent=2)


if __name__ == "__main__":
    main()
