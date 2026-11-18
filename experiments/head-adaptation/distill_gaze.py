"""
Gaze Estimation — distil the fine-tuned GazeDINO (teacher) into a smaller DINOv2 (student).

Same data, split and two phases as train_gaze.py; only the loss changes. The student learns from
  - the labels:                  angular loss to the ground truth,          weight 1 - alpha
  - the teacher's answers:       angular loss to the teacher's yaw/pitch,   weight alpha
  - the teacher's features:      1 - cosine between the student CLS token (projected to the
                                 teacher's size) and the teacher CLS token,  weight beta
The teacher sees the same augmented batch as the student ("consistent teaching"), so it runs online,
in half precision on a GPU. The best checkpoint holds only the student, in the GazeDINO format, so
eval_held_out.py and the inference code load it unchanged.

Usage:
    python distill_gaze.py --teacher checkpoints/best_run3_3.24deg.pt --max_subjects 50
    python distill_gaze.py --teacher ... --student dinov2_vits14 --alpha 0.5 --beta 1.0
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

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
    p.add_argument("--student",      default="dinov2_vits14",
                   choices=["dinov2_vits14", "dinov2_vitb14"])
    # the teacher was trained on the first 50 subjects with this split: the student must use the same
    p.add_argument("--max_subjects", type=int, default=50)
    p.add_argument("--max_frames",   type=int, default=200)
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--epochs_p1",    type=int, default=10)
    p.add_argument("--epochs_p2",    type=int, default=50)
    p.add_argument("--lr_p1",        type=float, default=1e-3)
    p.add_argument("--lr_p2",        type=float, default=5e-6)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--patience",     type=int,   default=12)
    p.add_argument("--alpha",        type=float, default=0.5, help="weight of the teacher's answers")
    p.add_argument("--beta",         type=float, default=1.0, help="weight of the teacher's features")
    p.add_argument("--max_batches",  type=int,   default=None, help="cap per epoch (smoke test)")
    return p.parse_args()


def load_teacher(path: str, backbone: str, device: torch.device) -> nn.Module:
    from models.gaze_model import GazeDINO
    teacher = GazeDINO(backbone=backbone)
    teacher.load_state_dict(torch.load(path, map_location="cpu")["model"])
    dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    teacher = teacher.eval().to(device, dtype)
    teacher.requires_grad_(False)
    return teacher


@torch.no_grad()
def teach(teacher: nn.Module, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher yaw/pitch and CLS token, back in fp32."""
    features = teacher.backbone(imgs.to(next(teacher.parameters()).dtype))
    return teacher.head(features).float(), features.float()


def main():
    args   = parse_args()
    device = get_device()
    print(f"[distill] Device: {device}")

    from data.gazegene_loader import load_gazegene
    from data.gaze_dataset    import LazyGazeDataset
    from models.gaze_model    import GazeDINO, angular_loss

    p_train, p_val, y_train, y_val = load_gazegene(
        args.dataset_dir,
        max_subjects=args.max_subjects,
        max_frames_per_camera=args.max_frames,
        lazy=True,
    )
    train_loader = DataLoader(LazyGazeDataset(p_train, y_train, augment=True),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(LazyGazeDataset(p_val, y_val, augment=False),
                              batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[distill] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    teacher = load_teacher(args.teacher, args.teacher_backbone, device)
    student = GazeDINO(backbone=args.student, dropout=args.dropout, freeze_backbone=True).to(device)
    # student CLS -> teacher CLS; used only for the loss, not saved with the student
    project = nn.Linear(student.backbone.embed_dim, teacher.backbone.embed_dim).to(device)

    print(f"[distill] Teacher: {args.teacher_backbone} ({args.teacher}), "
          f"student: {args.student} ({sum(p.numel() for p in student.parameters()):,} params)")

    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path("checkpoints") / f"distill_{run_id}"
    logs_dir = Path("logs")        / f"distill_{run_id}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"

    csv_file   = open(logs_dir / "training_log.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["phase", "epoch", "loss", "label_err_deg", "teacher_gap_deg", "feature_cos",
                         "val_err_deg", "lr"])

    def run_epoch(optimizer) -> tuple[float, float, float, float]:
        """One training epoch: (loss, error to labels, gap to teacher, feature cosine)."""
        student.train()
        totals = torch.zeros(4)
        n = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            t_preds, t_features = teach(teacher, imgs)

            features = student.backbone(imgs)
            preds    = student.head(features)
            label_loss   = angular_loss(preds, labels)
            teacher_loss = angular_loss(preds, t_preds)
            feature_cos  = F.cosine_similarity(project(features), t_features, dim=-1).mean()
            loss = (1 - args.alpha) * label_loss + args.alpha * teacher_loss + args.beta * (1 - feature_cos)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(student.parameters()) + list(project.parameters()), max_norm=1.0)
            optimizer.step()

            totals += torch.tensor([loss.item(), label_loss.item(), teacher_loss.item(), feature_cos.item()])
            n += 1
            if n == args.max_batches:
                break
        loss, label_err, teacher_gap, cos = (totals / n).tolist()
        return loss, label_err * 180 / torch.pi, teacher_gap * 180 / torch.pi, cos

    @torch.no_grad()
    def validate(model, forward) -> float:
        """Mean angular error (deg) to the labels over the validation split."""
        model.eval()
        err, n = 0.0, 0
        for imgs, labels in val_loader:
            err += angular_loss(forward(imgs.to(device)), labels.to(device)).item() * len(labels)
            n += len(labels)
            if args.max_batches and n >= args.max_batches * args.batch_size:
                break
        return err / n * 180 / torch.pi

    teacher_err = validate(teacher, lambda imgs: teach(teacher, imgs)[0])
    print(f"[distill] Teacher on the validation split: {teacher_err:.2f}°")

    best_val_err = float("inf")

    def train_phase(phase: int, epochs: int, lr: float, frozen: bool):
        nonlocal best_val_err
        if frozen:
            student.freeze_backbone()
        else:
            student.unfreeze_backbone()

        trainable = [p for p in student.parameters() if p.requires_grad] + list(project.parameters())
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
        print(f"\n[distill] === Phase {phase} ({'frozen' if frozen else 'unfrozen'}) "
              f"— {epochs} epochs, lr={lr}, trainable={sum(p.numel() for p in trainable):,} ===")

        no_improve = 0
        for epoch in range(1, epochs + 1):
            started = time.time()
            loss, label_err, teacher_gap, cos = run_epoch(optimizer)
            val_err = validate(student, student)
            scheduler.step()

            cur_lr = scheduler.get_last_lr()[0]
            print(f"  ep {epoch:03d}/{epochs}  loss {loss:.4f}  err {label_err:.2f}°  "
                  f"teacher gap {teacher_gap:.2f}°  feat cos {cos:.3f}  |  val_err {val_err:.2f}°  "
                  f"(teacher {teacher_err:.2f}°)  lr {cur_lr:.2e}  {time.time() - started:.0f}s")
            csv_writer.writerow([phase, epoch, loss, label_err, teacher_gap, cos, val_err, cur_lr])
            csv_file.flush()

            if val_err < best_val_err:
                best_val_err = val_err
                save_checkpoint(best_path, student, optimizer, epoch, val_err)
                print(f"  ✓ best student saved ({val_err:.2f}°)")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"  Early stopping (no improvement for {args.patience} epochs)")
                    break

    train_phase(1, args.epochs_p1, args.lr_p1, frozen=True)
    train_phase(2, args.epochs_p2, args.lr_p2, frozen=False)
    csv_file.close()

    print(f"\n[distill] Best student: {best_val_err:.2f}° (teacher {teacher_err:.2f}°) -> {best_path}")
    with open(logs_dir / "results.json", "w") as f:
        json.dump({
            "teacher": args.teacher, "teacher_backbone": args.teacher_backbone, "student": args.student,
            "alpha": args.alpha, "beta": args.beta,
            "teacher_val_err_deg": round(teacher_err, 4),
            "best_val_err_deg": round(best_val_err, 4),
        }, f, indent=2)


if __name__ == "__main__":
    main()
