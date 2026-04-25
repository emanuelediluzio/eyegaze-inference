"""
Gaze estimation model: DINOv2 backbone + regression head.

Input : (B, 3, 224, 224) RGB — ImageNet normalised
Output: (B, 2) — yaw, pitch in radians

Backbones available via torch.hub (facebookresearch/dinov2):
    dinov2_vits14  —  21 M params  (embed_dim=384)
    dinov2_vitb14  —  86 M params  (embed_dim=768)   <- default
    dinov2_vitl14  — 307 M params  (embed_dim=1024)
    dinov2_vitg14  — 1.1 B params  (embed_dim=1536)
"""
from __future__ import annotations
import torch
import torch.nn as nn


class GazeDINO(nn.Module):
    def __init__(
        self,
        backbone: str = "dinov2_vitb14",
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", backbone, pretrained=True
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        embed_dim = self.backbone.embed_dim  # 768 for ViT-B, 1024 for ViT-L

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 2),  # yaw, pitch
        )

        # init head weights
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # (B, embed_dim) — CLS token
        return self.head(features)    # (B, 2)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
