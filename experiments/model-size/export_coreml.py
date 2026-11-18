"""GazeDINO (or a distilled student) as Core ML packages, at several weight precisions, for size and speed tests.

    python export_coreml.py N2UCORE_DIR CHECKPOINT OUT_DIR [--student CKPT --backbone dinov2_vits14] [--name teacher]

CHECKPOINT is the inference artifact Iride loads (gazedino_inference.joblib). With --student, the student checkpoint
of distill_gaze.py replaces its weights. Output, in OUT_DIR:

  NAME_fp16.mlpackage    weights and compute in fp16
  NAME_w8.mlpackage      weights int8, linear per output channel (compute fp16)
  NAME_p6.mlpackage      weights palettized to 6 bits (64 values per tensor, k-means)
  NAME_p4.mlpackage      weights palettized to 4 bits, per grouped channel

Input "image": 1x3x224x224 float, the engine's normalized crop. Output "gaze": 1x2 (yaw, pitch in radians).
DINOv2 resizes its position embedding for every input size with a bicubic interpolation Core ML has not; the
input size is fixed here, so the resized embedding is computed once and stored as a constant.
"""
import argparse
import sys
from pathlib import Path

import coremltools as ct
import coremltools.optimize as cto
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "head-adaptation"))
from extract_session import studentModel  # noqa: E402


class FixedSize(nn.Module):
    """The model with DINOv2's position embedding resized once for 224x224."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        trunk = model.backbone
        with torch.no_grad():
            tokens = torch.zeros(1, 1 + 16 * 16, trunk.embed_dim)
            self.register_buffer("pos", trunk.interpolate_pos_encoding(tokens, 224, 224).clone())
        trunk.interpolate_pos_encoding = lambda x, w, h: self.pos

    def forward(self, image):
        return self.model(image)


def teacherModel(n2ucore, checkpoint):
    sys.path.insert(0, n2ucore)
    from N2uCore.inference.gazedinoengine import GazeDinoEngine
    from N2uCore.inference.gazeartifact import loadModelState

    model = GazeDinoEngine(checkpoint)._buildModel(Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main")
    model.load_state_dict(loadModelState(Path(checkpoint)), strict=True)
    return model


def size(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n2ucore")
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--student")
    ap.add_argument("--backbone", default="dinov2_vits14")
    ap.add_argument("--name", default="teacher")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = studentModel(args.student, args.backbone) if args.student else teacherModel(args.n2ucore, args.checkpoint)
    model = FixedSize(model.eval()).eval()
    example = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
        drift = (traced(example) - model(example)).abs().max().item()
    print(f"{args.name}: traced, max difference to eager {drift:.2e} rad", flush=True)

    fp16 = ct.convert(traced, inputs=[ct.TensorType(name="image", shape=example.shape)],
                      outputs=[ct.TensorType(name="gaze")], convert_to="mlprogram",
                      compute_precision=ct.precision.FLOAT16, minimum_deployment_target=ct.target.macOS15)
    variants = {
        "fp16": fp16,
        "w8": cto.coreml.linear_quantize_weights(fp16, cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                                             granularity="per_channel"))),
        "p6": cto.coreml.palettize_weights(fp16, cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(mode="kmeans", nbits=6))),
        "p4": cto.coreml.palettize_weights(fp16, cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(mode="kmeans", nbits=4, granularity="per_grouped_channel",
                                                        group_size=16))),
    }
    for tag, mlmodel in variants.items():
        path = out / f"{args.name}_{tag}.mlpackage"
        mlmodel.save(str(path))
        print(f"{args.name}_{tag}: {size(path):.1f} MB", flush=True)


if __name__ == "__main__":
    main()
