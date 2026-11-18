from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import torch

from N2uCore import log

FORMAT = "n2u.gazedino.inference.v1"
FORMAT_INT8 = "n2u.gazedino.inference.v1-int8"  # weight matrices as int8, one float32 scale per output row: 87 MB
                                                # instead of 324, same angles (MPIIFaceGaze 6.37 vs 6.35 deg)
SCALE = "@scale"                                # state_dict key of the scales of the int8 tensor <name>
ARCHITECTURE = {"name": "GazeDINO", "backbone": "dinov2_vitb14", "head_dims": [768, 512, 128, 2]}
PREPROCESSING = {
    "color": "RGB",
    "size": [224, 224],
    "resize": "PIL_bilinear_antialias",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
OUTPUT = {"names": ["yaw", "pitch"], "unit": "radians"}
FIELDS = {"format", "architecture", "preprocessing", "output", "state_dict", "metadata"}


def sha256(path):
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception as exc:
        log.err("sha256: {}".format(exc))
        return None


def validateArtifact(value):
    try:
        if not isinstance(value, dict) or set(value) != FIELDS or value["format"] not in (FORMAT, FORMAT_INT8):
            log.err("Unsupported GazeDINO inference artifact")
            return None
        for name, expected in [
            ("architecture", ARCHITECTURE),
            ("preprocessing", PREPROCESSING),
            ("output", OUTPUT),
        ]:
            if value[name] != expected:
                log.err("Incompatible GazeDINO " + name)
                return None
        state = value["state_dict"]
        if not isinstance(state, dict) or not state:
            log.err("Missing GazeDINO weights")
            return None
        int8 = value["format"] == FORMAT_INT8
        for name, array in state.items():
            if not isinstance(name, str) or not isinstance(array, np.ndarray):
                log.err("Invalid GazeDINO weight: " + str(name))
                return None
            if int8 and array.dtype == np.int8:
                scale = state.get(name + SCALE)
                valid = array.ndim >= 2 and isinstance(scale, np.ndarray) and scale.shape == (array.shape[0],)
            elif int8 and name.endswith(SCALE):
                quantized = state.get(name[: -len(SCALE)])
                valid = isinstance(quantized, np.ndarray) and quantized.dtype == np.int8 \
                    and array.dtype == np.float32 and np.isfinite(array).all() and (array > 0).all()
            else:
                valid = array.dtype == np.float32 and np.isfinite(array).all()
            if not valid:
                log.err("Invalid GazeDINO weight: " + str(name))
                return None
        metadata = value["metadata"]
        if not isinstance(metadata, dict) or set(metadata) != {"source_sha256", "validation_error_deg"}:
            log.err("Invalid GazeDINO metadata")
            return None
        digest = metadata["source_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            log.err("Invalid source digest")
            return None
        error = metadata["validation_error_deg"]
        if error is not None and (
            isinstance(error, bool)
            or not isinstance(error, (int, float))
            or not np.isfinite(error)
            or error < 0
        ):
            log.err("Invalid validation error")
            return None
        return value
    except Exception as exc:
        log.err("validateArtifact: {}".format(exc))
        return None


def loadModelState(path):
    """Read inference weights on CPU, or return None after logging failure."""
    try:
        path = Path(path)
        if path.suffix.lower() == ".joblib":
            artifact = validateArtifact(joblib.load(path))
            if artifact is None:
                return None
            state = artifact["state_dict"]
            return {name: torch.from_numpy(dequantize(state, name)) for name in state if not name.endswith(SCALE)}
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint["model"]
        if not isinstance(state, dict) or not state:
            log.err("Missing GazeDINO weights")
            return None
        return state
    except Exception as exc:
        log.err("loadModelState: {}".format(exc))
        return None


def quantize(array):
    """Symmetric int8 per output row (the first axis), as Core ML's linear_quantize_weights: (int8 array, scales)."""
    scale = np.abs(array.reshape(array.shape[0], -1)).max(axis=1) / 127.0
    scale = np.where(scale > 0, scale, 1.0).astype(np.float32)
    shape = (-1,) + (1,) * (array.ndim - 1)
    return np.clip(np.round(array / scale.reshape(shape)), -127, 127).astype(np.int8), scale


def dequantize(state, name):
    """A float32 weight of a state_dict, rebuilt from int8 and scales when it was quantized."""
    array = state[name]
    if array.dtype != np.int8:
        return array
    return array.astype(np.float32) * state[name + SCALE].reshape((-1,) + (1,) * (array.ndim - 1))


def exportInference(source, destination, int8=False):
    """The training checkpoint as the inference artifact; with int8, every weight matrix (Linear, patch embedding)
    as int8 with a scale per output row, the rest (biases, norms, tokens, position embedding) in float32."""
    try:
        source, destination = Path(source).expanduser(), Path(destination).expanduser()
        if destination.suffix.lower() != ".joblib":
            log.err("Destination must have a .joblib extension")
            return None
        if destination.exists():
            log.err("Destination already exists: {}".format(destination))
            return None
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        state = checkpoint["model"]
        if not isinstance(state, dict) or any(
            not isinstance(v, torch.Tensor) or v.dtype != torch.float32 for v in state.values()
        ):
            log.err("Expected a float32 model state_dict")
            return None
        weights = {name: tensor.detach().cpu().numpy() for name, tensor in state.items()}
        if int8:
            for name in [n for n, a in weights.items() if n.endswith(".weight") and a.ndim >= 2]:
                weights[name], weights[name + SCALE] = quantize(weights[name])
        artifact = validateArtifact(
            {
                "format": FORMAT_INT8 if int8 else FORMAT,
                "architecture": ARCHITECTURE.copy(),
                "preprocessing": PREPROCESSING.copy(),
                "output": OUTPUT.copy(),
                "state_dict": weights,
                "metadata": {
                    "source_sha256": sha256(source),
                    "validation_error_deg": checkpoint.get("val_err_deg"),
                },
            }
        )
        if artifact is None:
            return None
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".gaze-export-", dir=destination.parent)
        os.close(fd)
        try:
            joblib.dump(artifact, temporary, compress=3, protocol=4)
            restored = validateArtifact(joblib.load(temporary))
            if restored is None:
                return None
            restoredState = restored["state_dict"]
            if {n for n in restoredState if not n.endswith(SCALE)} != set(state):
                log.err("Export lost weight names")
                return None
            for name, tensor in state.items():
                original, weight = tensor.detach().cpu().numpy(), dequantize(restoredState, name)
                # int8: each value within half a step of its row; float32: identical
                tolerance = 0.0 if restoredState[name].dtype != np.int8 else \
                    restoredState[name + SCALE].reshape((-1,) + (1,) * (original.ndim - 1)) * 0.5001
                if weight.shape != original.shape or (np.abs(weight - original) > tolerance).any():
                    log.err("Export changed weight values: " + name)
                    return None
            # Publish the verified export without replacing an existing file.
            os.link(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return {
            "format": FORMAT_INT8 if int8 else FORMAT,
            "source_bytes": source.stat().st_size,
            "artifact_bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "tensor_count": len(state),
            "all_weights_exact": not int8,
            "int8_tensors": sum(1 for n in weights if n.endswith(SCALE)),
        }
    except Exception as exc:
        log.err("exportInference: {}".format(exc))
        return None
