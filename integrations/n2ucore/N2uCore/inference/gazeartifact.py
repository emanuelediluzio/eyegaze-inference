import hashlib
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import torch

from N2uCore import log

FORMAT = "n2u.gazedino.inference.v1"
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
        if not isinstance(value, dict) or set(value) != FIELDS or value["format"] != FORMAT:
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
        for name, array in state.items():
            if (
                not isinstance(name, str)
                or not isinstance(array, np.ndarray)
                or array.dtype != np.float32
                or not np.isfinite(array).all()
            ):
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
            return {name: torch.from_numpy(array) for name, array in artifact["state_dict"].items()}
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint["model"]
        if not isinstance(state, dict) or not state:
            log.err("Missing GazeDINO weights")
            return None
        return state
    except Exception as exc:
        log.err("loadModelState: {}".format(exc))
        return None


def exportInference(source, destination):
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
        artifact = validateArtifact(
            {
                "format": FORMAT,
                "architecture": ARCHITECTURE.copy(),
                "preprocessing": PREPROCESSING.copy(),
                "output": OUTPUT.copy(),
                "state_dict": {name: tensor.detach().cpu().numpy() for name, tensor in state.items()},
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
            if set(restored["state_dict"]) != set(state):
                log.err("Export lost weight names")
                return None
            for name, tensor in state.items():
                if not np.array_equal(restored["state_dict"][name], tensor.detach().cpu().numpy()):
                    log.err("Export changed weight values: " + name)
                    return None
            # Publish the verified export without replacing an existing file.
            os.link(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return {
            "format": FORMAT,
            "source_bytes": source.stat().st_size,
            "artifact_bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "tensor_count": len(state),
            "all_weights_exact": True,
        }
    except Exception as exc:
        log.err("exportInference: {}".format(exc))
        return None
