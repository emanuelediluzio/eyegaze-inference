"""Versioned inference-only GazeDINO artifacts: plain metadata + NumPy weights."""
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np

FORMAT = 'n2u.gazedino.inference.v1'
ARCHITECTURE = {'name': 'GazeDINO', 'backbone': 'dinov2_vitb14',
                'head_dims': [768, 512, 128, 2]}
PREPROCESSING = {'color': 'RGB', 'size': [224, 224], 'resize': 'PIL_bilinear_antialias',
                 'mean': [.485, .456, .406], 'std': [.229, .224, .225]}
OUTPUT = {'names': ['yaw', 'pitch'], 'unit': 'radians'}
FIELDS = {'format', 'architecture', 'preprocessing', 'output', 'state_dict', 'metadata'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact(value):
    if not isinstance(value, dict) or set(value) != FIELDS or value['format'] != FORMAT:
        raise ValueError('Unsupported GazeDINO inference artifact')
    for name, expected in [('architecture', ARCHITECTURE), ('preprocessing', PREPROCESSING), ('output', OUTPUT)]:
        if value[name] != expected:
            raise ValueError('Incompatible GazeDINO ' + name)
    state = value['state_dict']
    if not isinstance(state, dict) or not state:
        raise ValueError('Missing GazeDINO weights')
    for name, array in state.items():
        if not isinstance(name, str) or not isinstance(array, np.ndarray) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError('Invalid GazeDINO weight: ' + str(name))
    metadata = value['metadata']
    if not isinstance(metadata, dict) or set(metadata) != {'source_sha256', 'validation_error_deg'}:
        raise ValueError('Invalid GazeDINO metadata')
    digest = metadata['source_sha256']
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Invalid source digest')
    error = metadata['validation_error_deg']
    if error is not None and (isinstance(error, bool) or not isinstance(error, (int, float)) or not np.isfinite(error) or error < 0):
        raise ValueError('Invalid validation error')
    return value


def load_model_state(path):
    """Keep legacy .pt compatibility; joblib artifacts must be from trusted sources."""
    import torch
    path = Path(path)
    if path.suffix.lower() == '.joblib':
        import joblib
        artifact = validate_artifact(joblib.load(path))
        return {name: torch.from_numpy(array) for name, array in artifact['state_dict'].items()}
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    return checkpoint['model']


def export_inference(source, destination):
    """Drop all training state, preserve float32 values exactly, never overwrite."""
    import torch
    import joblib
    source, destination = Path(source).expanduser(), Path(destination).expanduser()
    if destination.suffix.lower() != '.joblib':
        raise ValueError('Destination must have a .joblib extension')
    if destination.exists():
        raise FileExistsError(destination)
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    state = checkpoint['model']
    if not isinstance(state, dict) or any(not isinstance(v, torch.Tensor) or v.dtype != torch.float32 for v in state.values()):
        raise ValueError('Expected a float32 model state_dict')
    artifact = validate_artifact({
        'format': FORMAT, 'architecture': ARCHITECTURE.copy(),
        'preprocessing': PREPROCESSING.copy(), 'output': OUTPUT.copy(),
        'state_dict': {name: tensor.detach().cpu().numpy() for name, tensor in state.items()},
        'metadata': {'source_sha256': sha256(source),
                     'validation_error_deg': checkpoint.get('val_err_deg')}})
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.gaze-export-', dir=destination.parent)
    os.close(fd)
    try:
        joblib.dump(artifact, temporary, compress=3, protocol=4)
        restored = validate_artifact(joblib.load(temporary))
        if set(restored['state_dict']) != set(state):
            raise ValueError('Export lost weight names')
        for name, tensor in state.items():
            if not np.array_equal(restored['state_dict'][name], tensor.detach().cpu().numpy()):
                raise ValueError('Export changed weight values: ' + name)
        # Same-filesystem hard link publishes only the fully verified file;
        # it fails rather than overwriting an existing destination.
        os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {'format': FORMAT, 'source_bytes': source.stat().st_size,
            'artifact_bytes': destination.stat().st_size, 'sha256': sha256(destination),
            'tensor_count': len(state), 'all_weights_exact': True}
