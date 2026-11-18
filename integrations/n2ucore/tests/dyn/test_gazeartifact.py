"""Inference export is lossless, bounded in scope and independent of training code."""
import copy
import joblib
import numpy as np
import pytest
import torch
from N2uCore.inference.gazeartifact import exportInference, loadModelState, validateArtifact


def checkpoint(tmp_path):
    path = tmp_path/'training.pt'
    torch.save({'model': {'head.weight': torch.arange(6, dtype=torch.float32).reshape(2,3),
                         'head.bias': torch.tensor([.2,-.1])},
                'optimizer': {'large_training_only': torch.ones(10000)},
                'epoch': 50, 'val_err_deg': 3.24}, path)
    return path


def test_export_excludes_training_state_and_preserves_outputs_exactly(tmp_path):
    source = checkpoint(tmp_path); output = tmp_path/'model.joblib'
    report = exportInference(source, output)
    assert report['all_weights_exact']
    assert report['artifact_bytes'] < report['source_bytes']
    package = joblib.load(output)
    assert 'optimizer' not in package and 'epoch' not in package
    original, restored = loadModelState(source), loadModelState(output)
    x = torch.tensor([[1.,2.,3.],[-1.,0.,2.]])
    for key in original: assert torch.equal(original[key], restored[key])
    assert torch.equal(x@original['head.weight'].T+original['head.bias'], x@restored['head.weight'].T+restored['head.bias'])
    for value in package['state_dict'].values():assert type(value) is np.ndarray


def test_export_never_overwrites_an_existing_artifact(tmp_path):
    source = checkpoint(tmp_path); output = tmp_path/'model.joblib';output.write_bytes(b'keep')
    assert exportInference(source, output) is None
    assert output.read_bytes() == b'keep'


def test_incompatible_or_nonfinite_artifacts_are_rejected(tmp_path):
    output = tmp_path/'model.joblib';exportInference(checkpoint(tmp_path), output)
    original = joblib.load(output)
    for field,value in [('format','other'),('preprocessing',{}),('architecture',{}),('output',{}),('metadata',{})]:
        invalid=copy.deepcopy(original);invalid[field]=value
        assert validateArtifact(invalid) is None
    invalid=copy.deepcopy(original);invalid['state_dict']['head.bias'][0]=np.nan
    assert validateArtifact(invalid) is None
    invalid=copy.deepcopy(original);invalid['optimizer']={}
    assert validateArtifact(invalid) is None


def test_int8_export_quantizes_weight_matrices_and_restores_float32_within_half_a_step(tmp_path):
    source = tmp_path/'training.pt'
    weight = torch.randn(4, 3)
    torch.save({'model': {'head.weight': weight, 'head.bias': torch.tensor([.2, -.1, 0., 1.])}, 'val_err_deg': 3.24},
               source)
    output = tmp_path/'model.joblib'
    report = exportInference(source, output, int8=True)
    assert report['format'] == 'n2u.gazedino.inference.v1-int8' and report['int8_tensors'] == 1
    state = joblib.load(output)['state_dict']
    assert state['head.weight'].dtype == np.int8 and state['head.weight@scale'].shape == (4,)
    assert state['head.bias'].dtype == np.float32                  # vectors stay float32
    restored = loadModelState(output)
    assert set(restored) == {'head.weight', 'head.bias'} and restored['head.weight'].dtype == torch.float32
    step = weight.abs().max(dim=1).values / 127
    assert ((restored['head.weight'] - weight).abs() <= step[:, None] / 2 + 1e-7).all()
    assert torch.equal(restored['head.bias'], torch.tensor([.2, -.1, 0., 1.]))


def test_int8_artifacts_without_scales_or_int8_in_float_artifacts_are_rejected(tmp_path):
    source = tmp_path/'training.pt'
    torch.save({'model': {'head.weight': torch.randn(4, 3), 'head.bias': torch.zeros(4)}, 'val_err_deg': 3.24}, source)
    output = tmp_path/'model.joblib'; exportInference(source, output, int8=True)
    original = joblib.load(output)
    invalid = copy.deepcopy(original); del invalid['state_dict']['head.weight@scale']
    assert validateArtifact(invalid) is None
    invalid = copy.deepcopy(original); invalid['state_dict']['head.weight@scale'][0] = 0
    assert validateArtifact(invalid) is None
    invalid = copy.deepcopy(original); invalid['format'] = 'n2u.gazedino.inference.v1'
    assert validateArtifact(invalid) is None
