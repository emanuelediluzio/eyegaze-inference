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
