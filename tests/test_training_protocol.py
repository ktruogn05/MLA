"""Prepared for the target machine; these tiny tests are not a full experiment."""
import numpy as np
import torch

from mla.config import DEFAULTS
from mla.data import Graph
from mla.train import fit, predict
from mla.utils import read_json


def test_train_never_requests_test_targets_and_resumes(tmp_path, monkeypatch):
    n = 8
    train = np.zeros((1,n), bool)
    val = train.copy()
    test = train.copy()
    train[0,:4] = True
    val[0,4:6] = True
    test[0,6:] = True
    np.savez(tmp_path / 'minesweeper.npz',
             node_features=np.random.default_rng(42).normal(size=(n,3)).astype('float32'),
             node_labels=np.array([0,1,0,1,0,1,0,1]),
             edges=np.array([[i,i+1] for i in range(n-1)]),
             train_masks=train, val_masks=val, test_masks=test)
    graph = Graph(tmp_path, 'minesweeper')
    original = graph.targets
    seen = []
    def guarded(split, subset):
        assert subset != 'test', 'Training must never request test targets'
        seen.append(subset)
        return original(split, subset)
    monkeypatch.setattr(graph, 'targets', guarded)
    cfg = dict(DEFAULTS, model='proposed', hidden=8, layers=1, k=2,
               attention_block=2, dropout=0.)
    plan = {'epochs': 2, 'patience': 2}
    job = tmp_path / 'job'
    cache = tmp_path / 'cache'
    first = fit(graph, cfg, 0, 42, plan, torch.device('cpu'), job, 'test-id', cache)
    before, _, _, _ = predict(graph, job / 'best.pt', torch.device('cpu'), cache)
    assert set(seen) == {'train','val'}
    assert first['epochs'] == 2
    # Simulate termination after the final last.pt save but before result.json commit.
    (job / 'result.json').unlink()
    second = fit(graph, cfg, 0, 42, plan, torch.device('cpu'), job, 'test-id', cache)
    after, _, _, _ = predict(graph, job / 'best.pt', torch.device('cpu'), cache)
    np.testing.assert_allclose(before, after)
    assert second['history'] == first['history']
    assert read_json(job / 'result.json')['identity'] == 'test-id'

