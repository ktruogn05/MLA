"""Small synthetic checks, not benchmark experiments. Not executed during code delivery."""
import numpy as np
import scipy.sparse as sp
import torch
import pytest

from mla.config import DEFAULTS, configurations, variants
from mla.data import candidates, exact_two_hop, features, normalize_adj, sparse_tensor
from mla.models import Classifier
from mla.train import node_loss


def toy():
    from types import SimpleNamespace
    adj = sp.csr_matrix(np.array([[0,1,0,0], [1,0,1,0], [0,1,0,0], [0,0,0,0]], np.float32))
    return SimpleNamespace(x=np.array([[1,0], [0,1], [1,0], [0,0]], np.float32),
                           adj=adj, degree=np.asarray(adj.sum(1)).ravel(), sha256='synthetic-test')


def test_strict_twohop():
    two = exact_two_hop(toy().adj).toarray()
    expected = np.zeros((4,4))
    expected[0,2] = expected[2,0] = 1
    np.testing.assert_array_equal(two, expected)


def test_proxy_and_candidates(tmp_path):
    graph = toy()
    unit, raw, proxy, degree = features(graph)
    assert np.isfinite(proxy).all() and raw[3] == 0
    cfg = dict(DEFAULTS, k=8)
    ids, info = candidates(graph, unit, cfg, tmp_path)
    for v, row in enumerate(ids):
        valid = row[row >= 0]
        assert v not in valid and len(set(valid)) == len(valid)
        assert not graph.adj[v, valid].toarray().any()
        assert len(valid) == 4-1-int(graph.degree[v])
    again, cached = candidates(graph, unit, cfg, tmp_path)
    np.testing.assert_array_equal(ids, again)
    assert cached['cache_hit']


def test_empty_attention_and_gradient():
    graph = toy()
    _, raw, proxy, degree = features(graph)
    data = dict(x=torch.tensor(graph.x), proxy=torch.tensor([.1, .2, -.3, 0.]),
                degree=torch.tensor(degree), adj=sparse_tensor(normalize_adj(graph.adj+sp.eye(4)), 'cpu'),
                candidates=torch.tensor([[2,-1], [3,-1], [0,3], [-1,-1]]))
    model = Classifier(2, 2, dict(DEFAULTS, model='proposed', dropout=0.))
    logits, extra = model(data)
    assert torch.isfinite(logits).all()
    loss = node_loss(logits[:2], torch.tensor([0, 1])).mean()
    loss.backward()
    assert model.gamma.grad is not None and torch.isfinite(model.gamma.grad)
    assert model.encoder.weight.grad is not None
    assert extra['alpha'].shape == (4,)


def test_train_targets_do_not_depend_on_heldout_labels():
    torch.manual_seed(7)
    logits = torch.randn(4, 1, requires_grad=True)
    mask = torch.tensor([True, False, True, False])
    a = torch.tensor([0, 0, 1, 1])
    b = torch.tensor([0, 1, 1, 0])
    first = torch.autograd.grad(node_loss(logits[mask], a[mask]).mean(), logits)[0]
    second = torch.autograd.grad(node_loss(logits[mask], b[mask]).mean(), logits)[0]
    torch.testing.assert_close(first, second)
    assert torch.equal(first[~mask], torch.zeros_like(first[~mask]))


@pytest.mark.parametrize('kind', ['mlp', 'gcn', 'global', 'fixed', 'learned', 'proposed', 'concat', 'h2gcn'])
def test_all_model_shapes_and_gradients(kind):
    graph = toy()
    _, _, proxy, degree = features(graph)
    data = dict(x=torch.tensor(graph.x), proxy=torch.tensor(proxy), degree=torch.tensor(degree),
                adj=sparse_tensor(normalize_adj(graph.adj+sp.eye(4)), 'cpu'),
                candidates=torch.tensor([[2,3], [3,-1], [0,3], [0,1]]),
                hops=[sparse_tensor(normalize_adj(a), 'cpu')
                      for a in [graph.adj, exact_two_hop(graph.adj)]])
    cfg = dict(DEFAULTS, model=kind, hidden=8, layers=2, dropout=0.)
    model = Classifier(2, 3, cfg)
    logits, _ = model(data)
    assert logits.shape == (4,3) and torch.isfinite(logits).all()
    node_loss(logits[:2], torch.tensor([0,1])).mean().backward()
    assert model.encoder.weight.grad is not None


def test_search_and_ablation_protocol():
    from pathlib import Path
    from mla.utils import read_json
    plan = read_json(Path(__file__).parents[1] / 'configs/full.json')
    selection = {}
    for dataset in plan['datasets']:
        selection[dataset] = {}
        for model in plan['models']:
            candidates_ = configurations(plan, model)
            assert len(candidates_) == 12
            assert candidates_ == configurations(plan, model)
            selection[dataset][model] = candidates_[0]
    rows = variants(plan, selection)
    for row in rows:
        if row['name'] == 'no_proxy':
            reference = dict(selection[row['dataset']]['proposed'], model='learned')
            assert row['config'] == reference

