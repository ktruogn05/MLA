"""CPU timing pilot on official Minesweeper; never evaluates the test labels."""
import argparse
import hashlib
import json
import platform
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()

    def monitor():
        while not stop.wait(0.05):
            peak[0] = max(peak[0], process.memory_info().rss)

    threading.Thread(target=monitor, daemon=True).start()
    path = Path('data/minesweeper.npz')
    raw = np.load(path)
    x = torch.from_numpy(raw['node_features'].copy())
    train = torch.from_numpy(raw['train_masks'][0].copy())
    val = torch.from_numpy(raw['val_masks'][0].copy())
    # Only materialize train and validation targets, not test targets.
    y_train = torch.from_numpy(raw['node_labels'][train.numpy()].copy()).float()
    y_val = raw['node_labels'][val.numpy()].copy()
    edges = raw['edges']
    n = len(x)
    start = time.perf_counter()
    adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    row, col = adj.nonzero()
    degree = np.asarray(adj.sum(1)).ravel()
    normalized = F.normalize(x, dim=1)
    proxy = torch.zeros(n)
    proxy.index_add_(0, torch.from_numpy(row.astype('int64')),
                    (normalized[row] * normalized[col]).sum(1))
    proxy /= torch.from_numpy(degree).float().clamp_min(1)

    def standardize(t):
        return (t - t.mean()) / (t.std(unbiased=False) + 1e-8)

    proxy = standardize(proxy)
    deg = standardize(torch.from_numpy(np.log1p(degree)).float())
    a = adj + sp.eye(n)
    scale = sp.diags(1 / np.sqrt(np.asarray(a.sum(1)).ravel()))
    a = (scale @ a @ scale).tocoo()
    sparse = torch.sparse_coo_tensor(np.vstack([a.row, a.col]),
                                     torch.from_numpy(a.data).float(), (n, n)).coalesce()
    pre_seconds = time.perf_counter() - start
    start = time.perf_counter()
    # Fixed random node order breaks exact cosine ties without using labels.
    order = torch.from_numpy(np.random.default_rng(42).permutation(n))
    candidates = []
    for begin in range(0, n, 64):
        end = min(begin + 64, n)
        scores = normalized[begin:end] @ normalized[order].T
        invalid = torch.from_numpy(adj[begin:end, :].toarray().astype(bool))[:, order]
        invalid |= torch.arange(begin, end)[:, None] == order[None, :]
        scores.masked_fill_(invalid, -torch.inf)
        chosen = scores.argsort(dim=1, descending=True, stable=True)[:, :8]
        candidates.append(order[chosen])
    candidates = torch.cat(candidates)
    retrieval_seconds = time.perf_counter() - start
    assert not torch.any(candidates == torch.arange(n)[:, None])
    assert not np.any(adj[np.repeat(np.arange(n), 8), candidates.numpy().ravel()])

    class Model(nn.Module):
        def __init__(self, kind):
            super().__init__()
            self.kind = kind
            self.encoder = nn.Linear(x.shape[1], 64)
            if kind != 'mlp':
                self.local = nn.ModuleList([nn.Linear(64, 64) for _ in range(2)])
            if kind in ('fixed', 'learned', 'proposed'):
                self.q = nn.Linear(64, 64, bias=False)
                self.k = nn.Linear(64, 64, bias=False)
                self.v = nn.Linear(64, 64, bias=False)
                if kind != 'fixed':
                    self.gate = nn.Sequential(nn.Linear(129, 32), nn.ReLU(), nn.Linear(32, 1))
                    self.gamma = nn.Parameter(torch.zeros(()), requires_grad=kind == 'proposed')
            self.out = nn.Linear(64, 1)

        def forward(self):
            h = F.relu(self.encoder(x))
            z = h
            if self.kind != 'mlp':
                for layer in self.local:
                    z = F.dropout(F.relu(torch.sparse.mm(sparse, layer(z))), 0.3, self.training)
            if self.kind in ('fixed', 'learned', 'proposed'):
                q, key, value = self.q(h), self.k(h), self.v(h)
                parts = []
                for b in range(0, n, 256):
                    ids = candidates[b:b+256]
                    weights = ((q[b:b+256, None] * key[ids]).sum(-1) / 8).softmax(-1)
                    parts.append((weights[..., None] * value[ids]).sum(1))
                g = torch.cat(parts)
                alpha = 0.5 if self.kind == 'fixed' else torch.sigmoid(
                    self.gate(torch.cat([deg[:, None], z, g], 1)) + self.gamma * proxy[:, None])
                z = alpha * z + (1 - alpha) * g
            return self.out(z).squeeze(-1)

    report = {'purpose': 'timing pilot, not final model comparison', 'dataset': str(path),
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'torch': torch.__version__, 'python': platform.python_version(),
              'platform': platform.platform(), 'cpu_threads': 2, 'split': 0, 'seed': 42,
              'nodes': n, 'hidden': 64, 'k': 8, 'gcn_layers': 2,
              'preprocess_seconds': pre_seconds, 'retrieval_seconds': retrieval_seconds,
              'models': []}
    print(f'Preprocess {pre_seconds:.2f}s, retrieval {retrieval_seconds:.2f}s', flush=True)
    for kind in ['mlp', 'gcn', 'fixed', 'learned', 'proposed']:
        torch.manual_seed(42)
        model = Model(kind)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        times = []
        for epoch in range(args.epochs):
            start = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model()[train], y_train)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                score = roc_auc_score(y_val, model()[val].numpy())
            times.append(time.perf_counter() - start)
        item = {'model': kind, 'epochs': args.epochs, 'total_seconds': sum(times),
                'median_epoch_seconds_after_warmup': float(np.median(times[3:])),
                'last_validation_auc': score, 'parameters': sum(p.numel() for p in model.parameters()),
                'epoch_seconds': times}
        report['models'].append(item)
        print(f'{kind}: {sum(times):.2f}s / {args.epochs} epochs, val AUC {score:.4f}', flush=True)
    stop.set()
    report['sampled_process_peak_rss_mib'] = peak[0] / 2**20
    report['available_ram_after_mib'] = psutil.virtual_memory().available / 2**20
    Path('results').mkdir(exist_ok=True)
    Path('results/cpu_pilot.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Peak process RSS: {report["sampled_process_peak_rss_mib"]:.0f} MiB', flush=True)


if __name__ == '__main__':
    main()
