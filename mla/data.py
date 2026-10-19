"""Label-free graph preprocessing, reproducible retrieval, official dataset loader."""
import time
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from .utils import digest, file_hash, read_json, write_json

DATASETS = {'minesweeper': (2, 'roc_auc'), 'roman-empire': (18, 'accuracy')}
REMOTE_FILES = {'minesweeper': 'minesweeper.npz', 'roman-empire': 'roman_empire.npz'}
CACHE_VERSION = 1


def source_commit():
    """Resolve an immutable revision, with Git fallback for GitHub API limits."""
    req = urllib.request.Request('https://api.github.com/repos/yandex-research/heterophilous-graphs/commits/main',
                                 headers={'User-Agent': 'mla-course-project'})
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            commit = json.load(response)['sha']
    except (urllib.error.URLError, TimeoutError):
        try:
            result = subprocess.run(
                ['git', 'ls-remote', 'https://github.com/yandex-research/heterophilous-graphs.git',
                 'refs/heads/main'], capture_output=True, text=True, check=True, timeout=60)
            commit = result.stdout.split()[0]
        except (OSError, subprocess.SubprocessError, IndexError) as exc:
            raise RuntimeError('Cannot resolve dataset revision. Check network access and install Git '
                               'for the GitHub API fallback, or download the files manually.') from exc
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Invalid source commit returned by GitHub')
    return commit


def download(root, names):
    names = list(names)
    for name in names:
        if name not in DATASETS:
            raise ValueError(name)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    commit = None
    for name in names:
        path = root / f'{name}.npz'
        if path.exists():
            print(f'Keep existing {path}; SHA256={file_hash(path)}')
            continue
        if commit is None:
            commit = source_commit()
        url = f'https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/{commit}/data/{REMOTE_FILES[name]}'
        temp = path.with_suffix('.download')
        with urllib.request.urlopen(url, timeout=180) as src, open(temp, 'wb') as dst:
            import shutil
            shutil.copyfileobj(src, dst)
        temp.replace(path)
        write_json(path.with_suffix('.source.json'), dict(url=url, commit=commit, sha256=file_hash(path)))


class Graph:
    def __init__(self, root, name):
        if name not in DATASETS:
            raise ValueError(f'Unsupported dataset: {name}')
        self.name = name
        path = Path(root) / f'{name}.npz'
        self.sha256 = file_hash(path)
        self.classes, self.metric = DATASETS[name]
        with np.load(path, allow_pickle=False) as data:
            self.x = data['node_features'].astype(np.float32)
            # Labels are stored for later masked access; not used in graph preprocessing.
            self._y = data['node_labels'].astype(np.int64)
            edges = data['edges'].astype(np.int64)
            self.masks = {s: data[f'{s}_masks'].astype(bool) for s in ['train', 'val', 'test']}
        n = len(self.x)
        if self.x.ndim != 2 or not np.isfinite(self.x).all():
            raise ValueError('Features must be a finite N x d matrix')
        if self._y.shape != (n,) or edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError('Invalid labels or edges shape')
        if len(edges) and (edges.min() < 0 or edges.max() >= n):
            raise ValueError('Edge index out of range')
        shapes = {m.shape for m in self.masks.values()}
        if len(shapes) != 1 or any(m.ndim != 2 or m.shape[1] != n for m in self.masks.values()):
            raise ValueError('Invalid split masks')
        cover = sum(m.astype(np.int8) for m in self.masks.values())
        if not np.all(cover == 1) or any(not m.any(1).all() for m in self.masks.values()):
            raise ValueError('Masks must partition all nodes with nonempty subsets')
        self.adj = sp.coo_matrix((np.ones(len(edges), np.float32), (edges[:, 0], edges[:, 1])),
                                 shape=(n, n)).tocsr()
        self.adj = self.adj.maximum(self.adj.T)
        self.adj.setdiag(0)
        self.adj.eliminate_zeros()
        self.adj.data[:] = 1
        self.degree = np.asarray(self.adj.sum(1)).ravel()

    def targets(self, split, subset):
        if subset not in self.masks:
            raise ValueError(subset)
        y = self._y[self.masks[subset][split]]
        if not np.all((0 <= y) & (y < self.classes)):
            raise ValueError('Labels out of range')
        return y

    def tensors(self, device):
        return torch.from_numpy(self.x).to(device)


def normalize_adj(adj):
    deg = np.asarray(adj.sum(1)).ravel()
    scale = np.zeros_like(deg, dtype=np.float32)
    np.divide(1., np.sqrt(deg), out=scale, where=deg > 0)
    return (sp.diags(scale) @ adj @ sp.diags(scale)).tocoo()


def sparse_tensor(adj, device):
    a = adj.tocoo()
    return torch.sparse_coo_tensor(np.vstack([a.row, a.col]), a.data.astype(np.float32),
                                   a.shape, device=device).coalesce()


def exact_two_hop(adj, block=128, max_edges=20000000):
    """Strict distance two, excluding ego and all one-hop neighbors."""
    pieces, total = [], 0
    for b in range(0, adj.shape[0], block):
        part = (adj[b:b+block] @ adj).tocsr()
        part.data[:] = 1
        part = part - part.multiply(adj[b:b+block])
        part = part.tolil()
        for r in range(part.shape[0]):
            part[r, b+r] = 0
        part = part.tocsr()
        part.eliminate_zeros()
        total += part.nnz
        if total > max_edges:
            raise MemoryError('Two-hop graph exceeds max_twohop_edges; increase limit explicitly')
        pieces.append(part)
    return sp.vstack(pieces, format='csr')


def standardize(x):
    return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)


def features(graph):
    x = graph.x
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    unit = np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)
    score = np.zeros(len(x), np.float32)
    rows, cols = graph.adj.nonzero()
    for b in range(0, len(rows), 32768):
        r, c = rows[b:b+32768], cols[b:b+32768]
        np.add.at(score, r, np.einsum('ij,ij->i', unit[r], unit[c]))
    score /= np.maximum(graph.degree, 1)
    return unit, score, standardize(score), standardize(np.log1p(graph.degree))


def candidates(graph, unit, cfg, cache_dir):
    """Returns padded IDs (-1 invalid), cache timing, and retrieval diagnostics."""
    mode, k = cfg['retrieval'], cfg['k']
    key_data = dict(version=CACHE_VERSION, sha=graph.sha256, mode=mode, k=k,
                    seed=cfg['retrieval_seed'], exclude_hops=cfg['exclude_hops'])
    path = Path(cache_dir) / f'{digest(key_data)}.npz'
    meta_path = path.with_suffix('.json')
    t = time.perf_counter()
    if path.exists() and meta_path.exists():
        with np.load(path, allow_pickle=False) as f:
            ids = f['ids']
        meta = read_json(meta_path)
        return ids, dict(meta, cache_hit=True, load_seconds=time.perf_counter()-t)
    n = len(unit)
    excluded = graph.adj
    if cfg['exclude_hops'] == 2:
        excluded = excluded.maximum(exact_two_hop(excluded, max_edges=cfg['max_twohop_edges']))
    if mode == 'full':
        # Standard all-node self-attention, unlike restricted non-local retrieval.
        if n > cfg['full_max_nodes']:
            raise MemoryError(f'Full attention disabled for N={n}; explicit full_max_nodes limit')
        return None, dict(cache_hit=False, build_seconds=0., mode='full')
    rng = np.random.default_rng(cfg['retrieval_seed'])
    order = rng.permutation(n)
    ids = np.full((n, k), -1, dtype=np.int64)
    boundary_ties = 0
    ann = None
    if mode == 'ann':
        from pynndescent import NNDescent
        ann = NNDescent(unit, metric='cosine', random_state=cfg['retrieval_seed'], n_jobs=1)
        ann.prepare()
    for b in range(0, n, cfg['retrieval_block']):
        end = min(n, b + cfg['retrieval_block'])
        scores = None
        if mode == 'exact':
            scores = unit[b:end] @ unit[order].T
        pool = ann.query(unit[b:end], k=min(n, max(k*8, 64)))[0] if ann is not None else None
        for offset, v in enumerate(range(b, end)):
            forbidden = np.append(excluded.indices[excluded.indptr[v]:excluded.indptr[v+1]], v)
            allowed = order[~np.isin(order, forbidden)]
            count = min(k, len(allowed))
            if not count:
                continue
            if mode == 'random':
                selected = rng.choice(allowed, count, replace=False)
            elif mode == 'exact':
                s = scores[offset]
                s[np.isin(order, forbidden)] = -np.inf
                ranking = np.argsort(-s, kind='stable')
                selected = order[ranking[:count]]
                if count < len(allowed) and s[ranking[count-1]] == s[ranking[count]]:
                    boundary_ties += 1
            elif mode == 'ann':
                valid = np.unique(pool[offset])
                valid = valid[(valid >= 0) & ~np.isin(valid, forbidden)]
                if len(valid) < count:
                    valid = allowed  # Exact fallback when ANN filtering leaves too few nodes.
                sim = unit[valid] @ unit[v]
                selected = valid[np.argsort(-sim, kind='stable')[:count]]
            else:
                raise ValueError(f'Unknown retrieval {mode}')
            ids[v, :count] = selected
    meta = dict(key_data, build_seconds=time.perf_counter()-t, cache_hit=False,
                boundary_tie_fraction=boundary_ties/n if mode == 'exact' else None)
    if mode == 'ann':
        recalls = []
        for v in rng.choice(n, min(n, 100), replace=False):
            forbidden = np.append(excluded.indices[excluded.indptr[v]:excluded.indptr[v+1]], v)
            allowed = order[~np.isin(order, forbidden)]
            exact = allowed[np.argsort(-(unit[allowed] @ unit[v]), kind='stable')[:k]]
            if len(exact):
                recalls.append(len(set(exact) & set(ids[v])) / len(exact))
        meta['sample_recall_at_k'] = float(np.mean(recalls)) if recalls else None
        meta['build_seconds'] = time.perf_counter()-t
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement for interrupted preprocessing.
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, ids=ids)
    temp.replace(path)
    write_json(meta_path, meta)
    return ids, meta


def prepare(graph, cfg, device, cache_dir):
    t = time.perf_counter()
    unit, raw, proxy, degree = features(graph)
    out = dict(x=graph.tensors(device), proxy=torch.from_numpy(proxy).to(device),
               degree=torch.from_numpy(degree).to(device), raw_proxy=raw)
    n = len(unit)
    out['adj'] = sparse_tensor(normalize_adj(graph.adj + sp.eye(n)), device)
    if cfg['model'] == 'h2gcn':
        two = exact_two_hop(graph.adj, max_edges=cfg['max_twohop_edges'])
        out['hops'] = [sparse_tensor(normalize_adj(a), device) for a in [graph.adj, two]]
    if cfg['backbone'] == 'gat':
        a = (graph.adj + sp.eye(n)).tocoo()
        out['edge_index'] = torch.tensor(np.vstack([a.row, a.col]), dtype=torch.long, device=device)
    preprocessing = time.perf_counter()-t
    retrieval = dict(build_seconds=0., cache_hit=False)
    if cfg['model'] in {'global', 'fixed', 'learned', 'proposed', 'concat'}:
        ids, retrieval = candidates(graph, unit, cfg, cache_dir)
        out['candidates'] = None if ids is None else torch.from_numpy(ids).to(device)
    return out, dict(preprocess_seconds=preprocessing, retrieval=retrieval)
