"""Notebook UI, consuming real checkpoints and evaluated predictions only."""
from pathlib import Path

import numpy as np
import torch

from .data import Graph, features
from .train import predict
from .utils import file_hash, read_json


def grid_coordinates(graph):
    side = int(np.sqrt(len(graph.x)))
    if side*side != len(graph.x):
        raise ValueError('Dataset is not a square grid')
    # Verify row-major mapping from actual topology, never infer coordinates from labels.
    for v in range(len(graph.x)):
        r, c = divmod(v, side)
        expected = {rr*side+cc for rr in range(max(0, r-1), min(side, r+2))
                    for cc in range(max(0, c-1), min(side, c+2)) if (rr, cc) != (r, c)}
        actual = set(graph.adj.indices[graph.adj.indptr[v]:graph.adj.indptr[v+1]])
        if expected != actual:
            raise ValueError('Row-major grid mapping does not match edges; provide verified coordinates first')
    return side


def launch(root='runs/full', data_root='data', cache='cache', split=0, seed=42):
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import display, clear_output

    root = Path(root)
    manifest = read_json(root / 'manifest.json')
    complete = read_json(root / 'evaluation/COMPLETE.json')
    if complete['manifest_sha256'] != file_hash(root / 'manifest.json'):
        raise ValueError('Complete final evaluation before opening the demo')
    graph = Graph(data_root, 'minesweeper')
    side = grid_coordinates(graph)
    rows = {x['name']: x for x in manifest['runs'] if x['dataset'] == 'minesweeper'
            and x['group'] == 'main' and x['split'] == split and x['seed'] == seed}
    required = ['gcn', 'global', 'learned', 'proposed']
    if not set(required) <= rows.keys():
        raise ValueError('Missing comparable main-model runs for this split/seed')
    probabilities = {}
    for name in required:
        row = rows[name]
        checkpoint_path = root / 'jobs' / row['job'] / 'best.pt'
        if file_hash(checkpoint_path) != row['checkpoint_sha256']:
            raise ValueError('Modified checkpoint')
        with np.load(root / 'evaluation' / f'{row["job"]}.npz') as f:
            probabilities[name] = f['probabilities']
    _, raw_proxy, _, _ = features(graph)
    test_nodes = np.flatnonzero(graph.masks['test'][split])
    node = widgets.BoundedIntText(value=int(test_nodes[0]), min=0, max=len(graph.x)-1, description='Node ID')
    truth = widgets.Checkbox(value=False, description='Show true label (post-hoc)')
    button = widgets.Button(description='Inspect node')
    output = widgets.Output()

    def render(_=None):
        with output:
            clear_output(wait=True)
            v = node.value
            if not graph.masks['test'][split, v]:
                print('Select a test node; example IDs:', test_nodes[:20].tolist())
                return
            p, extra, _, ckpt = predict(graph, root / 'jobs' / rows['proposed']['job'] / 'best.pt',
                                         torch.device('cpu'), cache, inspect_node=v)
            neighbors = graph.adj.indices[graph.adj.indptr[v]:graph.adj.indptr[v+1]]
            chosen = extra['attention']['ids'].numpy()
            weights = extra['attention']['weights'].numpy()
            alpha = float(extra['alpha'][v])
            print(f'Node {v}; proxy={raw_proxy[v]:.4f}; alpha={alpha:.4f}; '
                  f'local layers={ckpt["config"]["layers"]}')
            print('Observed features:', graph.x[v].tolist())
            print('Alpha mixes embeddings; attention weights are not causal explanations.')
            print('Highlighted neighbors are one-hop; deeper GCN layers see a larger region.')
            if truth.value:
                print('True label:', int(graph.targets(split, 'test')[np.flatnonzero(test_nodes == v)[0]]))
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            r, c = divmod(v, side)
            r0, r1, c0, c1 = max(0, r-10), min(side, r+11), max(0, c-10), min(side, c+11)
            image = axes[0].imshow(p.reshape(side, side), vmin=0, vmax=1, cmap='magma')
            axes[0].set_xlim(c0-.5, c1-.5)
            axes[0].set_ylim(r1-.5, r0-.5)
            axes[0].scatter(neighbors % side, neighbors // side, facecolors='none', edgecolors='cyan', label='1-hop')
            axes[0].scatter([c], [r], marker='x', c='lime', label='Selected')
            axes[0].set_title('Mine probability, selected region')
            axes[0].legend()
            split_map = np.zeros(len(p))
            split_map[graph.masks['val'][split]] = 1
            split_map[graph.masks['test'][split]] = 2
            axes[1].imshow(split_map.reshape(side, side), vmin=0, vmax=2, cmap='viridis', alpha=.4)
            axes[1].scatter(chosen % side, chosen // side, s=20, c='red', label='Non-local candidates')
            axes[1].scatter([c], [r], marker='x', c='black', label='Selected')
            axes[1].set_title('Masks: dark=train, medium=val, light=test')
            axes[1].legend()
            fig.colorbar(image, ax=axes[0])
            plt.show()
            display(pd.DataFrame([dict(model=m, mine_probability=float(probabilities[m][v]),
                                       prediction=int(probabilities[m][v] >= .5)) for m in required]))
            display(pd.DataFrame(dict(candidate=chosen, attention=weights)).sort_values('attention', ascending=False))
    button.on_click(render)
    display(widgets.VBox([widgets.HBox([node, truth, button]), output]))
    render()
    summary = root / 'analysis/runs.csv'
    if summary.exists():
        table = pd.read_csv(summary)
        display(table[(table.dataset == 'minesweeper') & (table.group.isin(['main', 'ablation']))])
    return dict(node=node, truth=truth, output=output)

