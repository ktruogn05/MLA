"""Post-test summary and oracle analysis; never called by training."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data import features
from .utils import file_hash, read_json, write_json


def correlation(a, b):
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    value = float(spearmanr(a, b).statistic)
    return value if np.isfinite(value) else None


def losses(y, p):
    if p.ndim == 1:
        p = np.clip(p, 1e-7, 1-1e-7)
        return -(y*np.log(p)+(1-y)*np.log1p(-p))
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1.))


def analyze(root, graphs):
    root = Path(root)
    complete = read_json(root / 'evaluation/COMPLETE.json')
    if complete['manifest_sha256'] != file_hash(root / 'manifest.json'):
        raise ValueError('Evaluation is not complete for this manifest')
    records = read_json(root / 'manifest.json')['runs']
    output = root / 'analysis'
    output.mkdir(exist_ok=True)
    rows = []
    for record in records:
        job = record['job']
        evaluation = read_json(root / 'evaluation' / f'{job}.json')['metrics']
        training = read_json(root / 'jobs' / job / 'result.json')
        rows.append(dict(dataset=record['dataset'], group=record['group'], model=record['name'],
                         split=record['split'], seed=record['seed'], **evaluation,
                         epochs=training['epochs'], parameters=training['trainable_parameters'],
                         compute_seconds=training['compute_seconds'],
                         peak_rss_mib=training['sampled_peak_rss_bytes']/2**20,
                         retrieval_build_seconds=training['preprocessing']['retrieval']['build_seconds']))
    table = pd.DataFrame(rows)
    table.to_csv(output / 'runs.csv', index=False)
    # Average extra seeds within split first; seeds are not new independent splits.
    numeric = ['accuracy', 'macro_f1', 'compute_seconds', 'peak_rss_mib', 'parameters']
    numeric += [key for key in ['roc_auc', 'average_precision'] if key in table]
    per_split = table.groupby(['dataset', 'group', 'model', 'split'])[numeric].mean()
    per_split.groupby(['dataset', 'group', 'model']).agg(['mean', 'std', 'count']).to_csv(output / 'summary.csv')
    comparisons, diagnostics, stats = [], [], {}
    for name, graph in graphs.items():
        # Oracle labels only used here, after explicit test-evaluation completion.
        y = graph._y
        r, c = graph.adj.nonzero()
        same = (y[r] == y[c]).astype(float)
        local = np.bincount(r, weights=same, minlength=len(y)) / np.maximum(graph.degree, 1)
        _, proxy, _, _ = features(graph)
        degree_mass = np.bincount(y, weights=graph.degree, minlength=graph.classes)
        expected = float(np.square(degree_mass/degree_mass.sum()).sum()) if degree_mass.sum() else None
        edge_h = float(same.mean()) if len(same) else None
        stats[name] = dict(nodes=len(y), undirected_edges=len(r)//2,
                           class_counts=np.bincount(y, minlength=graph.classes).tolist(),
                           edge_homophily=edge_h,
                           adjusted_homophily=(edge_h-expected)/(1-expected)
                           if expected is not None and expected < 1 else None)
        main = table[(table.dataset == name) & (table.group == 'main')]
        paired = main.pivot_table(index=['split', 'seed'], columns='model', values=graph.metric)
        if {'proposed', 'learned'} <= set(paired.columns):
            diff = (paired.proposed-paired.learned).groupby('split').mean()
            comparisons.append(dict(dataset=name, metric=graph.metric, mean_difference=float(diff.mean()),
                                     splits_improved=int((diff > 0).sum()), split_count=len(diff),
                                     differences={str(k): float(v) for k, v in diff.items()}))
        lookup = {(x['name'], x['split'], x['seed']): x for x in records
                  if x['dataset'] == name and x['group'] == 'main'}
        for (model, split, seed), record in lookup.items():
            if model != 'proposed':
                continue
            if not all((m, split, seed) in lookup for m in ['gcn', 'global']):
                continue
            def load(m):
                job = lookup[(m, split, seed)]['job']
                with np.load(root / 'evaluation' / f'{job}.npz') as f:
                    return {key: f[key] for key in f.files}
            prop, loc, glob = load('proposed'), load('gcn'), load('global')
            test = graph.masks['test'][split]
            nonisolated = test & (graph.degree > 0)
            delta = losses(y, glob['probabilities'])-losses(y, loc['probabilities'])
            alpha = prop['alpha']
            diagnostics.append(dict(dataset=name, split=split, seed=seed,
                spearman_proxy_homophily=correlation(proxy[nonisolated], local[nonisolated]),
                spearman_proxy_delta=correlation(proxy[test], delta[test]),
                spearman_proxy_alpha=correlation(proxy[test], alpha[test]),
                alpha_below_005=float((alpha[test] < .05).mean()),
                alpha_above_095=float((alpha[test] > .95).mean())))
            bounds = np.unique(np.quantile(proxy, np.linspace(0, 1, 6)))
            bins = np.searchsorted(bounds[1:-1], proxy, side='right')
            binned = pd.DataFrame(dict(bin=bins[test], delta=delta[test])).groupby('bin').agg(['mean', 'count'])
            binned.to_csv(output / f'{name}_{split}_{seed}_proxy_bins.csv')
            # Deterministic subsampling only for rendering; stats use all eligible test nodes.
            nodes = np.flatnonzero(nonisolated)[::max(1, nonisolated.sum()//2000)]
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
            axes[0].scatter(proxy[nodes], local[nodes], s=5, alpha=.3)
            axes[0].set(xlabel='Feature proxy', ylabel='Oracle local homophily')
            axes[1].scatter(proxy[nodes], delta[nodes], s=5, alpha=.3)
            axes[1].set(xlabel='Feature proxy', ylabel='Global loss - local loss')
            axes[2].hist(alpha[test], bins=20)
            axes[2].set(xlabel='Gate alpha', ylabel='Test nodes')
            fig.tight_layout()
            fig.savefig(output / f'{name}_{split}_{seed}.png', dpi=160)
            plt.close(fig)
    write_json(output / 'paired_comparisons.json', comparisons)
    write_json(output / 'proxy_diagnostics.json', diagnostics)
    write_json(output / 'oracle_dataset_statistics.json', stats)
    (output / 'README.md').write_text(
        '# Generated analysis\n\nsummary.csv: mean/sample std over splits after averaging seeds within split.\n'
        'Main tuned comparisons, fixed-config ablations and extensions remain separate.\n'
        'No node-level independence tests; overlapping splits are not independent graphs.\n'
        'These are measured results, not evidence of causal explanations.\n', encoding='utf-8')
