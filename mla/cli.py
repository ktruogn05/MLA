"""Explicit stages; no experiment runs on import or on `plan`."""
import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .config import configurations, run_identity, validate_plan, variants
from .data import Graph, download
from .train import evaluate, fit
from .utils import code_hash, digest, file_hash, read_json, write_json


def context(args):
    plan = read_json(args.config)
    validate_plan(plan)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    root = Path(args.output)
    if args.command not in {'plan', 'download'}:
        root.mkdir(parents=True, exist_ok=True)
    return plan, device, root


def register(plan, root, data_root):
    graphs = {name: Graph(data_root, name) for name in plan['datasets']}
    identity = dict(plan=plan, code=code_hash(), datasets={n: g.sha256 for n, g in graphs.items()})
    path = root / 'experiment.json'
    if path.exists():
        if read_json(path) != identity:
            raise ValueError('Plan/code/data differ from existing experiment. Use a new output directory.')
    else:
        write_json(path, identity)
    return graphs


def training_run(graph, cfg, split, seed, plan, device, root, cache):
    identity = run_identity(graph.sha256, cfg, split, seed, plan, code_hash())
    directory = root / 'jobs' / identity
    print(f'Job {identity[:12]}: {graph.name}/{cfg["model"]}, split={split}, seed={seed}', flush=True)
    started = time.perf_counter()
    try:
        result = fit(graph, cfg, split, seed, plan, device, directory, identity, cache)
    except Exception as exc:
        write_json(directory / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(),
                                                   config=cfg, split=split, seed=seed))
        raise  # Never silently discard failed configurations or change budgets.
    if not (directory / 'walltime.json').exists():
        write_json(directory / 'walltime.json', {'last_invocation_seconds': time.perf_counter()-started})
    return identity, result


def tune(args, plan, device, root, graphs):
    if (root / 'LOCK.json').exists():
        raise ValueError('Experiment is locked; tuning is closed. Use a new output for a new protocol.')
    selected, trials = {}, {}
    for dataset, graph in graphs.items():
        selected[dataset], trials[dataset] = {}, {}
        for model in plan['models']:
            candidates = configurations(plan, model)
            rows = []
            for cfg in candidates:
                values, jobs = [], []
                for split in plan['tune_splits']:
                    job, result = training_run(graph, cfg, split, plan['seed_base']+split,
                                                plan, device, root, args.cache)
                    values.append(result['best_validation'])
                    jobs.append(job)
                rows.append(dict(config=cfg, validation_mean=float(np.mean(values)), jobs=jobs))
            # Strict improvement + deterministic candidate order gives reproducible ties.
            winner = max(range(len(rows)), key=lambda i: rows[i]['validation_mean'])
            selected[dataset][model] = rows[winner]['config']
            trials[dataset][model] = rows
            write_json(root / 'search_progress.json', dict(selected=selected, trials=trials))
    write_json(root / 'selection.json', dict(selected=selected, trials=trials,
                                             experiment=digest(read_json(root / 'experiment.json'))))


def lock(plan, root):
    path = root / 'LOCK.json'
    if path.exists():
        raise ValueError('Already locked; do not overwrite the protocol')
    selection = read_json(root / 'selection.json')
    experiment = read_json(root / 'experiment.json')
    if selection['experiment'] != digest(experiment):
        raise ValueError('Selection does not match experiment')
    rows = variants(plan, selection['selected'])
    write_json(path, dict(experiment=experiment, selection_sha256=file_hash(root / 'selection.json'),
                          variants=rows, policy='No test-driven selection; all final jobs precede test evaluation'))


def check_lock(root):
    locked = read_json(root / 'LOCK.json')
    if locked['experiment'] != read_json(root / 'experiment.json'):
        raise ValueError('Experiment changed after lock')
    if locked['selection_sha256'] != file_hash(root / 'selection.json'):
        raise ValueError('Selection changed after lock')
    return locked


def final_fit(args, plan, device, root, graphs):
    locked = check_lock(root)
    manifest = []
    for variant in locked['variants']:
        graph = graphs[variant['dataset']]
        for split in plan['final_splits']:
            seeds = [plan['seed_base']+split]
            if variant['group'] == 'main':
                seeds += [s+split for s in plan.get('extra_seed_bases', [])]
            for seed in seeds:
                job, _ = training_run(graph, variant['config'], split, seed, plan, device, root, args.cache)
                manifest.append(dict(variant, split=split, seed=seed, job=job,
                                     checkpoint_sha256=file_hash(root / 'jobs' / job / 'best.pt')))
                write_json(root / 'fit_progress.json', manifest)
    write_json(root / 'manifest.json', dict(lock_sha256=file_hash(root / 'LOCK.json'), runs=manifest))


def final_evaluate(args, device, root, graphs):
    check_lock(root)
    manifest = read_json(root / 'manifest.json')
    if manifest['lock_sha256'] != file_hash(root / 'LOCK.json'):
        raise ValueError('Lock changed after fitting')
    # Validate every checkpoint before opening any test evaluation.
    for row in manifest['runs']:
        if row['checkpoint_sha256'] != file_hash(root / 'jobs' / row['job'] / 'best.pt'):
            raise ValueError('Checkpoint modified after manifest was finalized')
    evaluation = root / 'evaluation'
    evaluation.mkdir(exist_ok=True)
    unique = {}
    for row in manifest['runs']:
        job = row['job']
        if job in unique:
            continue
        result_path = evaluation / f'{job}.json'
        if result_path.exists():
            result = read_json(result_path)
            if result['checkpoint_sha256'] != row['checkpoint_sha256']:
                raise ValueError('Evaluation checkpoint mismatch')
        else:
            scores = evaluate(graphs[row['dataset']], root / 'jobs' / job / 'best.pt',
                              device, args.cache, evaluation / f'{job}.npz')
            result = dict(metrics=scores, checkpoint_sha256=row['checkpoint_sha256'])
            write_json(result_path, result)
        unique[job] = result
    write_json(evaluation / 'COMPLETE.json', dict(manifest_sha256=file_hash(root / 'manifest.json'),
                                                jobs=len(unique)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'download', 'tune', 'lock', 'fit', 'evaluate', 'analyze'])
    parser.add_argument('--config', default='configs/full.json')
    parser.add_argument('--output', default='runs/full')
    parser.add_argument('--data', default='data')
    parser.add_argument('--cache', default='cache')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, cuda:0, ...')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    plan, device, root = context(args)
    if args.command == 'plan':
        for name in plan['models']:
            configurations(plan, name)
        print(f"Datasets: {plan['datasets']}; models: {plan['models']}")
        count = len(plan['datasets'])*len(plan['models'])*plan['trials']*len(plan['tune_splits'])
        print(f'Tuning fits: {count}; final jobs are deduplicated by config/split/seed.')
        print('No data downloaded, no training started.')
        return
    if args.command == 'download':
        download(args.data, plan['datasets'])
        return
    graphs = register(plan, root, args.data)
    if args.command == 'tune':
        tune(args, plan, device, root, graphs)
    elif args.command == 'lock':
        lock(plan, root)
    elif args.command == 'fit':
        final_fit(args, plan, device, root, graphs)
    elif args.command == 'evaluate':
        final_evaluate(args, device, root, graphs)
    elif args.command == 'analyze':
        from .analysis import analyze
        analyze(root, graphs)


if __name__ == '__main__':
    main()

