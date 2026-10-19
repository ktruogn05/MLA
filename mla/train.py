import gc
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch.nn import functional as F

from .data import prepare
from .models import Classifier
from .utils import (MemoryMonitor, environment, load_torch, read_json, restore_rng, rng_state,
                    save_torch, seed_all, synchronize, write_json)


def probabilities(logits):
    return logits.sigmoid()[:, 0] if logits.shape[1] == 1 else logits.softmax(-1)


def metrics(y, p):
    binary = p.ndim == 1
    predicted = (p >= 0.5).astype(int) if binary else p.argmax(1)
    out = dict(accuracy=float(accuracy_score(y, predicted)),
               macro_f1=float(f1_score(y, predicted, average='macro', zero_division=0)))
    if binary:
        out['roc_auc'] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
        out['average_precision'] = float(average_precision_score(y, p)) if (y == 1).any() else None
    return out


def node_loss(logits, target):
    return (F.binary_cross_entropy_with_logits(logits[:, 0], target.float(), reduction='none')
            if logits.shape[1] == 1 else F.cross_entropy(logits, target.long(), reduction='none'))


def prepare_run(graph, cfg, seed, device, cache):
    data, timing = prepare(graph, cfg, device, cache)
    if cfg['shuffle_proxy']:
        perm = np.random.default_rng(seed).permutation(len(graph.x))
        data['proxy'] = data['proxy'][torch.as_tensor(perm, device=device)]
    return data, timing


def fit(graph, cfg, split, seed, plan, device, directory, identity, cache):
    """Train with masked train targets; select checkpoints using validation only."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / 'result.json'
    if result_path.exists():
        result = read_json(result_path)
        if result['identity'] != identity:
            raise ValueError('Run directory identity mismatch')
        return result
    seed_all(seed)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    with MemoryMonitor() as memory:
        data, preprocessing = prepare_run(graph, cfg, seed, device, cache)
        model = Classifier(graph.x.shape[1], graph.classes, cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=0)
        train = torch.from_numpy(graph.masks['train'][split]).to(device)
        val = torch.from_numpy(graph.masks['val'][split]).to(device)
        y_train = torch.from_numpy(graph.targets(split, 'train')).to(device)
        y_val = graph.targets(split, 'val')
        best, bad, first, history = -float('inf'), 0, 0, []
        last_path = directory / 'last.pt'
        if last_path.exists():
            state = load_torch(last_path, device)
            if state['identity'] != identity:
                raise ValueError('Checkpoint identity mismatch')
            model.load_state_dict(state['state'])
            optimizer.load_state_dict(state['optimizer'])
            restore_rng(state['rng'])
            best, bad, first = state['best'], state['bad'], state['epoch'] + 1
            history = state['history']
        metadata = dict(identity=identity, dataset=graph.name, dataset_hash=graph.sha256,
                        config=cfg, split=split, seed=seed, environment=environment(device))
        write_json(directory / 'metadata.json', metadata)
        for epoch in range(first, plan['epochs']):
            if bad >= plan['patience']:
                break
            synchronize(device)
            started = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(data)
            loss = node_loss(logits[train], y_train).mean()
            if cfg['l2']:
                loss = loss + cfg['l2'] * sum(p.square().sum() for p in model.parameters())
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                logits, _ = model(data)
                p = probabilities(logits[val]).cpu().numpy()
            score = metrics(y_val, p)[graph.metric]
            if score is None or not np.isfinite(score):
                raise ValueError('Validation metric is undefined')
            synchronize(device)
            elapsed = time.perf_counter()-started
            improved = score > best
            if improved:
                best, bad = score, 0
                save_torch(directory / 'best.pt', dict(metadata, state=model.state_dict(),
                           epoch=epoch, validation=score))
            else:
                bad += 1
            history.append(dict(epoch=epoch, train_loss=float(loss.detach()), validation=score,
                                compute_seconds=elapsed))
            # Last state supports exact epoch-boundary resume on the same environment.
            save_torch(last_path, dict(identity=identity, state=model.state_dict(),
                       optimizer=optimizer.state_dict(), rng=rng_state(), best=best,
                       bad=bad, epoch=epoch, history=history))
            if epoch % 25 == 0:
                print(f'{graph.name}/{cfg["model"]} split={split} epoch={epoch} val={score:.4f}', flush=True)
        best_ckpt = load_torch(directory / 'best.pt', 'cpu')
        result = dict(metadata, best_validation=best, best_epoch=best_ckpt['epoch'],
                      epochs=len(history), history=history, preprocessing=preprocessing,
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                      compute_seconds=sum(row['compute_seconds'] for row in history),
                      sampled_peak_rss_bytes=memory.peak,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)
        write_json(result_path, result)
    del data, model, optimizer
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def predict(graph, checkpoint_path, device, cache, inspect_node=None):
    checkpoint_data = load_torch(checkpoint_path, device)
    if checkpoint_data['dataset_hash'] != graph.sha256:
        raise ValueError('Checkpoint and dataset hash differ')
    cfg = checkpoint_data['config']
    data, _ = prepare_run(graph, cfg, checkpoint_data['seed'], device, cache)
    model = Classifier(graph.x.shape[1], graph.classes, cfg).to(device)
    model.load_state_dict(checkpoint_data['state'])
    model.eval()
    with torch.no_grad():
        logits, extra = model(data, inspect_node=inspect_node)
        p = probabilities(logits).cpu().numpy()
    return p, extra, data, checkpoint_data


def evaluate(graph, checkpoint_path, device, cache, output):
    p, extra, _, ckpt = predict(graph, checkpoint_path, device, cache)
    split = ckpt['split']
    mask = graph.masks['test'][split]
    result = metrics(graph.targets(split, 'test'), p[mask])
    payload = dict(probabilities=p, test_mask=mask)
    if 'alpha' in extra:
        payload['alpha'] = extra['alpha'].cpu().numpy()
    np.savez_compressed(output, **payload)
    return result
