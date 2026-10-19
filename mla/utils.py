import hashlib
import importlib.metadata
import json
import os
import platform
import random
import threading
from pathlib import Path

import numpy as np
import psutil
import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def code_hash():
    root = Path(__file__).parent
    return digest({p.name: file_hash(p) for p in sorted(root.glob('*.py'))})


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def load_torch(path, device='cpu'):
    # Checkpoints contain RNG/optimizer state: load only checkpoints you trust.
    return torch.load(path, map_location=device, weights_only=False)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def environment(device):
    return dict(python=platform.python_version(), platform=platform.platform(),
                torch=torch.__version__, numpy=np.__version__, device=str(device),
                gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                ram_bytes=psutil.virtual_memory().total, code_hash=code_hash(),
                packages={name: importlib.metadata.version(name)
                          for name in ['scipy', 'scikit-learn', 'psutil']})


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


class MemoryMonitor:
    def __enter__(self):
        self.stop = threading.Event()
        self.peak = psutil.Process().memory_info().rss
        def sample():
            while not self.stop.wait(0.05):
                self.peak = max(self.peak, psutil.Process().memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()

