import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device="auto"):
    return (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )


def load_config(path):
    with open(path, encoding="utf-8") as f:
        result = yaml.safe_load(f)
    if not isinstance(result, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return result


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def resource_metrics(device):
    import platform

    import psutil

    info = psutil.Process().memory_info()
    try:
        import resource

        divisor = 2**20 if platform.system() == "Darwin" else 1024
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor
    except ImportError:  # Windows peak working set, exposed by psutil.
        peak_mb = getattr(info, "peak_wset", info.rss) / 2**20
    return dict(
        ram_mb=info.rss / 2**20,
        peak_process_ram_mb=peak_mb,
        cuda_peak_mb=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
    )


class MemoryMonitor:
    """Sample process RSS at low frequency; sampling may miss brief transient peaks."""

    def __init__(self, interval=0.25):
        import threading

        import psutil

        self.interval = interval
        self.process = psutil.Process()
        self.peak_mb = 0.0
        self.event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        self.peak_mb = max(self.peak_mb, self.process.memory_info().rss / 2**20)

    def _run(self):
        while not self.event.wait(self.interval):
            self._sample()

    def __enter__(self):
        self._sample()
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.event.set()
        self.thread.join()
        self._sample()
