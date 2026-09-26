"""JSON / npz helpers: numpy-aware JSON, atomic writes, and a small on-disk cache."""
import os
import json
import math

import numpy as np


def to_jsonable(obj):
    """numpy scalars/arrays -> Python types, NaN/inf -> None (so the output is strict JSON)."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path, obj, indent=None):
    """Write atomically (temp file + rename) so a killed job never leaves a truncated file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(to_jsonable(obj), f, indent=indent)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def cached_json(path, compute):
    """compute() once and store it at `path`; later calls read it back."""
    if os.path.exists(path):
        return read_json(path)
    result = compute()
    write_json(path, result)
    return read_json(path)  # same types on a cache hit and a miss


def cached_npz(path, compute):
    """compute() -> {name: array} once and store it at `path`; later calls read it back."""
    if os.path.exists(path):
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    arrays = compute()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return arrays
