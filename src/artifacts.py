#!/usr/bin/env python3
"""
On-disk artifact contract shared by the attack driver and the renderer.

src/attack.py writes a run's numeric/image artifacts with the write_* helpers
src/visualisations.py reads them back with the read_* helpers

"""
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def read_epoch_study_csv(csv_path) -> List[Dict[str, float]]:
    """Load an epoch_study/{init}_{model}.csv (written by attack.run_epoch_study)
    into numeric row dicts; blank/None cells become NaN."""
    rows: List[Dict[str, float]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out: Dict[str, float] = {}
            for k, v in r.items():
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = float("nan")
            rows.append(out)
    return rows

def write_rows_bundle(npz_path: Path, json_path: Path, rows: List[Dict]) -> None:
    """Persist a list of example-row dicts. Each value is either a numpy image
    array (-> compressed .npz, keyed ``ex{i}__{key}``) or a JSON scalar / small
    metric dict (-> parallel .json), so the exact rows can be rebuilt for
    save_examples without re-running the attack."""
    arrays: Dict[str, np.ndarray] = {}
    meta: List[Dict] = []
    for i, row in enumerate(rows):
        scalars: Dict = {}
        for k, v in row.items():
            if isinstance(v, np.ndarray):
                arrays[f"ex{i}__{k}"] = v
            else:
                scalars[k] = v
        meta.append(scalars)
    np.savez_compressed(npz_path, **arrays)
    Path(json_path).write_text(json.dumps(meta, default=float), encoding="utf-8")

def read_rows_bundle(npz_path: Path, json_path: Path) -> List[Dict]:
    """Inverse of write_rows_bundle: reassemble the example-row dicts."""
    meta = json.loads(Path(json_path).read_text(encoding="utf-8"))
    rows: List[Dict] = [dict(m) for m in meta]
    if Path(npz_path).exists():
        with np.load(npz_path) as z:
            for key in z.files:
                idx_str, k = key[2:].split("__", 1)  # drop the 'ex' prefix
                rows[int(idx_str)][k] = z[key]
    return rows

def read_metric_rows(csv_path: Path) -> List[Dict[str, float]]:
    """Load per_sample_metrics.csv back into the list-of-float-dicts shape the
    aggregate figure functions expect (every column is numeric)."""
    rows: List[Dict[str, float]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    return rows

def write_transfer_bundle(npz_path: Path, json_path: Path, *,
                          model_names: List[str], attack_name: str, eps: float,
                          T: int, n_ex: int, gt_stack: np.ndarray,
                          recon: Dict[str, np.ndarray]) -> None:
    """Persist the cross-model transfer image stacks for one attack.

    ``gt_stack`` is [K, H, W]; ``recon`` maps an archive key to a [K, H, W]
    stack, keyed ``clean__{model}`` and ``pred__{src}__{tgt}``."""
    np.savez_compressed(npz_path, gt=gt_stack, **recon)
    Path(json_path).write_text(json.dumps({
        "model_names": model_names, "attack_name": attack_name,
        "eps": eps, "T": int(T), "n_ex": int(n_ex),
    }), encoding="utf-8")

def read_transfer_bundle(npz_path: Path, json_path: Path):
    meta = json.loads(Path(json_path).read_text(encoding="utf-8"))
    with np.load(npz_path) as z:
        data = {k: z[k] for k in z.files}
    return meta, data
