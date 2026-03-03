from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import torch


def discover_metric_files(root: Path, include_all_csv: bool = False) -> list[Path]:
    """Return sorted metric file candidates under ``root``.

    If ``include_all_csv`` is True, fallback to all CSV files when ``metrics.csv``
    is not found.
    """
    metrics_files = [Path(p) for p in glob.glob(str(root / "**/metrics.csv"), recursive=True)]
    unique_metrics = sorted(set(metrics_files))
    if unique_metrics:
        return unique_metrics

    if not include_all_csv:
        return []

    csv_files = [Path(p) for p in glob.glob(str(root / "**/*.csv"), recursive=True)]
    return sorted(set(csv_files))


def load_torch_checkpoint(ckpt_path: Path) -> dict[str, Any]:
    """Load a Lightning checkpoint across torch versions.

    PyTorch 2.6 changed ``torch.load`` default behavior around ``weights_only``.
    This helper preserves compatibility with both old and new versions.
    """
    try:
        checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(ckpt_path), map_location="cpu")

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint format is invalid: expected a mapping object.")
    return checkpoint
