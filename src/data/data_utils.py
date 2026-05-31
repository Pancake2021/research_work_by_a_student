"""Common utilities used across training/evaluation scripts."""

from __future__ import annotations

import json
import logging
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger("diploma_rl")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
logger.propagate = False


def seed_everything(seed: int = 42) -> None:
    """Seed python/numpy/torch when available for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # torch is optional for lightweight tasks (e.g. parsing/help commands)
        pass

    logger.info("Seed fixed: %s", seed)


def check_gpu() -> dict[str, Any]:
    """Return runtime GPU info without crashing on CPU-only environments."""
    info: dict[str, Any] = {
        "available": False,
        "device": "cpu",
        "name": None,
        "total_memory_gb": None,
    }

    try:
        import torch

        info["available"] = bool(torch.cuda.is_available())
        if info["available"]:
            device_idx = 0
            props = torch.cuda.get_device_properties(device_idx)
            info["device"] = f"cuda:{device_idx}"
            info["name"] = props.name
            info["total_memory_gb"] = round(props.total_memory / (1024**3), 2)
            logger.info("GPU detected: %s (%s GB)", info["name"], info["total_memory_gb"])
        else:
            logger.warning("CUDA is not available, running on CPU")
    except Exception as exc:
        logger.warning("GPU check skipped: %s", exc)

    return info


def save_results_json(payload: dict[str, Any], save_path: str) -> None:
    """Save a dict as UTF-8 JSON with parent directory creation."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("Results saved: %s", path)


def load_results_json(path: str) -> dict[str, Any] | None:
    """Load a UTF-8 JSON payload if it exists."""
    path_obj = Path(path)
    if not path_obj.exists():
        logger.warning("File not found: %s", path_obj)
        return None
    with path_obj.open("r", encoding="utf-8") as f:
        return json.load(f)


def balance_dataset(dataset, label_field: str = "label", seed: int = 42):
    """Undersample a HuggingFace Dataset to the smallest class size."""
    labels = dataset[label_field]
    counter = Counter(labels)
    min_count = min(counter.values())
    indices_by_label: dict[str, list[int]] = {}
    for idx, label in enumerate(labels):
        indices_by_label.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    selected_indices = []
    for indices in indices_by_label.values():
        rng.shuffle(indices)
        selected_indices.extend(indices[:min_count])
    rng.shuffle(selected_indices)
    return dataset.select(selected_indices)


def dataset_stats(dataset, label_field: str = "label") -> dict[str, Any]:
    """Return basic label and text-length statistics for list/HF datasets."""
    labels = dataset[label_field] if hasattr(dataset, "__getitem__") else [row[label_field] for row in dataset]
    try:
        texts = dataset["text"]
    except Exception:
        try:
            texts = dataset["prompt"]
        except Exception:
            texts = [row.get("text", row.get("prompt", "")) for row in dataset]
    lengths = [len(str(text).split()) for text in texts]
    counter = Counter(labels)
    total = len(labels) or 1
    return {
        "total": len(labels),
        "label_distribution": {key: f"{value} ({value / total:.1%})" for key, value in counter.items()},
        "avg_text_length_words": round(float(np.mean(lengths)), 1) if lengths else 0,
        "max_text_length_words": max(lengths) if lengths else 0,
        "min_text_length_words": min(lengths) if lengths else 0,
    }
