"""Common utilities used across training/evaluation scripts."""

from __future__ import annotations

import json
import logging
import os
import random
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
            info["total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
            logger.info(
                "GPU detected: %s (%s GB)",
                info["name"],
                info["total_memory_gb"],
            )
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
