"""Lightweight data package exports.

Heavy dataset dependencies are imported lazily so prompt/reward utilities can be
tested without installing the full ML stack.
"""

from src.data.preprocessor import (
    UEBA_LABELS,
    build_chat_prompt,
    build_prompt,
    build_response,
    build_ueba_prompt,
    build_ueba_response,
    format_example,
    normalize_label,
    parse_evidence,
    parse_label,
    parse_reasoning,
)

__all__ = [
    "UEBA_LABELS",
    "build_chat_prompt",
    "build_prompt",
    "build_response",
    "build_ueba_prompt",
    "build_ueba_response",
    "check_gpu",
    "format_example",
    "load_behavior_dataset",
    "logger",
    "normalize_label",
    "parse_evidence",
    "parse_label",
    "parse_reasoning",
    "save_results_json",
    "seed_everything",
]


def __getattr__(name):
    if name == "load_behavior_dataset":
        from src.data.dataset_loader import load_behavior_dataset

        return load_behavior_dataset
    if name in {"logger", "seed_everything", "check_gpu", "save_results_json"}:
        from src.data import data_utils

        return getattr(data_utils, name)
    raise AttributeError(f"module 'src.data' has no attribute {name!r}")
