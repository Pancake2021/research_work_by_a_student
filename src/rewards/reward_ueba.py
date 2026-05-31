"""
reward_ueba.py
==============
Reward-функции для UEBA/insider-threat GRPO экспериментов.
"""

from __future__ import annotations

from typing import Any

from src.data.preprocessor import UEBA_LABELS, parse_evidence, parse_label, parse_reasoning
from src.evaluation.ueba_metrics import evidence_overlap


def reward_ueba_accuracy(
    prompts: list[str],
    completions: list[str],
    true_labels: list[str] | None = None,
    **kwargs,
) -> list[float]:
    """RF1: награда только за правильный risk label."""
    labels = _resolve_labels(true_labels, kwargs)
    return [1.0 if parse_label(completion) == label else 0.0 for completion, label in zip(completions, labels)]


def reward_ueba_format(
    prompts: list[str],
    completions: list[str],
    true_labels: list[str] | None = None,
    **kwargs,
) -> list[float]:
    """RF2: класс + строгий SOC-формат."""
    labels = _resolve_labels(true_labels, kwargs)
    rewards = []
    for completion, label in zip(completions, labels):
        risk = parse_label(completion)
        reward = 1.0 if risk == label else 0.0
        if risk in UEBA_LABELS:
            reward += 0.1
        if parse_evidence(completion):
            reward += 0.1
        if parse_reasoning(completion):
            reward += 0.1
        rewards.append(reward)
    return rewards


def reward_ueba_evidence(
    prompts: list[str],
    completions: list[str],
    true_labels: list[str] | None = None,
    evidence_labels: list[list[str]] | None = None,
    **kwargs,
) -> list[float]:
    """RF3: класс + формат + совпадение evidence."""
    labels = _resolve_labels(true_labels, kwargs)
    evidences = evidence_labels or kwargs.get("evidence") or kwargs.get("evidence_labels") or [[] for _ in labels]
    rewards = []
    for completion, label, true_evidence in zip(completions, labels, evidences):
        risk = parse_label(completion)
        predicted_evidence = parse_evidence(completion)
        reward = 1.0 if risk == label else 0.0
        reward += 0.2 if risk in UEBA_LABELS else -0.2
        reward += 0.2 if parse_reasoning(completion) else 0.0
        reward += 0.4 * evidence_overlap(true_evidence, predicted_evidence)
        reward -= hallucination_penalty(true_evidence, predicted_evidence)
        rewards.append(round(max(-0.5, reward), 4))
    return rewards


def hallucination_penalty(true_evidence: list[str] | str, predicted_evidence: list[str]) -> float:
    """Штрафует длинный список признаков, не подтвержденных source evidence."""
    if isinstance(true_evidence, str):
        true_items = [true_evidence]
    else:
        true_items = list(true_evidence or [])
    if not predicted_evidence:
        return 0.2
    true_text = " ".join(true_items).lower()
    unsupported = 0
    for item in predicted_evidence:
        tokens = [token for token in item.lower().split() if len(token) >= 4]
        if tokens and not any(token in true_text for token in tokens):
            unsupported += 1
    return min(0.3, unsupported * 0.1)


def get_ueba_reward(name: str):
    registry = {
        "ueba_accuracy": reward_ueba_accuracy,
        "ueba_format": reward_ueba_format,
        "ueba_evidence": reward_ueba_evidence,
    }
    if name not in registry:
        raise ValueError(f"Unknown UEBA reward: {name}. Available: {list(registry)}")
    return registry[name]


def _resolve_labels(true_labels: list[str] | None, kwargs: dict[str, Any]) -> list[str]:
    labels = true_labels or kwargs.get("risk_label") or kwargs.get("label") or kwargs.get("true_labels")
    if labels is None:
        raise ValueError("UEBA rewards require true_labels, risk_label, or label")
    return list(labels)
