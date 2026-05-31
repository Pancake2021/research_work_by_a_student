"""
ueba_metrics.py
===============
Метрики для UEBA/insider-threat сценариев.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.data.preprocessor import UEBA_LABELS, parse_evidence, parse_label, parse_reasoning


def evaluate_ueba_predictions(
    examples: list[dict[str, Any]],
    responses: list[str],
) -> dict[str, Any]:
    """Считает качество класса, формата и evidence для UEBA-ответов."""
    true_labels = [example.get("risk_label") or example.get("label") for example in examples]
    pred_labels = [parse_label(response) for response in responses]
    labels = ["normal", "suspicious", "malicious"]
    confusion = _confusion_matrix(true_labels, pred_labels, labels)
    per_class = {label: _class_scores(confusion, label, labels) for label in labels}
    n = len(true_labels) or 1
    correct = sum(1 for true, pred in zip(true_labels, pred_labels) if true == pred)
    supports = Counter(true_labels)
    macro_f1 = sum(per_class[label]["f1"] for label in labels) / len(labels)
    weighted_f1 = sum(per_class[label]["f1"] * supports.get(label, 0) for label in labels) / n
    format_flags = [is_valid_ueba_format(response) for response in responses]
    evidence_scores = [
        evidence_overlap(example.get("evidence", []), parse_evidence(response))
        for example, response in zip(examples, responses)
    ]
    return {
        "accuracy": round(correct / n, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "recall_malicious": round(per_class["malicious"]["recall"], 4),
        "false_positive_rate": round(_malicious_false_positive_rate(confusion), 4),
        "valid_format_rate": round(sum(format_flags) / n, 4),
        "evidence_hit_rate": round(sum(evidence_scores) / n, 4),
        "avg_response_length": round(sum(len(str(r).split()) for r in responses) / n, 2),
        "n_samples": len(true_labels),
        "confusion": confusion,
        "per_class": per_class,
    }


def is_valid_ueba_format(response: str) -> bool:
    """Проверяет обязательные поля SOC-ответа."""
    risk = parse_label(response)
    evidence = parse_evidence(response)
    reasoning = parse_reasoning(response)
    return risk in UEBA_LABELS and len(evidence) >= 1 and bool(reasoning)


def evidence_overlap(true_evidence: list[str] | str, predicted_evidence: list[str]) -> float:
    """Доля эталонных evidence items, найденных в ответе."""
    if isinstance(true_evidence, str):
        true_items = [true_evidence]
    else:
        true_items = list(true_evidence or [])
    if not true_items:
        return 1.0 if not predicted_evidence else 0.5
    pred_text = " ".join(predicted_evidence).lower()
    hits = 0
    for item in true_items:
        tokens = [token for token in str(item).lower().split() if len(token) >= 4]
        if tokens and any(token in pred_text for token in tokens):
            hits += 1
    return hits / len(true_items)


def _confusion_matrix(true_labels, pred_labels, labels):
    matrix = {true: {pred: 0 for pred in labels + ["unknown"]} for true in labels}
    for true, pred in zip(true_labels, pred_labels):
        true = true if true in labels else "normal"
        pred = pred if pred in labels else "unknown"
        matrix[true][pred] += 1
    return matrix


def _class_scores(confusion, label, labels):
    tp = confusion[label][label]
    fp = sum(confusion[other][label] for other in labels if other != label)
    fn = sum(count for pred, count in confusion[label].items() if pred != label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": sum(confusion[label].values()),
    }


def _malicious_false_positive_rate(confusion):
    non_malicious_total = sum(sum(confusion[label].values()) for label in ("normal", "suspicious"))
    false_malicious = confusion["normal"]["malicious"] + confusion["suspicious"]["malicious"]
    return false_malicious / non_malicious_total if non_malicious_total else 0.0
