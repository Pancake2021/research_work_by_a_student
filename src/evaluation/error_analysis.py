"""
error_analysis.py
=================
Анализ ошибок модели — самая интересная часть диплома.

Что анализируем:
  1. Типы ошибок: false positive vs false negative
  2. Длина текста и точность
  3. Примеры с наибольшим confidence-мismatch
  4. Кластеризация ошибочных примеров
"""

from typing import List, Optional, Dict, Any
from collections import Counter, defaultdict

from src.data.data_utils import logger


# ──────────────────────────────────────────────────────────────────────────────
# Анализ ошибок
# ──────────────────────────────────────────────────────────────────────────────

def analyze_errors(
    examples: List[Dict],
    true_labels: List[str],
    predicted_labels: List[Optional[str]],
    method_name: str = "unknown",
    n_examples: int = 10,
) -> Dict[str, Any]:
    """
    Анализирует ошибки модели.

    Args:
        examples:        Список примеров с полем 'text'
        true_labels:     Эталонные метки
        predicted_labels: Предсказанные метки (может содержать None)
        method_name:     Название метода
        n_examples:      Число примеров для вывода

    Returns:
        dict с анализом ошибок
    """
    errors = []
    correct = []

    for i, (ex, pred, true) in enumerate(zip(examples, predicted_labels, true_labels)):
        is_correct = pred == true
        record = {
            "idx":        i,
            "text":       ex.get("text", "")[:200],
            "true":       true,
            "predicted":  pred,
            "text_len":   len(ex.get("text", "").split()),
            "is_correct": is_correct,
        }
        if is_correct:
            correct.append(record)
        else:
            errors.append(record)

    # Типы ошибок
    error_types = Counter()
    for e in errors:
        if e["predicted"] is None:
            error_types["no_answer"] += 1
        elif e["true"] == "positive" and e["predicted"] == "negative":
            error_types["false_negative"] += 1
        elif e["true"] == "negative" and e["predicted"] == "positive":
            error_types["false_positive"] += 1

    # Анализ по длине текста
    len_bins = {"short (< 10)": [], "medium (10-50)": [], "long (> 50)": []}
    for r in [*errors, *correct]:
        if r["text_len"] < 10:
            len_bins["short (< 10)"].append(r["is_correct"])
        elif r["text_len"] <= 50:
            len_bins["medium (10-50)"].append(r["is_correct"])
        else:
            len_bins["long (> 50)"].append(r["is_correct"])

    len_accuracy = {}
    for bin_name, is_correct_list in len_bins.items():
        if is_correct_list:
            len_accuracy[bin_name] = round(sum(is_correct_list) / len(is_correct_list), 3)

    analysis = {
        "method":          method_name,
        "total":           len(examples),
        "n_errors":        len(errors),
        "n_correct":       len(correct),
        "error_rate":      round(len(errors) / len(examples), 4),
        "error_types":     dict(error_types),
        "accuracy_by_len": len_accuracy,
        "sample_errors":   errors[:n_examples],
    }

    # Логирование
    logger.info(f"\n=== Анализ ошибок: {method_name} ===")
    logger.info(f"  Всего ошибок: {len(errors)}/{len(examples)} ({analysis['error_rate']:.1%})")
    logger.info(f"  Типы ошибок: {dict(error_types)}")
    logger.info(f"  Точность по длине текста: {len_accuracy}")
    logger.info(f"\n  Примеры ошибок (первые {min(5, len(errors))}):")
    for err in errors[:5]:
        logger.info(f"    [{err['true']} → {err['predicted']}] {err['text'][:80]}...")

    return analysis
