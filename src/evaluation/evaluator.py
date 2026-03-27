"""
evaluator.py
============
Финальная оценка обученных моделей.
Сравнение всех методов (baseline, PPO, GRPO, DAPO, λ-GRPO) по единым метрикам.
"""

import os
import time
from typing import Dict, List, Optional, Any

try:
    from datasets import Dataset
except ImportError:  # pragma: no cover
    Dataset = Any

from src.data.preprocessor import parse_label
from src.data.data_utils import logger, save_results_json
from src.models.baseline_eval import run_inference, evaluate_predictions


# ──────────────────────────────────────────────────────────────────────────────
# Оценка одного метода
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_checkpoint(
    model,
    tokenizer,
    test_dataset: Dataset,
    method_name: str = "unknown",
    max_new_tokens: int = 128,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Оценивает модель на тестовом датасете.

    Args:
        model:         Обученная модель (LoRA + base)
        tokenizer:     Токенизатор
        test_dataset:  Тестовый датасет
        method_name:   Название метода (для логов и графиков)
        max_new_tokens: Макс. токенов генерации
        save_path:     Сохранить JSON результаты

    Returns:
        dict с метриками
    """
    logger.info(f"Оценка: {method_name}")

    examples = [{"text": ex["text"]} for ex in test_dataset]
    true_labels = [ex["label"] for ex in test_dataset]

    t0 = time.time()
    raw_responses = run_inference(model, tokenizer, examples, max_new_tokens=max_new_tokens)
    inference_time = round(time.time() - t0, 2)

    predicted_labels = [parse_label(r) for r in raw_responses]
    metrics = evaluate_predictions(true_labels, predicted_labels)
    metrics["method"] = method_name
    metrics["inference_time_sec"] = inference_time

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_results_json(metrics, save_path)

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Сравнение нескольких методов
# ──────────────────────────────────────────────────────────────────────────────

def compare_methods(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Создаёт сводную таблицу сравнения методов.

    Args:
        results: Список dict — результаты evaluate_checkpoint для каждого метода

    Returns:
        dict со сводной таблицей и лучшим методом
    """
    comparison = {
        "methods":  [],
        "f1":       [],
        "accuracy": [],
        "none_rate":[],
        "time":     [],
    }

    for r in results:
        comparison["methods"].append(r.get("method", "unknown"))
        comparison["f1"].append(r.get("f1_weighted", 0.0))
        comparison["accuracy"].append(r.get("accuracy", 0.0))
        comparison["none_rate"].append(r.get("none_rate", 0.0))
        comparison["time"].append(r.get("inference_time_sec", 0.0))

    best_idx = comparison["f1"].index(max(comparison["f1"]))
    comparison["best_method"] = comparison["methods"][best_idx]
    comparison["best_f1"]     = comparison["f1"][best_idx]

    logger.info("\n=== Сравнение методов ===")
    logger.info(f"{'Метод':<20} {'F1':>8} {'Accuracy':>10} {'None%':>8}")
    logger.info("-" * 50)
    for i, method in enumerate(comparison["methods"]):
        logger.info(
            f"{method:<20} {comparison['f1'][i]:>8.4f} "
            f"{comparison['accuracy'][i]:>10.4f} "
            f"{comparison['none_rate'][i]:>8.2%}"
        )
    logger.info(f"\n🏆 Лучший метод: {comparison['best_method']} (F1={comparison['best_f1']:.4f})")

    return comparison
