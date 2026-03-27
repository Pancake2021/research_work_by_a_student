"""
baseline_eval.py
================
Замер baseline метрик до любого RL-обучения.

Это критически важная точка отсчёта для всего исследования —
запускать ОБЯЗАТЕЛЬНО перед первым обучением.
"""

import time
from typing import List, Optional, Dict, Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from datasets import Dataset
except ImportError:  # pragma: no cover
    Dataset = Any

from src.data.preprocessor import parse_label, build_chat_prompt
from src.data.data_utils import logger, save_results_json


# ──────────────────────────────────────────────────────────────────────────────
# Инференс
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(
    model,
    tokenizer,
    examples: List[dict],
    max_new_tokens: int = 128,
    temperature: float = 0.1,
    batch_size: int = 4,
) -> List[Optional[str]]:
    """
    Прогоняет список примеров через модель и возвращает ответы.

    Args:
        model:          Обученная или baseline модель
        tokenizer:      Токенизатор
        examples:       Список dict с ключом 'text'
        max_new_tokens: Максимум новых токенов
        temperature:    Температура генерации (низкая = детерминированная)
        batch_size:     Размер батча инференса

    Returns:
        Список сырых строк — ответов модели (или None при ошибке)
    """
    if torch is None:
        raise ImportError("torch is required for inference. Install project dependencies.")

    model.eval()
    responses = []

    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        prompts = [build_chat_prompt(ex["text"], tokenizer) for ex in batch]

        with torch.no_grad():
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0,
                pad_token_id=tokenizer.pad_token_id,
            )

        for idx, output in enumerate(outputs):
            # Use true prompt length for each sample (important with padding).
            input_len = int(inputs["attention_mask"][idx].sum().item())
            new_tokens = output[input_len:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
            responses.append(decoded)

        if (i // batch_size) % 5 == 0:
            logger.info(f"  Инференс: {min(i+batch_size, len(examples))}/{len(examples)}")

    return responses


# ──────────────────────────────────────────────────────────────────────────────
# Оценка метрик
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_predictions(
    true_labels: List[str],
    predicted_labels: List[Optional[str]],
    label_names: List[str] = ["positive", "negative"],
) -> Dict[str, Any]:
    """
    Вычисляет метрики качества классификации.

    Returns:
        dict с accuracy, f1_weighted, f1_macro, classification_report
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        classification_report,
    )

    # None → "unknown" для корректного подсчёта
    preds_clean = [p if p in label_names else "unknown" for p in predicted_labels]

    accuracy  = accuracy_score(true_labels, preds_clean)
    f1_w      = f1_score(true_labels, preds_clean, average="weighted", zero_division=0)
    f1_macro  = f1_score(true_labels, preds_clean, average="macro",    zero_division=0)
    none_rate = sum(1 for p in predicted_labels if p is None) / len(predicted_labels)

    report = classification_report(
        true_labels, preds_clean,
        labels=label_names,
        zero_division=0,
    )

    metrics = {
        "accuracy":      round(accuracy, 4),
        "f1_weighted":   round(f1_w, 4),
        "f1_macro":      round(f1_macro, 4),
        "none_rate":     round(none_rate, 4),   # доля неразобранных ответов
        "n_samples":     len(true_labels),
        "report":        report,
    }

    logger.info("=== Метрики ===")
    logger.info(f"  Accuracy:    {accuracy:.4f}")
    logger.info(f"  F1 weighted: {f1_w:.4f}")
    logger.info(f"  F1 macro:    {f1_macro:.4f}")
    logger.info(f"  None rate:   {none_rate:.4f}  (доля нечитаемых ответов)")
    logger.info(f"\n{report}")

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Главная функция baseline
# ──────────────────────────────────────────────────────────────────────────────

def run_baseline_evaluation(
    model,
    tokenizer,
    test_dataset: Dataset,
    save_path: Optional[str] = None,
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    """
    Полный цикл baseline评估:
    1. Инференс на тестовом датасете
    2. Парсинг ответов
    3. Вычисление метрик
    4. Сохранение результатов

    Returns:
        Словарь с метриками
    """
    logger.info("=" * 50)
    logger.info("BASELINE EVALUATION — запускается до обучения")
    logger.info("=" * 50)

    examples = [{"text": ex["text"]} for ex in test_dataset]
    true_labels = [ex["label"] for ex in test_dataset]

    t0 = time.time()
    raw_responses = run_inference(model, tokenizer, examples, max_new_tokens=max_new_tokens)
    inference_time = round(time.time() - t0, 2)

    predicted_labels = [parse_label(r) for r in raw_responses]

    metrics = evaluate_predictions(true_labels, predicted_labels)
    metrics["inference_time_sec"] = inference_time
    metrics["method"] = "baseline_no_rl"

    # Примеры ответов для анализа
    metrics["sample_outputs"] = [
        {
            "text":      examples[i]["text"][:100],
            "true":      true_labels[i],
            "predicted": predicted_labels[i],
            "raw":       raw_responses[i][:200],
        }
        for i in range(min(5, len(examples)))
    ]

    if save_path:
        save_results_json(metrics, save_path)

    logger.info(f"\n{'='*50}")
    logger.info(f"⭐ BASELINE F1 (weighted): {metrics['f1_weighted']:.4f}")
    logger.info(f"   Запиши это число! Это точка отсчёта для RL-обучения.")
    logger.info(f"{'='*50}")

    return metrics
