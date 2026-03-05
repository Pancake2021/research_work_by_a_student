"""
reward_entropy.py — RF4: Accuracy + entropy bonus (для DAPO / борьба с entropy collapse)
========================================================================================

Проблема: в базовом GRPO энтропия политики падает в ноль —
модель перестаёт исследовать разные ответы (entropy collapse).

Решение: добавляем бонус за разнообразие токенов в ответе.
Метрика разнообразия: unique_token_ratio = |unique_tokens| / |all_tokens|

Алгоритм:
  reward = base_accuracy + α × unique_token_ratio

Диапазон: приблизительно [0.0, 1.0 + α]
"""

from src.data.preprocessor import parse_label


# Коэффициент энтропийного бонуса (настраивается как гиперпараметр)
ENTROPY_BONUS_WEIGHT = 0.1   # α: из тех-плана
BASE_REWARD_CORRECT  = 1.0
BASE_REWARD_WRONG    = 0.0


def reward_with_entropy_bonus(
    prompts: list[str],
    completions: list[str],
    true_labels: list[str],
    tokenizer=None,
    entropy_weight: float = ENTROPY_BONUS_WEIGHT,
    **kwargs,
) -> list[float]:
    """
    RF4: Accuracy + энтропийный бонус для предотвращения entropy collapse.

    Args:
        prompts:        Список промптов
        completions:    Список ответов модели
        true_labels:    Эталонные метки
        tokenizer:      Токенизатор (нужен для подсчёта уникальных токенов).
                        Если None — используем посимвольный fallback.
        entropy_weight: Вес энтропийного бонуса (α)

    Returns:
        Список наград ≈ [0.0, 1.0 + α]
    """
    rewards = []
    for completion, true_label in zip(completions, true_labels):
        predicted = parse_label(completion)
        base = BASE_REWARD_CORRECT if predicted == true_label else BASE_REWARD_WRONG

        # Вычисляем уникальность токенов
        unique_ratio = _compute_unique_ratio(completion, tokenizer)
        entropy_bonus = entropy_weight * unique_ratio

        rewards.append(base + entropy_bonus)

    return rewards


def _compute_unique_ratio(text: str, tokenizer=None) -> float:
    """
    Вычисляет долю уникальных токенов в тексте.

    Args:
        text:      Строка ответа модели
        tokenizer: Опциональный токенизатор HuggingFace

    Returns:
        float в [0.0, 1.0] — отношение уникальных токенов к общему числу
    """
    if not text:
        return 0.0

    if tokenizer is not None:
        try:
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not token_ids:
                return 0.0
            return len(set(token_ids)) / len(token_ids)
        except Exception:
            pass

    # Fallback: токенизация по пробелам
    words = text.split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)
