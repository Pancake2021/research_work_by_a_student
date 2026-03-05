"""
reward_accuracy.py — RF1: Простая бинарная accuracy reward
===========================================================

Самый простой вариант: 1.0 если ответ правильный, 0.0 иначе.
Используется как baseline reward для сравнения.

Диапазон: {0.0, 1.0}
"""

from src.data.preprocessor import parse_label


def reward_accuracy(
    prompts: list[str],
    completions: list[str],
    true_labels: list[str],
    **kwargs,
) -> list[float]:
    """
    RF1: Accuracy reward — правильно / неправильно.

    Сигнатура соответствует TRL GRPOTrainer reward_funcs:
      f(prompts, completions, **kwargs) -> list[float]

    Метки передаются через kwargs['true_labels'] или напрямую.

    Args:
        prompts:     Список промптов (батч)
        completions: Список ответов модели (батч)
        true_labels: Список эталонных меток ['positive'|'negative']

    Returns:
        Список наград — {0.0, 1.0}
    """
    rewards = []
    for completion, true_label in zip(completions, true_labels):
        predicted = parse_label(completion)
        reward = 1.0 if predicted == true_label else 0.0
        rewards.append(reward)
    return rewards
