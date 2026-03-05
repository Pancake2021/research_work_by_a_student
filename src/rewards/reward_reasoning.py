"""
reward_reasoning.py — RF2: Accuracy + бонус за наличие рассуждения
===================================================================

Мотивация: модель должна не только дать правильный ответ,
но и обосновать его. Это повышает интерпретируемость.

Алгоритм:
  1. Базовая награда: 1.0 если метка верна, 0.0 иначе
  2. Бонус +0.2 если в ответе присутствует раздел "Анализ:"

Диапазон: {0.0, 1.0, 0.2, 1.2}
"""

from src.data.preprocessor import parse_label, parse_reasoning


# Настраиваемые параметры
BASE_REWARD_CORRECT   = 1.0
BASE_REWARD_INCORRECT = 0.0
REASONING_BONUS       = 0.2   # бонус за наличие рассуждения
MIN_REASONING_WORDS   = 3     # минимум слов в рассуждении для бонуса


def reward_with_reasoning(
    prompts: list[str],
    completions: list[str],
    true_labels: list[str],
    **kwargs,
) -> list[float]:
    """
    RF2: Accuracy + бонус за рассуждение.

    Args:
        prompts:     Список промптов
        completions: Список ответов модели
        true_labels: Эталонные метки

    Returns:
        Список наград в диапазоне {0.0, 0.2, 1.0, 1.2}
    """
    rewards = []
    for completion, true_label in zip(completions, true_labels):
        predicted = parse_label(completion)

        # 1. Базовая награда за правильный ответ
        base = BASE_REWARD_CORRECT if predicted == true_label else BASE_REWARD_INCORRECT

        # 2. Бонус за осмысленное рассуждение
        reasoning = parse_reasoning(completion)
        has_reasoning = (
            reasoning is not None
            and len(reasoning.split()) >= MIN_REASONING_WORDS
        )
        bonus = REASONING_BONUS if has_reasoning else 0.0

        rewards.append(base + bonus)

    return rewards
