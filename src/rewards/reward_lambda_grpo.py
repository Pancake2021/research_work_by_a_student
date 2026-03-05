"""
reward_lambda_grpo.py — RF5: λ-GRPO взвешенная награда по длине ответа
=======================================================================

Мотивация: модель должна давать развёрнутые, но не избыточные ответы.
Слишком короткие ответы — поверхностны. Слишком длинные — шум.

Алгоритм (из тех-плана):
  length_penalty = min(len(response.split()) / TARGET_WORDS, 1.0)
  reward = base_accuracy × length_penalty

Таким образом:
  - Ответ < TARGET_WORDS слов → reward масштабируется вниз
  - Ответ ≥ TARGET_WORDS слов → полная награда (если ответ правильный)

Диапазон: [0.0, 1.0]
"""

from src.data.preprocessor import parse_label


# Настройки
TARGET_WORDS    = 50     # нормальная длина ответа в словах (из тех-плана)
BASE_REWARD     = 1.0    # максимальная награда за правильный ответ
WRONG_PENALTY   = 0.0    # награда за неправильный ответ


def reward_length_weighted(
    prompts: list[str],
    completions: list[str],
    true_labels: list[str],
    target_words: int = TARGET_WORDS,
    **kwargs,
) -> list[float]:
    """
    RF5: λ-GRPO — accuracy × length_penalty.

    Правильный, но слишком короткий ответ → пониженная награда.
    Полная награда только при правильном И развёрнутом ответе.

    Args:
        prompts:      Список промптов
        completions:  Список ответов модели
        true_labels:  Эталонные метки
        target_words: Целевая длина ответа в словах

    Returns:
        Список наград в [0.0, 1.0]
    """
    rewards = []
    for completion, true_label in zip(completions, true_labels):
        predicted = parse_label(completion)

        if predicted != true_label:
            # Неправильный ответ — базовый штраф независимо от длины
            rewards.append(WRONG_PENALTY)
            continue

        # Длинное-нормированное вознаграждение
        length_penalty = _compute_length_penalty(completion, target_words)
        rewards.append(BASE_REWARD * length_penalty)

    return rewards


def _compute_length_penalty(text: str, target_words: int = TARGET_WORDS) -> float:
    """
    Вычисляет нормированный штраф за длину.

    Returns:
        float в [0.0, 1.0]:
          0.0  → пустой ответ
          0.5  → ответ в два раза короче target
          1.0  → ответ достиг или превысил target
    """
    word_count = len(text.split())
    if word_count == 0:
        return 0.0
    return min(word_count / target_words, 1.0)
