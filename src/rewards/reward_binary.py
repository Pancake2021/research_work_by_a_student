"""
reward_binary.py — RF3: Бинарная reward со штрафами (DeepSeek-R1 стиль)
=========================================================================

Мотивация из DeepSeek-R1 paper (arxiv.org/abs/2501.12948):
  Бинарная reward (правильно/неправильно) часто лучше дробных значений.
  Добавляем штраф за "нечитаемый" ответ (модель не дала метку).

Алгоритм:
   Правильно → 0.0   (baseline, не наказываем и не поощряем сверх меры)
   Нет метки → -0.5  (штраф: модель уклонилась от ответа)
   Неверно   → -1.0  (штраф: уверенная ошибка)

Диапазон: {-1.0, -0.5, 0.0}

Примечание: используй эту RF когда хочешь сравнить с DeepSeek-R1 подходом.
"""

from src.data.preprocessor import parse_label


# Настраиваемые параметры (из тех-плана)
REWARD_CORRECT    =  0.0   # правильный ответ
REWARD_NO_ANSWER  = -0.5   # ответ не распознан (нет метки)
REWARD_WRONG      = -1.0   # неправильная метка


def reward_binary(
    prompts: list[str],
    completions: list[str],
    true_labels: list[str],
    **kwargs,
) -> list[float]:
    """
    RF3: Бинарная reward со штрафами в стиле DeepSeek-R1.

    Args:
        prompts:     Список промптов
        completions: Список ответов модели
        true_labels: Эталонные метки

    Returns:
        Список наград: {-1.0, -0.5, 0.0}
    """
    rewards = []
    for completion, true_label in zip(completions, true_labels):
        predicted = parse_label(completion)

        if predicted is None:
            reward = REWARD_NO_ANSWER   # модель не дала ответа
        elif predicted == true_label:
            reward = REWARD_CORRECT     # правильно
        else:
            reward = REWARD_WRONG       # неправильно

        rewards.append(reward)

    return rewards
