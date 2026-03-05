"""
dapo_trainer.py
===============
DAPO — модификация GRPO с entropy bonus и CLIP-higher для борьбы с entropy collapse.

Реализации в порядке убывания сложности:
  1. Entropy Bonus GRPO (этот файл) — самый простой и эффективный
  2. λ-GRPO (reward_lambda_grpo.py) — взвешенная награда по длине
  3. Полный DAPO (arxiv.org/abs/2503.14476) — clip_higher + token-level KL

Ссылки:
  - DAPO paper: https://arxiv.org/abs/2503.14476
  - GRPO paper:  https://arxiv.org/abs/2402.03300
"""

import os
from typing import Optional, Callable
from dataclasses import dataclass

from datasets import Dataset
from src.data.data_utils import logger, check_gpu
from src.training.grpo_trainer import GRPOTrainingConfig, _setup_wandb


# ──────────────────────────────────────────────────────────────────────────────
# Конфиг DAPO (наследует GRPO)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DAPOTrainingConfig(GRPOTrainingConfig):
    """
    DAPO конфиг — расширяет GRPO энтропийным бонусом.

    Дополнительные параметры:
        entropy_weight:  Вес энтропийного бонуса (α). 0.1 по умолчанию.
        clip_higher:     Clip только снизу (asymmetric clipping) — ключевая идея DAPO.
        use_token_kl:    Использовать token-level KL вместо sequence-level (тяжелее).
    """
    entropy_weight:   float = 0.1    # α из тех-плана
    clip_higher:      bool  = False  # True = полный DAPO, False = simplified
    use_token_kl:     bool  = False  # True = token-level KL (медленнее)
    output_dir:       str   = "./outputs/dapo"


# ──────────────────────────────────────────────────────────────────────────────
# Тренер DAPO
# ──────────────────────────────────────────────────────────────────────────────

def train_dapo(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    config: Optional[DAPOTrainingConfig] = None,
    run_name: str = "dapo_experiment",
) -> dict:
    """
    Запускает DAPO обучение (GRPO + entropy bonus reward).

    DAPO = GRPO с модифицированной reward function:
      reward = accuracy_reward + α × unique_token_ratio

    Args:
        model:         Модель с LoRA адаптерами
        tokenizer:     Токенизатор
        train_dataset: Обучающий датасет
        eval_dataset:  Валидационный датасет (опционально)
        config:        DAPOTrainingConfig
        run_name:      Имя запуска в WandB

    Returns:
        dict с метриками обучения
    """
    if config is None:
        config = DAPOTrainingConfig()

    # Энтропийная reward function с заданным весом
    from src.rewards.reward_entropy import reward_with_entropy_bonus

    def dapo_reward_fn(prompts, completions, true_labels, **kwargs):
        return reward_with_entropy_bonus(
            prompts=prompts,
            completions=completions,
            true_labels=true_labels,
            tokenizer=tokenizer,
            entropy_weight=config.entropy_weight,
        )

    logger.info(f"DAPO: entropy_weight={config.entropy_weight}, clip_higher={config.clip_higher}")

    # Используем GRPO тренер с DAPO reward
    from src.training.grpo_trainer import train_grpo
    metrics = train_grpo(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_fn=dapo_reward_fn,
        config=config,  # DAPOConfig совместим с GRPOConfig (наследование)
        run_name=run_name,
    )

    metrics["method"] = "dapo"
    metrics["entropy_weight"] = config.entropy_weight
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# λ-GRPO (альтернативная модификация)
# ──────────────────────────────────────────────────────────────────────────────

def train_lambda_grpo(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    target_words: int = 50,
    run_name: str = "lambda_grpo_experiment",
) -> dict:
    """
    Запускает λ-GRPO обучение (GRPO + length-weighted reward).

    Args:
        model:        Модель с LoRA
        tokenizer:    Токенизатор
        train_dataset: Обучающий датасет
        eval_dataset:  Валидационный датасет
        target_words: Целевая длина ответа в словах
        run_name:     Имя WandB run

    Returns:
        dict с метриками
    """
    from src.rewards.reward_lambda_grpo import reward_length_weighted

    def lambda_reward_fn(prompts, completions, true_labels, **kwargs):
        return reward_length_weighted(
            prompts=prompts,
            completions=completions,
            true_labels=true_labels,
            target_words=target_words,
        )

    config = GRPOTrainingConfig(output_dir="./outputs/lambda_grpo")

    from src.training.grpo_trainer import train_grpo
    metrics = train_grpo(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_fn=lambda_reward_fn,
        config=config,
        run_name=run_name,
    )

    metrics["method"] = "lambda_grpo"
    metrics["target_words"] = target_words
    return metrics
