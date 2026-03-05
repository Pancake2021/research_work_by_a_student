"""
ppo_trainer.py
==============
Обучение LLM через PPO (Proximal Policy Optimization).

PPO инструктируется sentiment-classification reward.
Требует дополнительную critic-модель → ~2× больше VRAM чем GRPO.

На Colab T4 (16GB) — использовать gradient_checkpointing + load_in_4bit.
На A100 (40GB) — работает без ограничений.

Ссылка: https://arxiv.org/abs/1707.06347
"""

import os
from typing import Optional, Callable
from dataclasses import dataclass

from datasets import Dataset
from src.data.data_utils import logger, check_gpu


# ──────────────────────────────────────────────────────────────────────────────
# Конфиг PPO
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PPOTrainingConfig:
    """Гиперпараметры PPO обучения."""

    output_dir:                  str   = "./outputs/ppo"
    num_train_epochs:            int   = 3
    per_device_train_batch_size: int   = 4
    gradient_accumulation_steps: int   = 4
    learning_rate:               float = 5e-6

    # PPO специфические
    kl_penalty:                  str   = "kl"    # 'kl' или 'abs' или 'mse'
    init_kl_coef:                float = 0.1     # начальный коэффициент KL
    adap_kl_ctrl:                bool  = True    # адаптивный KL
    target_kl:                   float = 6.0     # целевой KL

    # Генерация
    max_new_tokens:              int   = 256
    max_prompt_length:           int   = 512

    # Логирование
    logging_steps:               int   = 10
    save_steps:                  int   = 100
    report_to:                   str   = "wandb"
    seed:                        int   = 42
    fp16:                        bool  = False
    bf16:                        bool  = True
    gradient_checkpointing:      bool  = True    # критично для T4


# ──────────────────────────────────────────────────────────────────────────────
# Тренер PPO
# ──────────────────────────────────────────────────────────────────────────────

def train_ppo(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    reward_fn: Optional[Callable] = None,
    config: Optional[PPOTrainingConfig] = None,
    run_name: str = "ppo_experiment",
) -> dict:
    """
    Запускает PPO обучение.

    Args:
        model:         Модель с LoRA адаптерами
        tokenizer:     Токенизатор
        train_dataset: Обучающий датасет
        eval_dataset:  Валидационный датасет (опционально)
        reward_fn:     Функция награды из src.rewards
        config:        Конфиг обучения
        run_name:      Имя запуска в WandB

    Returns:
        dict с метриками обучения
    """
    if config is None:
        config = PPOTrainingConfig()

    if reward_fn is None:
        from src.rewards import reward_with_reasoning as reward_fn

    check_gpu()

    try:
        from trl import PPOConfig, PPOTrainer

        ppo_config = PPOConfig(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            kl_penalty=config.kl_penalty,
            init_kl_coef=config.init_kl_coef,
            adap_kl_ctrl=config.adap_kl_ctrl,
            target_kl=config.target_kl,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            report_to=config.report_to,
            seed=config.seed,
            fp16=config.fp16,
            bf16=config.bf16,
            gradient_checkpointing=config.gradient_checkpointing,
            run_name=run_name,
        )

        def reward_wrapper(prompts, completions, **kwargs):
            true_labels = kwargs.get("label", [])
            return reward_fn(
                prompts=prompts,
                completions=completions,
                true_labels=true_labels,
                tokenizer=tokenizer,
            )

        trainer = PPOTrainer(
            model=model,
            processing_class=tokenizer,
            config=ppo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            reward_funcs=[reward_wrapper],
        )

        logger.info("=" * 50)
        logger.info("ВНИМАНИЕ: PPO требует ~2× больше VRAM чем GRPO")
        logger.info("  T4 (16GB): используй fp16=True, batch_size=2")
        logger.info("  A100 (40GB): стандартные настройки")
        logger.info("=" * 50)

        logger.info(f"Запуск PPO обучения: {config.num_train_epochs} эпох")
        train_result = trainer.train()

        metrics = train_result.metrics
        metrics["method"] = "ppo"
        metrics["run_name"] = run_name
        logger.info(f"PPO завершён: {metrics}")

        return metrics

    except ImportError as e:
        logger.error(f"TRL не установлен: {e}")
        raise
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("🔴 OOM! Попробуй:")
            logger.error("  - Уменьшить per_device_train_batch_size до 1-2")
            logger.error("  - Включить gradient_checkpointing=True")
            logger.error("  - Использовать load_in_4bit=True")
            logger.error("  - Переключиться на GRPO (меньше памяти)")
        raise
