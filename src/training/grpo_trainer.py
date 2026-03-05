"""
grpo_trainer.py
===============
Обучение LLM через GRPO (Group Relative Policy Optimization).

GRPO — ключевой алгоритм, используемый в DeepSeek-R1.
Отличие от PPO: не требует вспомогательной critic-модели.
Вместо этого использует группу сгенерированных ответов
для оценки относительного качества каждого (group-based baseline).

Ссылка: https://arxiv.org/abs/2402.03300
"""

import os
from typing import Optional, Callable
from dataclasses import dataclass, field

from datasets import Dataset, DatasetDict
from src.data.data_utils import logger, check_gpu


# ──────────────────────────────────────────────────────────────────────────────
# Конфиг GRPO
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GRPOTrainingConfig:
    """Гиперпараметры GRPO обучения."""

    output_dir:                  str   = "./outputs/grpo"
    num_train_epochs:            int   = 3
    per_device_train_batch_size: int   = 4
    gradient_accumulation_steps: int   = 4    # эффективный батч = 4×4 = 16
    learning_rate:               float = 5e-6
    num_generations:             int   = 8    # G — группа ответов (ключевая идея GRPO)
    max_new_tokens:              int   = 256
    max_prompt_length:           int   = 512
    logging_steps:               int   = 10
    save_steps:                  int   = 100
    eval_steps:                  int   = 50
    warmup_ratio:                float = 0.1
    weight_decay:                float = 0.01
    report_to:                   str   = "wandb"
    seed:                        int   = 42
    fp16:                        bool  = False
    bf16:                        bool  = True   # A100/H100; для T4 ставь fp16=True
    dataloader_num_workers:      int   = 0
    remove_unused_columns:       bool  = False


# ──────────────────────────────────────────────────────────────────────────────
# Тренер GRPO
# ──────────────────────────────────────────────────────────────────────────────

def train_grpo(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    reward_fn: Optional[Callable] = None,
    config: Optional[GRPOTrainingConfig] = None,
    run_name: str = "grpo_experiment",
) -> dict:
    """
    Запускает GRPO обучение.

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
        config = GRPOTrainingConfig()

    if reward_fn is None:
        from src.rewards import reward_with_reasoning as reward_fn
        logger.info("Reward function не задана — используется reward_with_reasoning")

    check_gpu()
    _setup_wandb(run_name)

    try:
        from trl import GRPOTrainer, GRPOConfig

        grpo_config = GRPOConfig(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            num_generations=config.num_generations,
            max_new_tokens=config.max_new_tokens,
            max_prompt_length=config.max_prompt_length,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            report_to=config.report_to,
            seed=config.seed,
            fp16=config.fp16,
            bf16=config.bf16,
            dataloader_num_workers=config.dataloader_num_workers,
            remove_unused_columns=config.remove_unused_columns,
            run_name=run_name,
        )

        # Создаём обёртку reward_fn для совместимости с TRL API
        # TRL передаёт (prompts, completions, **row_kwargs) где row_kwargs — столбцы датасета
        def reward_wrapper(prompts, completions, **kwargs):
            true_labels = kwargs.get("label", [])
            return reward_fn(
                prompts=prompts,
                completions=completions,
                true_labels=true_labels,
                tokenizer=tokenizer,
            )

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            config=grpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            reward_funcs=[reward_wrapper],
        )

        logger.info(f"Запуск GRPO обучения: {config.num_train_epochs} эпох")
        logger.info(f"  Batch size: {config.per_device_train_batch_size} × {config.gradient_accumulation_steps} acc steps")
        logger.info(f"  Generations per prompt: {config.num_generations}")

        train_result = trainer.train()

        metrics = train_result.metrics
        metrics["method"] = "grpo"
        metrics["run_name"] = run_name
        logger.info(f"GRPO завершён: {metrics}")

        return metrics

    except ImportError as e:
        logger.error(f"TRL не установлен: {e}")
        logger.error("Запусти: pip install trl>=0.12.0")
        raise


# ──────────────────────────────────────────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────────────────────────────────────────

def _setup_wandb(run_name: str) -> None:
    """Инициализация WandB."""
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        logger.warning("WANDB_API_KEY не задан — логирование в файл")
        return
    try:
        import wandb
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "behavior-analysis-rl-llm"),
            entity=os.getenv("WANDB_ENTITY"),
            name=run_name,
            config={},
        )
        logger.info(f"WandB инициализирован: run={run_name}")
    except ImportError:
        logger.warning("WandB не установлен")
