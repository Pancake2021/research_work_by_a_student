#!/usr/bin/env python3
"""
run_full_pipeline.py
====================
Единый скрипт запуска всего пайплайна дипломной работы.

Использование:
    python scripts/run_full_pipeline.py --mode baseline
    python scripts/run_full_pipeline.py --mode grpo --reward reasoning
    python scripts/run_full_pipeline.py --mode ppo
    python scripts/run_full_pipeline.py --mode dapo --entropy-weight 0.1
    python scripts/run_full_pipeline.py --mode lambda_grpo
    python scripts/run_full_pipeline.py --mode compare  # все методы + сравнение
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.data.data_utils import logger, check_gpu, seed_everything
from src.data.dataset_loader import load_behavior_dataset
from src.models.model_loader import load_model, prepare_for_inference, save_model
from src.models.baseline_eval import run_baseline_evaluation
from src.rewards import get_reward_fn
from src.evaluation.evaluator import evaluate_checkpoint, compare_methods
from src.visualization.plots import plot_all_from_results, plot_radar_chart


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="RL-обучение LLM для поведенческого анализа")
    parser.add_argument(
        "--mode",
        choices=["baseline", "grpo", "ppo", "dapo", "lambda_grpo", "compare"],
        default="baseline",
        help="Режим запуска",
    )
    parser.add_argument(
        "--dataset", choices=["iemocap", "cmu_mosi", "local_json", "synthetic"],
        default="iemocap", help="Датасет"
    )
    parser.add_argument("--reward", default="reasoning",
        choices=["accuracy", "reasoning", "binary", "entropy", "lambda_grpo"],
        help="Reward function (для GRPO/PPO)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--entropy-weight", type=float, default=0.1)
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--push-to-hub", action="store_true",
        help="Опубликовать модель на HuggingFace Hub")
    parser.add_argument("--hub-repo", default="behavior-analysis-grpo-qwen2.5")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Шаги пайплайна
# ──────────────────────────────────────────────────────────────────────────────

def step_load_data(args):
    """Фаза 1: Загрузка и подготовка данных."""
    logger.info("=" * 60)
    logger.info("ФАЗА 1: ЗАГРУЗКА ДАННЫХ")
    logger.info("=" * 60)

    dataset = load_behavior_dataset(
        dataset_name=args.dataset,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed,
        save_path=f"{args.output_dir}/dataset",
    )
    return dataset


def step_load_model(with_lora: bool = True):
    """Фаза 2: Загрузка модели."""
    logger.info("=" * 60)
    logger.info("ФАЗА 2: ЗАГРУЗКА МОДЕЛИ")
    logger.info("=" * 60)
    return load_model(with_lora=with_lora)


def step_baseline(model, tokenizer, dataset, args):
    """Фаза 2: Baseline оценка (ДО обучения)."""
    logger.info("=" * 60)
    logger.info("ФАЗА 2: BASELINE ОЦЕНКА")
    logger.info("=" * 60)

    model = prepare_for_inference(model)
    results = run_baseline_evaluation(
        model, tokenizer, dataset["test"],
        save_path=f"{args.output_dir}/results/baseline.json",
    )
    return results


def step_train_grpo(model, tokenizer, dataset, args):
    """Фаза 4: Обучение GRPO."""
    from src.training.grpo_trainer import train_grpo, GRPOTrainingConfig

    config = GRPOTrainingConfig(
        output_dir=f"{args.output_dir}/grpo",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    reward_fn = get_reward_fn(args.reward)
    return train_grpo(
        model, tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        reward_fn=reward_fn,
        config=config,
        run_name=f"grpo_{args.reward}",
    )


def step_train_ppo(model, tokenizer, dataset, args):
    """Фаза 5: Обучение PPO."""
    from src.training.ppo_trainer import train_ppo, PPOTrainingConfig

    config = PPOTrainingConfig(
        output_dir=f"{args.output_dir}/ppo",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    reward_fn = get_reward_fn(args.reward)
    return train_ppo(
        model, tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        reward_fn=reward_fn,
        config=config,
        run_name=f"ppo_{args.reward}",
    )


def step_train_dapo(model, tokenizer, dataset, args):
    """Фаза 6: Обучение DAPO."""
    from src.training.dapo_trainer import train_dapo, DAPOTrainingConfig

    config = DAPOTrainingConfig(
        output_dir=f"{args.output_dir}/dapo",
        num_train_epochs=args.epochs,
        entropy_weight=args.entropy_weight,
    )
    return train_dapo(
        model, tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        config=config,
        run_name="dapo_experiment",
    )


def step_train_lambda_grpo(model, tokenizer, dataset, args):
    """Фаза 6: Обучение λ-GRPO."""
    from src.training.dapo_trainer import train_lambda_grpo

    return train_lambda_grpo(
        model, tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        run_name="lambda_grpo_experiment",
    )


def step_compare(model, tokenizer, dataset, args):
    """Фаза 7: Сравнение всех методов."""
    all_results = []

    # Baseline
    baseline_model, baseline_tokenizer = step_load_model(with_lora=True)
    baseline_res = step_baseline(baseline_model, baseline_tokenizer, dataset, args)
    all_results.append(baseline_res)

    # TODO во второй итерации: загрузить обученные чекпоинты GRPO/PPO/DAPO
    # и добавить их в all_results через evaluate_checkpoint()

    comparison = compare_methods(all_results)
    plot_all_from_results(all_results)

    return comparison


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    seed_everything(args.seed)
    check_gpu()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/results", exist_ok=True)

    logger.info(f"Режим: {args.mode.upper()}")
    logger.info(f"Датасет: {args.dataset} | Reward: {args.reward}")

    # Загружаем данные
    dataset = step_load_data(args)

    if args.mode == "baseline":
        model, tokenizer = step_load_model(with_lora=False)
        results = step_baseline(model, tokenizer, dataset, args)
        logger.info(f"\n✅ Baseline F1: {results['f1_weighted']:.4f}")

    elif args.mode == "grpo":
        model, tokenizer = step_load_model(with_lora=True)
        train_results = step_train_grpo(model, tokenizer, dataset, args)

        # Сохраняем модель
        save_model(model, tokenizer, f"{args.output_dir}/grpo/best_model")

        if args.push_to_hub:
            from src.models.model_loader import push_to_hub
            push_to_hub(model, tokenizer, args.hub_repo)

    elif args.mode == "ppo":
        model, tokenizer = step_load_model(with_lora=True)
        step_train_ppo(model, tokenizer, dataset, args)
        save_model(model, tokenizer, f"{args.output_dir}/ppo/best_model")

    elif args.mode == "dapo":
        model, tokenizer = step_load_model(with_lora=True)
        step_train_dapo(model, tokenizer, dataset, args)
        save_model(model, tokenizer, f"{args.output_dir}/dapo/best_model")

    elif args.mode == "lambda_grpo":
        model, tokenizer = step_load_model(with_lora=True)
        step_train_lambda_grpo(model, tokenizer, dataset, args)
        save_model(model, tokenizer, f"{args.output_dir}/lambda_grpo/best_model")

    elif args.mode == "compare":
        model, tokenizer = step_load_model(with_lora=True)
        step_compare(model, tokenizer, dataset, args)

    logger.info("\n🎉 Пайплайн завершён успешно!")


if __name__ == "__main__":
    main()
