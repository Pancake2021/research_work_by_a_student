#!/usr/bin/env python3
"""
run_evaluation.py
=================
Финальная оценка обученных моделей.

Использование:
    # Оценить один чекпоинт
    python scripts/run_evaluation.py --checkpoint ./outputs/grpo/best_model --method grpo

    # Сравнить все методы
    python scripts/run_evaluation.py --compare \
        --grpo ./outputs/grpo/best_model \
        --ppo  ./outputs/ppo/best_model \
        --dapo ./outputs/dapo/best_model
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.data.data_utils import logger, seed_everything
from src.data.dataset_loader import load_behavior_dataset
from src.models.model_loader import load_model, prepare_for_inference
from src.evaluation.evaluator import evaluate_checkpoint, compare_methods
from src.evaluation.error_analysis import analyze_errors
from src.visualization.plots import plot_all_from_results, plot_radar_chart


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Путь к чекпоинту модели")
    parser.add_argument("--method", default="unknown", help="Название метода")
    parser.add_argument("--compare", action="store_true", help="Режим сравнения")
    parser.add_argument("--grpo",  help="Чекпоинт GRPO")
    parser.add_argument("--ppo",   help="Чекпоинт PPO")
    parser.add_argument("--dapo",  help="Чекпоинт DAPO")
    parser.add_argument("--baseline", help="Базовая модель (без RL)")
    parser.add_argument("--dataset", default="iemocap")
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_trained_model(checkpoint_path: str, base_model_name: str = None):
    """Загружает модель из чекпоинта LoRA."""
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if base_model_name is None:
        base_model_name = os.getenv("MODEL_NAME", "unsloth/Qwen2.5-1.5B-Instruct")

    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        model.eval()

    return model, tokenizer


def evaluate_one(checkpoint_path, method_name, test_dataset, output_dir):
    """Оценивает один чекпоинт."""
    logger.info(f"\nОцениваю: {method_name} от {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        logger.warning(f"Чекпоинт не найден: {checkpoint_path}")
        return None

    model, tokenizer = load_trained_model(checkpoint_path)
    results = evaluate_checkpoint(
        model, tokenizer, test_dataset,
        method_name=method_name,
        save_path=f"{output_dir}/results/{method_name}_eval.json",
    )

    # Анализ ошибок
    from src.data.preprocessor import parse_label, build_chat_prompt
    from src.models.baseline_eval import run_inference
    examples = [{"text": ex["text"]} for ex in test_dataset]
    responses = run_inference(model, tokenizer, examples)
    preds = [parse_label(r) for r in responses]
    true = [ex["label"] for ex in test_dataset]

    analyze_errors(examples, true, preds, method_name=method_name)
    return results


def main():
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(f"{args.output_dir}/results", exist_ok=True)
    os.makedirs(f"{args.output_dir}/plots", exist_ok=True)

    # Загружаем тестовый датасет
    dataset = load_behavior_dataset(
        dataset_name=args.dataset,
        train_size=10,  # только для структуры
        test_size=args.test_size,
        seed=args.seed,
    )
    test_dataset = dataset["test"]

    if args.compare:
        all_results = []
        checkpoints = {}
        if args.baseline: checkpoints["baseline"]   = args.baseline
        if args.grpo:     checkpoints["grpo"]        = args.grpo
        if args.ppo:      checkpoints["ppo"]         = args.ppo
        if args.dapo:     checkpoints["dapo"]        = args.dapo

        for method_name, checkpoint_path in checkpoints.items():
            result = evaluate_one(checkpoint_path, method_name, test_dataset, args.output_dir)
            if result:
                all_results.append(result)

        if all_results:
            comparison = compare_methods(all_results)
            plot_all_from_results(all_results)

            # Radar chart
            radar_scores = {}
            for r in all_results:
                radar_scores[r["method"]] = {
                    "quality":   r.get("f1_weighted", 0),
                    "speed":     0.7,  # заполнить вручную
                    "memory":    0.8,  # заполнить вручную
                    "stability": 0.75, # заполнить вручную
                }
            plot_radar_chart(radar_scores)

            logger.info(f"\n🏆 Лучший метод: {comparison['best_method']} F1={comparison['best_f1']:.4f}")

    elif args.checkpoint:
        evaluate_one(args.checkpoint, args.method, test_dataset, args.output_dir)

    else:
        logger.error("Укажи --checkpoint или --compare")
        sys.exit(1)


if __name__ == "__main__":
    main()
