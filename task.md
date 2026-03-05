# Диплом: Оптимизация RL-обучения LLM для задач поведенческого анализа

## Фаза 0 — Окружение
- [ ] Создать `requirements.txt`
- [ ] Создать `setup.py` / `pyproject.toml`
- [ ] Создать `.env.example`
- [ ] Создать `README.md`

## Фаза 1 — Данные
- [ ] `src/data/dataset_loader.py` — загрузка датасета IEMOCAP/CMU-MOSI
- [ ] `src/data/preprocessor.py` — препроцессинг и форматирование
- [ ] `src/data/data_utils.py` — вспомогательные функции

## Фаза 2 — Baseline модель
- [ ] `src/models/model_loader.py` — загрузка Qwen-2.5-1.5B через Unsloth
- [ ] `src/models/baseline_eval.py` — замер baseline метрик

## Фаза 3 — Reward Functions
- [ ] `src/rewards/reward_accuracy.py` — RF1: простая accuracy
- [ ] `src/rewards/reward_reasoning.py` — RF2: с бонусом за рассуждение
- [ ] `src/rewards/reward_binary.py` — RF3: бинарная со штрафом (DeepSeek-R1 стиль)
- [ ] `src/rewards/__init__.py`

## Фаза 4 — Обучение GRPO
- [ ] `src/training/grpo_trainer.py`
- [ ] `configs/grpo_config.yaml`

## Фаза 5 — Обучение PPO
- [ ] `src/training/ppo_trainer.py`
- [ ] `configs/ppo_config.yaml`

## Фаза 6 — Модификации GRPO
- [ ] `src/training/dapo_trainer.py` — энтропийный бонус
- [ ] `src/rewards/reward_entropy.py` — entropy-aware reward
- [ ] `src/rewards/reward_lambda_grpo.py` — λ-GRPO взвешенная награда

## Фаза 7 — Оценка и анализ
- [ ] `src/evaluation/evaluator.py` — финальные метрики
- [ ] `src/evaluation/error_analysis.py` — анализ ошибок
- [ ] `src/visualization/plots.py` — графики для диплома

## Фаза 8 — Сохранение модели
- [ ] `src/utils/model_saver.py`
- [ ] `notebooks/01_baseline.ipynb` — Colab-notebook для baseline
- [ ] `notebooks/02_grpo_training.ipynb`
- [ ] `notebooks/03_ppo_training.ipynb`
- [ ] `notebooks/04_evaluation.ipynb`

## Финал
- [ ] `scripts/run_full_pipeline.py` — единый скрипт запуска
- [ ] `scripts/run_evaluation.py`
