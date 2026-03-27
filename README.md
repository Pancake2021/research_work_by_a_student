# Оптимизация RL-обучения LLM для задач поведенческого анализа

**Autor:** Теридакс Макута  
**Period:** Март — Май 2026  
**Topic:** Fine-tuning open-source LLM (1.5B–3B params) через Reinforcement Learning на задаче классификации поведенческих паттернов в тексте. Сравнение PPO vs GRPO vs модификации (DAPO, λ-GRPO).

---

## Архитектура пайплайна

```
Данные → Препроцессинг → Base LLM (Qwen-2.5-1.5B) → RL-обучение → Reward Function → Оценка
```

---

## Структура проекта

```
research_work_by_a_student/
├── configs/                    # YAML конфиги обучения
│   ├── grpo_config.yaml
│   └── ppo_config.yaml
├── notebooks/                  # Jupyter/Colab ноутбуки
│   ├── 01_baseline.ipynb
│   ├── 02_grpo_training.ipynb
│   ├── 03_ppo_training.ipynb
│   └── 04_evaluation.ipynb
├── scripts/                    # Скрипты запуска
│   ├── run_full_pipeline.py
│   └── run_evaluation.py
├── src/
│   ├── data/                   # Данные и препроцессинг
│   │   ├── dataset_loader.py
│   │   ├── preprocessor.py
│   │   └── data_utils.py
│   ├── models/                 # Загрузка и eval базовой модели
│   │   ├── model_loader.py
│   │   └── baseline_eval.py
│   ├── rewards/                # Reward functions (RF1, RF2, RF3)
│   │   ├── __init__.py
│   │   ├── reward_accuracy.py
│   │   ├── reward_reasoning.py
│   │   ├── reward_binary.py
│   │   ├── reward_entropy.py
│   │   └── reward_lambda_grpo.py
│   ├── training/               # Тренеры PPO / GRPO / DAPO
│   │   ├── grpo_trainer.py
│   │   ├── ppo_trainer.py
│   │   └── dapo_trainer.py
│   ├── evaluation/             # Метрики и анализ ошибок
│   │   ├── evaluator.py
│   │   └── error_analysis.py
│   └── visualization/          # Графики для диплома
│       └── plots.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## Быстрый старт (Colab / Kaggle)

### 1. Установка зависимостей

```bash
pip install unsloth transformers trl datasets accelerate peft
pip install wandb bitsandbytes sentencepiece evaluate scikit-learn
pip install plotly kaleido
```

### 2. Переменные окружения

```bash
cp .env.example .env
# Заполнить HF_TOKEN и WANDB_API_KEY
```

### 3. Проверка GPU

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

### 4. Запуск baseline

```bash
python scripts/run_full_pipeline.py --mode baseline
```

### 5. Обучение GRPO

```bash
python scripts/run_full_pipeline.py --mode grpo --reward reasoning
```

### 6. Обучение PPO

```bash
python scripts/run_full_pipeline.py --mode ppo
```

### 7. Финальная оценка

```bash
python scripts/run_evaluation.py --checkpoint ./grpo_output
```

---

## Запуск Через Colab MCP (Удалённо)

Для запуска экспериментов в Google Colab (GPU/TPU) используй:

1. Colab сервер: `agent/colab_mcp_server.py`
2. Локальный оркестратор: `python -m agent.experiment_runner ...`

Подробный quickstart:
[`agent/COLAB_QUICKSTART.md`](agent/COLAB_QUICKSTART.md)

`run_full_pipeline.py` печатает machine-readable JSON-события в stdout для устойчивого парсинга метрик агентом.

---

## Reward Functions

| RF | Описание | Диапазон | Когда использовать |
|----|----------|----------|-------------------|
| RF1 `accuracy` | Бинарная: правильно/нет | `{0.0, 1.0}` | Простой baseline |
| RF2 `reasoning` | RF1 + бонус за наличие раздела "Анализ:" | `{0.0, 1.2}` | Основной эксперимент |
| RF3 `binary` | 0 / -0.5 / -1.0 (штрафы за отказ и ошибку) | `{-1.0, 0.0}` | DeepSeek-R1 стиль |
| RF4 `entropy` | RF1 + энтропийный бонус токенов | `[0.0, 1.1]` | λ-GRPO / DAPO |
| RF5 `lambda_grpo` | RF1 × нормированная длина ответа | `[0.0, 1.0]` | λ-GRPO |

---

## Методы обучения

| Метод | Описание | GPU RAM (4-bit LoRA) |
|-------|----------|----------------------|
| Baseline | Нет RL, только инференс | ~4 GB |
| GRPO | Group Relative Policy Optimization | ~8–10 GB |
| PPO | Proximal Policy Optimization + critic | ~14–16 GB |
| DAPO | GRPO + entropy bonus | ~8–10 GB |
| λ-GRPO | GRPO + length-weighted reward | ~8–10 GB |

---

## Таблица результатов (заполнять в процессе)

| Метод | Reward Fn | F1-score | Время (ч) | GPU RAM | Стабильность |
|-------|-----------|----------|-----------|---------|--------------|
| Baseline | — | — | — | — | — |
| GRPO | RF2 | — | — | — | — |
| PPO | RF2 | — | — | — | — |
| DAPO | RF4 | — | — | — | — |
| λ-GRPO | RF5 | — | — | — | — |

---

## Литература

1. DeepSeek-R1: https://arxiv.org/abs/2501.12948
2. TRL Docs: https://huggingface.co/docs/trl
3. Unsloth GRPO: https://unsloth.ai/blog/grpo
4. PPO (Schulman et al., 2017): https://arxiv.org/abs/1707.06347
5. DAPO: https://arxiv.org/abs/2503.14476
