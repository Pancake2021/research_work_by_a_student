# Small LLM + GRPO для UEBA / Insider Threat Detection

НИРС: исследование применимости малых языковых моделей и reward-based RL-дообучения для анализа поведения пользователей в задачах информационной безопасности.

Фокус работы: UEBA-сценарии на основе CERT Insider Threat Dataset R4.2. Модель получает текстовое описание активности пользователя и должна вернуть SOC-полезный ответ:

```text
Риск: <normal|suspicious|malicious>
Признаки: <2-4 признака риска>
Обоснование: <краткое объяснение>
```

Qwen2.5 не заявляется как SOTA. Он используется как стабильный baseline. Основная стратегия выбора модели — предварительный model bake-off актуальных small LLM на одном dev-set.

## Исследовательская схема

1. Подготовить CERT CSV в UEBA-сценарии.
2. Разбить данные по пользователям, чтобы исключить leakage.
3. Запустить classical ML baseline по агрегированным признакам.
4. Запустить zero/few-shot bake-off small LLM.
5. Выбрать main model по F1, recall malicious, valid format rate, evidence hit rate и VRAM.
6. Провести SFT/GRPO эксперименты с reward-функциями.
7. Сравнить качество, объяснимость, формат и ошибки.

## Структура проекта

```text
research_work_by_a_student/
├── configs/
│   ├── grpo_config.yaml
│   ├── model_registry.yaml
│   ├── ppo_config.yaml
│   └── tech_stack.yaml
├── notebooks/
│   └── colab_diploma_experiments.ipynb
├── scripts/
│   ├── model_bakeoff.py
│   ├── prepare_cert_dataset.py
│   ├── run_full_pipeline.py
│   ├── run_ueba_baseline.py
│   ├── setup_uv_env.sh
│   └── macos_nightly_runner.sh
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── rewards/
│   ├── training/
│   └── visualization/
├── pyproject.toml
├── requirements.txt
├── requirements-gpu.txt
└── README.md
```

## Модели для bake-off

Список кандидатов хранится в `configs/model_registry.yaml`.

| Роль | Модель |
|---|---|
| main candidate | `Qwen/Qwen3-4B-Instruct-2507` |
| practical baseline | `Qwen/Qwen2.5-3B-Instruct` |
| small fallback | `Qwen/Qwen2.5-1.5B-Instruct` |
| alternative open | `HuggingFaceTB/SmolLM3-3B` |
| alternative strong | `microsoft/Phi-4-mini-instruct` |
| optional | `google/gemma-4-E4B-it` |

## Стек проекта

Машинно-читаемая фиксация стека хранится в `configs/tech_stack.yaml`.

| Часть | Основной инструмент | Роль |
|---|---|---|
| Dataset pipeline | `pandas`, `datasets` | CERT CSV, JSONL splits, HF Dataset interop |
| Classical baseline | `scikit-learn` | Logistic Regression / RandomForest |
| Compatible inference | `transformers` | fallback для любого HF model card |
| Fast inference | `vLLM` | быстрый zero/few-shot bake-off и evaluation |
| Efficient finetuning | `Unsloth` | 4-bit QLoRA/SFT/GRPO на 8GB GPU |
| RL/SFT trainers | `TRL` | `SFTTrainer`, `GRPOTrainer`, reward integration |
| Adapters | `PEFT` | LoRA/QLoRA checkpoints |
| Quantization | `bitsandbytes` | NF4/4-bit режим для RTX 4060 8GB |
| Tracking | `wandb` + local outputs | curves, configs, metrics, predictions |

Default для bake-off — `transformers`, потому что он максимально совместим. Для реальных массовых прогонов на CUDA/Linux использовать `--backend vllm`. Для SFT/GRPO основной путь — `Unsloth + TRL + PEFT + bitsandbytes`; fallback — `Transformers + TRL + PEFT + bitsandbytes`.

## Окружение

Быстрая настройка через `uv`:

```bash
scripts/setup_uv_env.sh
```

Дополнительно:

```bash
scripts/setup_uv_env.sh local colab
scripts/setup_uv_env.sh local cuda
scripts/setup_uv_env.sh --help
```

Примечание: группа `cuda` нужна для Linux/CUDA-окружений, для macOS обычно не подходит.

Portable pip setup:

```bash
pip install -r requirements.txt
```

CUDA/Linux setup:

```bash
pip install -r requirements-gpu.txt
```

## Подготовка данных

Smoke dataset без CERT:

```bash
python scripts/prepare_cert_dataset.py \
  --synthetic-smoke \
  --output-dir outputs/cert_ueba_smoke
```

Реальный CERT dataset:

```bash
python scripts/prepare_cert_dataset.py \
  --data-dir /path/to/cert/r4.2 \
  --labels /path/to/labels.csv \
  --output-dir outputs/cert_ueba
```

Ожидаемые CERT файлы: `logon.csv`, `device.csv`, `file.csv`, `email.csv`, `http.csv`.

## Baseline и bake-off

Classical ML baseline:

```bash
python scripts/run_ueba_baseline.py \
  --train-jsonl outputs/cert_ueba/train.jsonl \
  --test-jsonl outputs/cert_ueba/test.jsonl \
  --model logreg
```

Mock bake-off для проверки пайплайна без скачивания моделей:

```bash
python scripts/model_bakeoff.py \
  --dataset-jsonl outputs/cert_ueba_smoke/dev.jsonl \
  --registry configs/model_registry.yaml \
  --mock
```

Реальный zero-shot bake-off:

```bash
python scripts/model_bakeoff.py \
  --dataset-jsonl outputs/cert_ueba/dev.jsonl \
  --registry configs/model_registry.yaml \
  --models qwen3_4b_instruct_2507 qwen2_5_3b_instruct smollm3_3b phi4_mini_instruct \
  --limit 100
```

Быстрый bake-off через vLLM:

```bash
python scripts/model_bakeoff.py \
  --dataset-jsonl outputs/cert_ueba/dev.jsonl \
  --registry configs/model_registry.yaml \
  --backend vllm \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --models qwen3_4b_instruct_2507 qwen2_5_3b_instruct smollm3_3b phi4_mini_instruct \
  --limit 100
```

Few-shot bake-off:

```bash
python scripts/model_bakeoff.py \
  --dataset-jsonl outputs/cert_ueba/dev.jsonl \
  --few-shot-k 3 \
  --limit 100
```

## Colab Notebook

Для стабильного запуска в Colab используется:

- `notebooks/colab_diploma_experiments.ipynb`

Он включает установку зависимостей, обновление ветки `develop`, запуск EXP-01..EXP-07, логирование, сохранение данных и построение графиков.

## Ночной фоновый запуск на macOS

Для долгого прогона в фоне:

```bash
scripts/macos_nightly_runner.sh start baseline
```

Проверка/остановка:

```bash
scripts/macos_nightly_runner.sh status
scripts/macos_nightly_runner.sh stop
```

Важно: MacBook при закрытии крышки обычно уходит в сон. Для надежных CUDA-прогонов использовать Linux/Colab.

## Метрики

Основные метрики:

- `accuracy`
- `macro_f1`
- `weighted_f1`
- `recall_malicious`
- `false_positive_rate`
- `valid_format_rate`
- `evidence_hit_rate`
- `avg_response_length`
- `train_time_minutes`
- `peak_vram_gb`

Критерий выбора main model: не только F1, но и способность стабильно выдавать корректный SOC-формат с признаками риска.

## Reward-функции

Legacy rewards:

| RF | Описание | Диапазон |
|----|----------|----------|
| `accuracy` | Бинарная: правильно/нет | `{0.0, 1.0}` |
| `reasoning` | Accuracy + бонус за раздел "Анализ:" | `{0.0, 1.2}` |
| `binary` | штрафы за отказ и ошибку | `{-1.0, -0.5, 0.0}` |
| `entropy` | accuracy + энтропийный бонус | `[0.0, 1.1]` |
| `lambda_grpo` | accuracy × нормированная длина | `[0.0, 1.0}` |

UEBA rewards:

- `ueba_accuracy` — награда только за правильный risk label;
- `ueba_format` — risk label + наличие обязательных полей;
- `ueba_evidence` — risk label + формат + совпадение evidence + штраф за неподтвержденные признаки.

GRPO остается центральным RL-методом. PPO не является обязательным, так как хуже подходит под RTX 4060 8GB и требует более сложной value/reward-model схемы.

## Outputs

Каждый запуск пишет результаты в отдельный каталог:

```text
outputs/<run_id>/
├── run_config.json
├── model_registry.yaml
├── summary.json
├── summary.md
└── <model_key>/
    ├── metrics.json
    ├── predictions.jsonl
    └── samples.md
```

## Статус и планы НИРС

Все решения и журналы НИРС ведутся в Obsidian:

```text
/Users/glebpankeev/Documents/Obsidian Vault/SSAU/НИРС
```
