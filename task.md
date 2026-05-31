# НИРС: Small LLM + GRPO для UEBA

## Фаза 0 — База знаний и постановка
- [ ] Создать Obsidian-файлы `00_STATUS_LOG.md` ... `06_PRESENTATION_OUTLINE.md`
- [ ] Зафиксировать цель, объект, предмет, гипотезы в `01_RESEARCH_SPEC.md`
- [ ] Зафиксировать модельную стратегию bake-off first
- [x] Зафиксировать tech stack: vLLM, Unsloth, TRL, PEFT, bitsandbytes, Transformers

## Фаза 1 — CERT / UEBA данные
- [x] Добавить `src/data/scenario_builder.py`
- [x] Добавить `src/data/cert_loader.py`
- [x] Добавить `scripts/prepare_cert_dataset.py`
- [ ] Скачать/подключить CERT Insider Threat Dataset R4.2
- [ ] Подготовить реальные `train/dev/test.jsonl`
- [ ] Проверить split by user и отсутствие leakage

## Фаза 2 — Метрики и rewards
- [x] Добавить `src/evaluation/ueba_metrics.py`
- [x] Добавить `src/rewards/reward_ueba.py`
- [x] Обновить preprocessor под `normal/suspicious/malicious`
- [ ] Проверить evidence scoring на реальных примерах

## Фаза 3 — Baseline
- [x] Добавить `scripts/run_ueba_baseline.py`
- [ ] Запустить Logistic Regression baseline
- [ ] Запустить RandomForest baseline
- [ ] Сохранить результаты в `04_EXPERIMENT_LOG.md`

## Фаза 4 — Model bake-off
- [x] Добавить `configs/model_registry.yaml`
- [x] Добавить `configs/tech_stack.yaml`
- [x] Добавить `scripts/model_bakeoff.py`
- [x] Добавить optional vLLM backend для bake-off
- [x] Добавить ночной tmux pipeline `scripts/nightly_ueba_pipeline.sh`
- [x] Smoke-test `--mock`
- [ ] Zero-shot bake-off: Qwen3-4B, Qwen2.5-3B, SmolLM3-3B, Phi-4-mini
- [ ] Few-shot bake-off
- [ ] Зафиксировать выбор main model в `03_MODEL_BAKEOFF.md`

## Фаза 5 — SFT / GRPO
- [ ] Адаптировать SFT script под UEBA JSONL
- [ ] Адаптировать GRPO trainer под актуальный TRL API
- [ ] Запустить GRPO RF1 `ueba_accuracy`
- [ ] Запустить GRPO RF2 `ueba_format`
- [ ] Запустить GRPO RF3 `ueba_evidence`
- [ ] Сравнить rewards по F1, recall malicious, format/evidence metrics

## Фаза 6 — Анализ результатов
- [ ] Построить итоговую таблицу метрик
- [ ] Построить графики F1 / recall / FPR / valid format / evidence hit
- [ ] Провести анализ false positive / false negative
- [ ] Выбрать финальные результаты для отчета

## Фаза 7 — Отчет
- [ ] Сформировать `05_REPORT_OUTLINE.md`
- [ ] Написать введение, цель, объект, предмет
- [ ] Написать раздел UEBA и insider threat detection
- [ ] Написать раздел small LLM / SFT / GRPO / reward design
- [ ] Написать раздел датасета и экспериментов
- [ ] Вставить таблицы, графики и анализ ошибок
- [ ] Подготовить финальный DOCX в стиле примеров НИРС

## Фаза 8 — Презентация
- [ ] Сформировать `06_PRESENTATION_OUTLINE.md`
- [ ] Подготовить 8-10 слайдов
- [ ] Добавить отдельный слайд “Почему эта модель?”
- [ ] Добавить финальные графики и краткие выводы
