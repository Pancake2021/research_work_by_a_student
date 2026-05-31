"""Dataset loading for behavior-analysis RL experiments."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

from src.data.data_utils import logger

try:
    from datasets import Dataset as HFDataset
    from datasets import DatasetDict as HFDatasetDict
except ImportError:  # pragma: no cover
    HFDataset = None
    HFDatasetDict = None


def load_behavior_dataset(
    dataset_name: str = "iemocap",
    train_size: int = 500,
    test_size: int = 100,
    seed: int = 42,
    save_path: str | None = None,
    local_json_path: str | None = None,
    local_path: str | None = None,
):
    """Return DatasetDict/list dict with mandatory `text` and `label` columns.

    Supported dataset_name values:
    - `synthetic`: generated local binary sentiment data
    - `local_json`: read from json/jsonl/csv file
    - `iemocap`, `cmu_mosi`: best-effort load from Hugging Face aliases;
      automatically falls back to synthetic when unavailable.
    """
    rng = random.Random(seed)
    target_train = max(int(train_size), 1)
    target_test = max(int(test_size), 1)

    if dataset_name == "synthetic":
        train_rows = _generate_synthetic_rows(target_train, rng)
        test_rows = _generate_synthetic_rows(target_test, rng)
    elif dataset_name == "local_json":
        resolved_path = local_json_path or local_path or os.getenv("DATASET_JSON_PATH")
        if not resolved_path:
            raise ValueError("For dataset_name='local_json' set --dataset-path or DATASET_JSON_PATH")
        rows = _normalize_rows(_load_local_rows(resolved_path))
        if len(rows) < target_train + target_test:
            raise ValueError(f"Not enough rows in {resolved_path}: need at least {target_train + target_test}")
        rng.shuffle(rows)
        train_rows = rows[:target_train]
        test_rows = rows[target_train : target_train + target_test]
    else:
        rows = _normalize_rows(_load_remote_or_fallback(dataset_name, target_train + target_test, seed))
        rng.shuffle(rows)
        train_rows = rows[:target_train]
        test_rows = rows[target_train : target_train + target_test]

    if HFDataset is not None and HFDatasetDict is not None:
        train_ds = HFDataset.from_list(train_rows)
        test_ds = HFDataset.from_list(test_rows)
        dataset = HFDatasetDict({"train": train_ds, "test": test_ds})
        train_len = len(train_ds)
        test_len = len(test_ds)
    else:
        logger.warning("Package 'datasets' is unavailable; using plain python lists for train/test.")
        dataset = {"train": train_rows, "test": test_rows}
        train_len = len(train_rows)
        test_len = len(test_rows)

    logger.info("Dataset ready: %s | train=%s test=%s", dataset_name, train_len, test_len)

    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        _save_jsonl(dataset["train"], save_dir / "train.jsonl")
        _save_jsonl(dataset["test"], save_dir / "test.jsonl")
        logger.info("Saved dataset snapshot to %s", save_dir)

    return dataset


def _load_remote_or_fallback(dataset_name: str, min_rows: int, seed: int):
    """Best-effort remote loading; fallback to synthetic if unavailable."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.warning("Package 'datasets' unavailable, remote loading skipped.")
        rng = random.Random(seed)
        return _generate_synthetic_rows(min_rows * 2, rng)

    aliases = {
        "iemocap": [("daily_dialog", None), ("emotion", None)],
        "cmu_mosi": [("tweet_eval", "sentiment"), ("imdb", None)],
    }

    for ds_name, config in aliases.get(dataset_name, []):
        try:
            ds = load_dataset(ds_name, config, split="train") if config else load_dataset(ds_name, split="train")
            rows = []
            for ex in ds:
                text = _first_existing(ex, ["text", "sentence", "utterance", "review"]) or ""
                label = _map_label(_first_existing(ex, ["label", "sentiment", "emotion"]))
                if text and label in {"positive", "negative"}:
                    rows.append({"text": text, "label": label})
                if len(rows) >= min_rows * 2:
                    break
            if len(rows) >= min_rows:
                logger.info("Loaded remote dataset via alias: %s", ds_name)
                return rows
        except Exception as exc:
            logger.warning("Remote dataset alias failed (%s): %s", ds_name, exc)

    logger.warning("Falling back to synthetic dataset for '%s'", dataset_name)
    rng = random.Random(seed)
    return _generate_synthetic_rows(min_rows * 2, rng)


def _load_local_rows(path: str):
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = path_obj.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        with path_obj.open("r", encoding="utf-8") as f:
            if suffix == ".jsonl":
                return [json.loads(line) for line in f if line.strip()]
            data = json.load(f)
            if isinstance(data, dict):
                data = data.get("data", [])
            return list(data)

    if suffix == ".csv":
        with path_obj.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    raise ValueError(f"Unsupported local dataset format: {suffix}")


def _normalize_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        text = _first_existing(row, ["text", "utterance", "sentence", "review", "content"])
        raw_label = _first_existing(row, ["label", "sentiment", "target", "class"])
        label = _map_label(raw_label)
        if not text or label not in {"positive", "negative"}:
            continue
        normalized.append({"text": str(text), "label": label})
    return normalized


def _map_label(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return "positive" if float(raw) > 0 else "negative"

    value = str(raw).strip().lower()
    positives = {"1", "positive", "pos", "happy", "joy", "upbeat"}
    negatives = {"0", "-1", "negative", "neg", "sad", "angry", "fear", "disgust"}

    if value in positives:
        return "positive"
    if value in negatives:
        return "negative"
    return None


def _first_existing(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _generate_synthetic_rows(n: int, rng: random.Random) -> list[dict[str, str]]:
    pos_templates = [
        "Команда быстро решила проблему клиента и получила благодарность.",
        "Пользователь отмечает высокое качество сервиса и поддержку.",
        "Диалог завершился конструктивно, участники довольны результатом.",
        "Сотрудник проявил эмпатию и предложил полезное решение.",
    ]
    neg_templates = [
        "Клиент раздражен из-за долгого ожидания и отсутствия ответа.",
        "Общение завершилось конфликтом и недовольством сторон.",
        "Пользователь жалуется на ошибки и неработающие функции.",
        "В сообщении много критики, разочарования и недоверия.",
    ]

    rows: list[dict[str, str]] = []
    for _ in range(n):
        label = rng.choice(["positive", "negative"])
        text = rng.choice(pos_templates if label == "positive" else neg_templates)
        rows.append({"text": text, "label": label})
    return rows


def _save_jsonl(rows, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
