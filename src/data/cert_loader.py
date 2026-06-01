"""
cert_loader.py
==============
Загрузка CERT Insider Threat CSV и сбор UEBA-сценариев.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from src.data.scenario_builder import ScenarioRecord, build_scenario_record


CERT_EVENT_FILES = {
    "logon": "logon.csv",
    "device": "device.csv",
    "file": "file.csv",
    "email": "email.csv",
    "http": "http.csv",
}


def load_cert_events(
    data_dir: str | Path,
    max_rows_per_file: int | None = None,
    csv_chunksize: int = 100_000,
) -> list[dict[str, Any]]:
    """Загружает доступные CERT CSV-файлы в общий список событий."""
    return list(iter_cert_events(data_dir, max_rows_per_file=max_rows_per_file, csv_chunksize=csv_chunksize))


def iter_cert_events(
    data_dir: str | Path,
    max_rows_per_file: int | None = None,
    csv_chunksize: int = 100_000,
) -> Iterator[dict[str, Any]]:
    """Стримит CERT CSV-файлы чанками, чтобы не держать весь датасет в pandas DataFrame."""
    import pandas as pd

    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"CERT data directory does not exist: {root}")

    csv_paths = _find_cert_event_files(root)
    missing = sorted(set(CERT_EVENT_FILES) - set(csv_paths))
    if missing:
        found = ", ".join(f"{event_type}={path}" for event_type, path in sorted(csv_paths.items()))
        raise FileNotFoundError(
            "CERT data directory does not contain required event CSV files. "
            f"Missing: {', '.join(missing)}. Found: {found or 'none'}. "
            "Expected files: logon.csv, device.csv, file.csv, email.csv, http.csv. "
            "If Kaggle unpacked nested folders, pass the directory that contains those files "
            "or keep using the current path after this recursive lookup fix."
        )

    for event_type, path in csv_paths.items():
        reader = pd.read_csv(path, nrows=max_rows_per_file, chunksize=csv_chunksize)
        total_rows = 0
        for chunk_index, frame in enumerate(reader, start=1):
            total_rows += len(frame)
            print(
                f"[cert_loader] {event_type}: chunk={chunk_index} rows={len(frame)} total={total_rows} file={path}",
                file=sys.stderr,
                flush=True,
            )
            for row in frame.to_dict(orient="records"):
                row = {str(key).lower(): value for key, value in row.items()}
                row["event_type"] = event_type
                row["user_id"] = row.get("user") or row.get("userid") or row.get("user_id") or ""
                row["timestamp"] = row.get("date") or row.get("time") or row.get("timestamp") or ""
                yield row


def _find_cert_event_files(root: Path) -> dict[str, Path]:
    """Find CERT event CSV files directly or in Kaggle/Figshare nested folders."""
    found: dict[str, Path] = {}
    for event_type, filename in CERT_EVENT_FILES.items():
        direct = root / filename
        if direct.exists():
            found[event_type] = direct
            continue

        matches = sorted(root.rglob(filename))
        if matches:
            found[event_type] = matches[0]
    return found


def build_cert_scenarios(
    data_dir: str | Path,
    labels_path: str | Path | None = None,
    max_rows_per_file: int | None = None,
    csv_chunksize: int = 100_000,
    max_events_per_group: int = 300,
    min_events_per_group: int = 2,
    checkpoint_path: str | Path | None = None,
    resume_checkpoint: bool = False,
) -> list[dict[str, Any]]:
    """Создает prompt/response examples из CERT событий."""
    import pandas as pd

    label_map = load_label_map(labels_path) if labels_path else {}
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    state = _load_build_checkpoint(checkpoint) if resume_checkpoint and checkpoint else None
    if state:
        grouped = defaultdict(list, state["grouped"])
        total_events = int(state["total_events"])
        kept_events = int(state["kept_events"])
        rows_done_by_type = dict(state["rows_done_by_type"])
        chunks_done_by_type = dict(state["chunks_done_by_type"])
        completed_event_types = set(state.get("completed_event_types", []))
        print(f"[cert_loader] resumed checkpoint={checkpoint}", file=sys.stderr, flush=True)
    else:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        total_events = 0
        kept_events = 0
        rows_done_by_type: dict[str, int] = {}
        chunks_done_by_type: dict[str, int] = {}
        completed_event_types: set[str] = set()

    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"CERT data directory does not exist: {root}")
    csv_paths = _find_cert_event_files(root)
    missing = sorted(set(CERT_EVENT_FILES) - set(csv_paths))
    if missing:
        found = ", ".join(f"{event_type}={path}" for event_type, path in sorted(csv_paths.items()))
        raise FileNotFoundError(
            "CERT data directory does not contain required event CSV files. "
            f"Missing: {', '.join(missing)}. Found: {found or 'none'}. "
            "Expected files: logon.csv, device.csv, file.csv, email.csv, http.csv."
        )

    for event_type, path in csv_paths.items():
        if event_type in completed_event_types:
            print(f"[cert_loader] {event_type}: already complete", file=sys.stderr, flush=True)
            continue
        rows_done = rows_done_by_type.get(event_type, 0)
        if max_rows_per_file is not None and rows_done >= max_rows_per_file:
            print(f"[cert_loader] {event_type}: already complete rows={rows_done}", file=sys.stderr, flush=True)
            completed_event_types.add(event_type)
            continue

        nrows = None if max_rows_per_file is None else max_rows_per_file - rows_done
        skiprows = range(1, rows_done + 1) if rows_done else None
        reader = pd.read_csv(path, nrows=nrows, chunksize=csv_chunksize, skiprows=skiprows)
        total_rows_for_file = rows_done
        for frame in reader:
            if len(frame) == 0:
                continue
            chunk_index = chunks_done_by_type.get(event_type, 0) + 1
            total_rows_for_file += len(frame)
            print(
                f"[cert_loader] {event_type}: chunk={chunk_index} rows={len(frame)} "
                f"total={total_rows_for_file} file={path}",
                file=sys.stderr,
                flush=True,
            )
            for row in frame.to_dict(orient="records"):
                event = {str(key).lower(): value for key, value in row.items()}
                event["event_type"] = event_type
                event["user_id"] = event.get("user") or event.get("userid") or event.get("user_id") or ""
                event["timestamp"] = event.get("date") or event.get("time") or event.get("timestamp") or ""
                total_events += 1
                user_id = str(event.get("user_id") or "")
                timestamp = str(event.get("timestamp") or "")
                date = timestamp[:10] if len(timestamp) >= 10 else "unknown"
                if not user_id:
                    continue
                key = (user_id, date)
                if len(grouped[key]) < max_events_per_group:
                    grouped[key].append(event)
                    kept_events += 1

            rows_done_by_type[event_type] = total_rows_for_file
            chunks_done_by_type[event_type] = chunk_index
            if checkpoint:
                _save_build_checkpoint(
                    checkpoint,
                    {
                        "version": 1,
                        "grouped": dict(grouped),
                        "total_events": total_events,
                        "kept_events": kept_events,
                        "rows_done_by_type": rows_done_by_type,
                        "chunks_done_by_type": chunks_done_by_type,
                        "completed_event_types": sorted(completed_event_types),
                    },
                )
        completed_event_types.add(event_type)
        if checkpoint:
            _save_build_checkpoint(
                checkpoint,
                {
                    "version": 1,
                    "grouped": dict(grouped),
                    "total_events": total_events,
                    "kept_events": kept_events,
                    "rows_done_by_type": rows_done_by_type,
                    "chunks_done_by_type": chunks_done_by_type,
                    "completed_event_types": sorted(completed_event_types),
                },
            )

    records: list[ScenarioRecord] = []
    for (user_id, date), group_events in grouped.items():
        if len(group_events) < min_events_per_group:
            continue
        label = label_map.get((user_id, date)) or label_map.get((user_id, "*"))
        records.append(build_scenario_record(user_id, date, group_events, risk_label=label))

    print(
        "[cert_loader] scenarios="
        f"{len(records)} groups={len(grouped)} events_seen={total_events} events_kept={kept_events} "
        f"max_events_per_group={max_events_per_group}",
        file=sys.stderr,
        flush=True,
    )
    return [record.to_example() for record in records]


def _load_build_checkpoint(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    with path.open("rb") as f:
        state = pickle.load(f)
    if state.get("version") != 1:
        raise ValueError(f"Unsupported CERT checkpoint version: {state.get('version')}")
    return state


def _save_build_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(state, f)
    tmp_path.replace(path)


def split_by_user(
    examples: list[dict[str, Any]],
    train_ratio: float = 0.7,
    dev_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Разбивает данные по user_id, чтобы один пользователь не попадал в разные split."""
    import random

    users = sorted({str(example["user_id"]) for example in examples})
    rng = random.Random(seed)
    rng.shuffle(users)
    n_train = int(len(users) * train_ratio)
    n_dev = int(len(users) * dev_ratio)
    if len(users) >= 3:
        n_train = max(1, min(n_train, len(users) - 2))
        n_dev = max(1, min(n_dev, len(users) - n_train - 1))
    train_users = set(users[:n_train])
    dev_users = set(users[n_train : n_train + n_dev])

    splits = {"train": [], "dev": [], "test": []}
    for example in examples:
        user_id = str(example["user_id"])
        if user_id in train_users:
            splits["train"].append(example)
        elif user_id in dev_users:
            splits["dev"].append(example)
        else:
            splits["test"].append(example)
    return splits


def load_label_map(path: str | Path) -> dict[tuple[str, str], str]:
    """Загружает явную разметку user/date -> risk_label из CSV или JSONL."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    rows: list[dict[str, Any]]
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        rows = loaded if isinstance(loaded, list) else loaded.get("labels", [])
    else:
        import pandas as pd

        rows = pd.read_csv(path).to_dict(orient="records")

    label_map = {}
    for row in rows:
        user_id = str(row.get("user_id") or row.get("user") or row.get("User") or "")
        date = str(row.get("date") or row.get("day") or row.get("Date") or "*")
        label = str(row.get("risk_label") or row.get("label") or row.get("class") or "").lower()
        if user_id and label:
            label_map[(user_id, date[:10] if date != "*" else "*")] = label
    return label_map


def write_jsonl(examples: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
