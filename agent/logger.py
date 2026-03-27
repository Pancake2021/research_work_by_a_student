#!/usr/bin/env python3
"""Local run logger for Colab experiments."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LoggerConfig:
    root_dir: Path


class ExperimentLogger:
    def __init__(self, root_dir: str | Path):
        self.config = LoggerConfig(root_dir=Path(root_dir))
        self.config.root_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir = self.config.root_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.config.root_dir / "experiment_db.json"
        self._lock = threading.Lock()
        self._ensure_db()

    def _ensure_db(self) -> None:
        if not self.db_path.exists():
            self._write_db({"runs": []})

    def _read_db(self) -> dict[str, Any]:
        if not self.db_path.exists():
            return {"runs": []}
        return json.loads(self.db_path.read_text(encoding="utf-8"))

    def _write_db(self, payload: dict[str, Any]) -> None:
        self.db_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _run_dir(self, run_id: str) -> Path:
        path = self.runs_dir / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def create_run(self, method: str, reward_fn: str, config: dict[str, Any]) -> dict[str, str]:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"{method}_{reward_fn}_{ts}".replace("-", "_")
        run_dir = self._run_dir(run_id)

        record = {
            "run_id": run_id,
            "method": method,
            "reward_fn": reward_fn,
            "status": "running",
            "started_at": _now_iso(),
            "finished_at": None,
            "duration_hours": None,
            "runtime": {},
            "training": {"steps": []},
            "eval": {},
            "config_snapshot": config,
            "artifacts": {},
        }

        with self._lock:
            db = self._read_db()
            db.setdefault("runs", []).append(record)
            self._write_db(db)

        (run_dir / "metrics_stream.jsonl").touch(exist_ok=True)
        (run_dir / "stdout.log").touch(exist_ok=True)
        (run_dir / "stderr.log").touch(exist_ok=True)

        return {"run_id": run_id}

    def log_metrics(self, run_id: str, step: int, metrics: dict[str, Any]) -> dict[str, str]:
        row = {
            "ts": _now_iso(),
            "step": step,
            "metrics": metrics,
        }

        run_dir = self._run_dir(run_id)
        metrics_file = run_dir / "metrics_stream.jsonl"
        with metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        with self._lock:
            db = self._read_db()
            run = self._find_run(db, run_id)
            run["training"]["steps"].append(row)
            self._write_db(db)

        return {"status": "ok"}

    def log_runtime(self, run_id: str, runtime_info: dict[str, Any]) -> dict[str, str]:
        with self._lock:
            db = self._read_db()
            run = self._find_run(db, run_id)
            run["runtime"] = runtime_info
            self._write_db(db)
        return {"status": "ok"}

    def save_artifact(self, run_id: str, artifact_type: str, path: str) -> dict[str, str]:
        with self._lock:
            db = self._read_db()
            run = self._find_run(db, run_id)
            run.setdefault("artifacts", {})[artifact_type] = path
            self._write_db(db)
        return {"status": "ok"}

    def finish_run(self, run_id: str, status: str, eval_metrics: dict[str, Any] | None = None) -> dict[str, str]:
        finished_at = datetime.now(timezone.utc)
        with self._lock:
            db = self._read_db()
            run = self._find_run(db, run_id)
            started = datetime.fromisoformat(run["started_at"])
            duration_h = round((finished_at - started).total_seconds() / 3600.0, 4)
            run["status"] = status
            run["finished_at"] = finished_at.isoformat()
            run["duration_hours"] = duration_h
            if eval_metrics:
                run["eval"] = eval_metrics
            self._write_db(db)
        return {"status": "ok"}

    def get_experiment_summary(self) -> list[dict[str, Any]]:
        with self._lock:
            db = self._read_db()
            return db.get("runs", [])

    def append_stdout(self, run_id: str, line: str) -> None:
        run_dir = self._run_dir(run_id)
        with (run_dir / "stdout.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def append_stderr(self, run_id: str, line: str) -> None:
        run_dir = self._run_dir(run_id)
        with (run_dir / "stderr.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    @staticmethod
    def _find_run(db: dict[str, Any], run_id: str) -> dict[str, Any]:
        for run in db.get("runs", []):
            if run.get("run_id") == run_id:
                return run
        raise KeyError(f"run not found: {run_id}")
