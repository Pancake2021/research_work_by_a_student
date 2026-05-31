#!/usr/bin/env python3
"""
Prepare CERT Insider Threat data as UEBA prompt/response JSONL splits.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cert_loader import build_cert_scenarios, split_by_user, write_jsonl
from src.data.scenario_builder import build_scenario_record


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare CERT UEBA scenarios")
    parser.add_argument("--data-dir", help="Directory with CERT CSV files")
    parser.add_argument("--labels", help="Optional CSV/JSON/JSONL labels user/date -> risk_label")
    parser.add_argument("--output-dir", default="./outputs/cert_ueba_dataset")
    parser.add_argument("--max-rows-per-file", type=int)
    parser.add_argument("--min-events-per-group", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-smoke", action="store_true", help="Write a tiny synthetic UEBA dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic_smoke:
        examples = _synthetic_examples()
    else:
        if not args.data_dir:
            raise SystemExit("--data-dir is required unless --synthetic-smoke is set")
        examples = build_cert_scenarios(
            data_dir=args.data_dir,
            labels_path=args.labels,
            max_rows_per_file=args.max_rows_per_file,
            min_events_per_group=args.min_events_per_group,
        )

    splits = split_by_user(examples, seed=args.seed)
    for split_name, split_examples in splits.items():
        write_jsonl(split_examples, output_dir / f"{split_name}.jsonl")

    summary = {
        "total": len(examples),
        "splits": {name: len(items) for name, items in splits.items()},
        "labels": dict(Counter(example["risk_label"] for example in examples)),
        "users": len({example["user_id"] for example in examples}),
        "seed": args.seed,
        "source": "synthetic_smoke" if args.synthetic_smoke else str(args.data_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _synthetic_examples():
    normal = build_scenario_record(
        "USR001",
        "2026-03-01",
        [
            {"event_type": "logon", "user_id": "USR001", "timestamp": "2026-03-01 09:10:00", "activity": "logon"},
            {"event_type": "http", "user_id": "USR001", "timestamp": "2026-03-01 10:05:00", "url": "intranet.company.com"},
        ],
        "normal",
    )
    suspicious = build_scenario_record(
        "USR002",
        "2026-03-02",
        [
            {"event_type": "logon", "user_id": "USR002", "timestamp": "2026-03-02 22:10:00", "activity": "logon"},
            {"event_type": "device", "user_id": "USR002", "timestamp": "2026-03-02 22:15:00", "activity": "connect usb"},
            {"event_type": "file", "user_id": "USR002", "timestamp": "2026-03-02 22:20:00", "filename": "project.zip"},
        ],
        "suspicious",
    )
    malicious = build_scenario_record(
        "USR003",
        "2026-03-03",
        [
            {"event_type": "logon", "user_id": "USR003", "timestamp": "2026-03-03 23:10:00", "activity": "logon"},
            {"event_type": "device", "user_id": "USR003", "timestamp": "2026-03-03 23:15:00", "activity": "connect usb"},
            *[
                {
                    "event_type": "file",
                    "user_id": "USR003",
                    "timestamp": "2026-03-03 23:20:00",
                    "filename": f"secret_{i}.docx",
                }
                for i in range(30)
            ],
            {"event_type": "email", "user_id": "USR003", "timestamp": "2026-03-03 23:55:00", "to": "out@example.org"},
            {"event_type": "http", "user_id": "USR003", "timestamp": "2026-03-03 23:57:00", "url": "dropbox.com/upload"},
        ],
        "malicious",
    )
    return [item.to_example() for item in (normal, suspicious, malicious)]


if __name__ == "__main__":
    main()
