#!/usr/bin/env python3
"""
Classical ML baseline for CERT/UEBA scenarios.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cert_loader import read_jsonl
from src.data.scenario_builder import extract_behavior_features
from src.evaluation.ueba_metrics import evaluate_ueba_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Run classical UEBA baseline")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--output-dir", default="./outputs/ueba_baseline")
    parser.add_argument("--model", choices=["logreg", "rf"], default="logreg")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = read_jsonl(args.train_jsonl)
    test = read_jsonl(args.test_jsonl)
    x_train, y_train = featurize(train)
    x_test, _ = featurize(test)

    if len(set(y_train)) < 2:
        from sklearn.dummy import DummyClassifier

        clf = DummyClassifier(strategy="most_frequent")
    elif args.model == "rf":
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    else:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test).tolist()
    responses = [
        f"Риск: {label}\nПризнаки: baseline numeric features\nОбоснование: классический ML baseline по агрегированным признакам."
        for label in predictions
    ]
    metrics = evaluate_ueba_predictions(test, responses)
    metrics.update(
        {
            "method": args.model,
            "n_train": len(train),
            "n_test": len(test),
            "train_label_distribution": dict(Counter(y_train)),
            "estimator": clf.__class__.__name__,
        }
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def featurize(examples):
    rows = []
    labels = []
    for example in examples:
        features = extract_behavior_features(example.get("source_events", []))
        rows.append(
            [
                features.get("total_events", 0),
                features.get("after_hours_events", 0),
                features.get("external_emails", 0),
                features.get("file_events", 0),
                features.get("usb_events", 0),
                features.get("suspicious_urls", 0),
                features.get("logon_failures", 0),
            ]
        )
        labels.append(example.get("risk_label") or example.get("label"))
    return rows, labels


if __name__ == "__main__":
    main()
