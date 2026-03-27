# Agent Instructions

1. Use `agent.experiment_runner` as the only entrypoint for remote experiment chain.
2. Do not reorder experiments; keep EXP-01..EXP-07 strict sequence.
3. Parse only strict JSON lines from `run_full_pipeline.py` for metrics.
4. On any failure/OOM, stop chain and mark run status as `failed`.
5. Persist metrics locally in `experiment_db.json` and `runs/<run_id>/metrics_stream.jsonl`.
6. Prefer Colab GPU runtime; TPU is allowed but may not support all training scripts.
