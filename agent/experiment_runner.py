#!/usr/bin/env python3
"""Run diploma experiments on remote Colab through colab-mcp."""

from __future__ import annotations

import argparse
import json
import shlex
import time
from dataclasses import dataclass
from typing import Any

from agent.logger import ExperimentLogger
from agent.mcp_client import ColabMCPClient, parse_json_lines


@dataclass(frozen=True)
class ExperimentSpec:
    exp_id: str
    mode: str
    reward: str
    min_gpu_ram_gb: float
    timeout_sec: int


EXPERIMENT_CHAIN: list[ExperimentSpec] = [
    ExperimentSpec("EXP-01", "baseline", "none", 3.5, 6 * 3600),
    ExperimentSpec("EXP-02", "grpo", "accuracy", 8.0, 8 * 3600),
    ExperimentSpec("EXP-03", "grpo", "reasoning", 8.0, 8 * 3600),
    ExperimentSpec("EXP-04", "grpo", "binary", 8.0, 8 * 3600),
    ExperimentSpec("EXP-05", "ppo", "reasoning", 13.0, 10 * 3600),
    ExperimentSpec("EXP-06", "dapo", "entropy", 9.0, 8 * 3600),
    ExperimentSpec("EXP-07", "lambda_grpo", "lambda_grpo", 9.0, 8 * 3600),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EXP-01..EXP-07 on Colab via MCP")
    parser.add_argument("--colab-url", required=True, help="Public URL of colab-mcp server")
    parser.add_argument("--colab-api-key", default="", help="Optional X-API-Key")
    parser.add_argument("--repo-url", default="https://github.com/Pancake2021/research_work_by_a_student.git")
    parser.add_argument("--colab-repo-dir", default="/content/research_work_by_a_student")
    parser.add_argument("--local-root", default=".", help="Local project root for logs/artifacts")
    parser.add_argument("--dataset", default="iemocap", choices=["iemocap", "cmu_mosi", "local_json", "synthetic"])
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--poll-interval", type=int, default=20)
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--only", default="", help="Comma-separated EXP ids (e.g. EXP-01,EXP-02)")
    parser.add_argument("--download-artifacts", action="store_true", help="Download output tarball after each run")
    return parser.parse_args()


def ensure_remote_repo(client: ColabMCPClient, repo_url: str, repo_dir: str) -> None:
    cmd = f"""
set -euo pipefail
if [ ! -d {shlex.quote(repo_dir)}/.git ]; then
  git clone {shlex.quote(repo_url)} {shlex.quote(repo_dir)}
else
  cd {shlex.quote(repo_dir)}
  git fetch --all || true
  git pull --ff-only || true
fi
cd {shlex.quote(repo_dir)}
python -m pip install -q --upgrade pip setuptools wheel
python -m pip install -q --ignore-installed blinker
python -m pip install -q flask requests pyngrok psutil
python -m pip install -q -r requirements.txt
""".strip()
    resp = client.execute_cell(cmd, timeout=2400, wait=True)
    if resp.get("status") != "ok":
        raise RuntimeError(f"bootstrap failed: {resp}")


def validate_runtime(info: dict[str, Any], min_gpu_ram: float) -> None:
    accelerator = info.get("accelerator")
    gpu_ram = info.get("gpu_ram_gb") or 0

    if accelerator not in {"gpu", "tpu"}:
        raise RuntimeError(f"Colab runtime is not GPU/TPU: {info}")

    if accelerator == "gpu" and gpu_ram < min_gpu_ram:
        raise RuntimeError(
            f"Insufficient GPU RAM: required >= {min_gpu_ram} GB, got {gpu_ram} GB"
        )


def run_experiment(
    client: ColabMCPClient,
    logger: ExperimentLogger,
    spec: ExperimentSpec,
    args: argparse.Namespace,
) -> None:
    runtime = client.get_runtime_info()
    validate_runtime(runtime, spec.min_gpu_ram_gb)

    run_meta = logger.create_run(
        method=spec.mode,
        reward_fn=spec.reward,
        config={
            "exp_id": spec.exp_id,
            "dataset": args.dataset,
            "train_size": args.train_size,
            "test_size": args.test_size,
        },
    )
    run_id = run_meta["run_id"]
    logger.log_runtime(run_id, runtime)

    remote_output = f"/content/outputs/{run_id}"

    reward_arg = "" if spec.mode == "baseline" else f" --reward {spec.reward}"
    cmd = (
        f"cd {shlex.quote(args.colab_repo_dir)} && "
        f"python scripts/run_full_pipeline.py --mode {spec.mode}{reward_arg} "
        f"--dataset {args.dataset} --train-size {args.train_size} --test-size {args.test_size} "
        f"--output-dir {remote_output} --run-id {run_id} --json-metrics"
    )

    execute_resp = client.execute_cell(cmd, timeout=spec.timeout_sec, wait=False)
    if execute_resp.get("status") not in {"started", "ok"}:
        logger.finish_run(run_id, "failed")
        raise RuntimeError(f"failed to start {spec.exp_id}: {execute_resp}")

    seen_lines: set[str] = set()
    max_step = 0
    final_eval: dict[str, Any] = {}

    while True:
        poll = client.stream_logs(last_n_lines=400)
        lines = poll.get("lines", [])
        new_lines: list[str] = []

        for line in lines:
            if line in seen_lines:
                continue
            seen_lines.add(line)
            new_lines.append(line)
            if line.startswith("[stderr]"):
                logger.append_stderr(run_id, line)
            else:
                logger.append_stdout(run_id, line)

        events = parse_json_lines(new_lines)
        for event in events:
            event_type = event.get("event", "unknown")
            step = int(event.get("step", max_step))
            if step > max_step:
                max_step = step
            logger.log_metrics(run_id, step, event)
            if event_type in {"baseline_complete", "training_complete", "pipeline_finished"}:
                final_eval = event

        status = poll.get("execution_status")
        last_result = poll.get("last_result", {})
        if status == "idle" and last_result:
            if last_result.get("status") == "ok":
                logger.finish_run(run_id, "success", eval_metrics=final_eval)
            else:
                logger.finish_run(run_id, "failed", eval_metrics=final_eval)
                raise RuntimeError(f"{spec.exp_id} failed: {last_result}")
            break

        time.sleep(args.poll_interval)

    # Track remote artifacts (download can be done later or on-demand)
    logger.save_artifact(run_id, "remote_output_dir", remote_output)
    logger.save_artifact(run_id, "remote_results_json", f"{remote_output}/results")

    if args.download_artifacts:
        local_archive = f"{args.local_root}/results/{run_id}/remote_outputs.tgz"
        remote_archive = f"/content/{run_id}_outputs.tgz"
        archive_cmd = (
            f"set -e; cd {shlex.quote(remote_output)} && "
            f"tar -czf {shlex.quote(remote_archive)} ."
        )
        archive_resp = client.execute_cell(archive_cmd, timeout=1800, wait=True)
        if archive_resp.get("status") == "ok":
            dl = client.download_file(remote_archive, local_archive)
            logger.save_artifact(run_id, "local_artifact_archive", dl["local_path"])


def main() -> None:
    args = parse_args()

    client = ColabMCPClient(args.colab_url, api_key=args.colab_api_key, timeout=180)
    logger = ExperimentLogger(args.local_root)

    health = client.health()
    if health.get("status") != "ok":
        raise RuntimeError(f"colab-mcp is unavailable: {health}")

    if not args.skip_bootstrap:
        ensure_remote_repo(client, args.repo_url, args.colab_repo_dir)

    selected = set(x.strip() for x in args.only.split(",") if x.strip())
    chain = [x for x in EXPERIMENT_CHAIN if not selected or x.exp_id in selected]

    if not chain:
        raise ValueError("No experiments selected")

    for spec in chain:
        print(f"\n=== {spec.exp_id} {spec.mode}/{spec.reward} ===", flush=True)
        run_experiment(client, logger, spec, args)

    summary = logger.get_experiment_summary()
    print(json.dumps({"status": "ok", "runs": len(summary)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
