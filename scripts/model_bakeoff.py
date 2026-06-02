#!/usr/bin/env python3
"""
Run zero/few-shot bake-off for small LLM candidates on UEBA dev-set.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cert_loader import read_jsonl, write_jsonl
from src.data.preprocessor import UEBA_SYSTEM_PROMPT, build_ueba_prompt
from src.evaluation.ueba_metrics import evaluate_ueba_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Small LLM bake-off for UEBA scenarios")
    parser.add_argument("--dataset-jsonl", required=True, help="Dev/test JSONL prepared by prepare_cert_dataset.py")
    parser.add_argument("--registry", default="./configs/model_registry.yaml")
    parser.add_argument("--models", nargs="*", help="Registry keys to evaluate")
    parser.add_argument("--output-dir", default="./outputs/model_bakeoff")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--few-shot-k", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--max-model-len", type=int, default=4096, help="vLLM max_model_len safety cap")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="vLLM GPU memory fraction")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable vLLM CUDA graph capture/warmup")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Do not load models; emit deterministic mock responses")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    output_root.mkdir(parents=True, exist_ok=True)
    examples = read_jsonl(args.dataset_jsonl)[: args.limit]
    registry = load_registry(args.registry)
    selected = select_models(registry, args.models, include_optional=args.include_optional)
    shot_examples = examples[: args.few_shot_k] if args.few_shot_k else []
    eval_examples = examples[args.few_shot_k :] if args.few_shot_k else examples
    shutil.copyfile(args.registry, output_root / "model_registry.yaml")
    (output_root / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = []
    for model_key, model_cfg in selected.items():
        model_dir = output_root / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            if args.mock:
                responses = [mock_response(example) for example in eval_examples]
                load_status = "mock"
            elif model_cfg.get("task") != "text-generation":
                responses = ["" for _ in eval_examples]
                load_status = f"skipped_task:{model_cfg.get('task')}"
            else:
                if args.backend == "vllm":
                    responses = run_vllm_model(
                        model_cfg,
                        eval_examples,
                        shot_examples,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        max_model_len=args.max_model_len,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        enforce_eager=args.enforce_eager,
                    )
                else:
                    responses = run_transformers_model(
                        model_cfg,
                        eval_examples,
                        shot_examples,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
                load_status = "ok"
            metrics = evaluate_ueba_predictions(eval_examples, responses)
            metrics.update(
                {
                    "model_key": model_key,
                    "hf_id": model_cfg.get("hf_id"),
                    "status": load_status,
                    "backend": "mock" if args.mock else args.backend,
                    "elapsed_sec": round(time.time() - t0, 2),
                }
            )
        except Exception as exc:
            responses = []
            metrics = {
                "model_key": model_key,
                "hf_id": model_cfg.get("hf_id"),
                "status": "failed",
                "backend": "mock" if args.mock else args.backend,
                "error": repr(exc),
                "elapsed_sec": round(time.time() - t0, 2),
            }

        write_jsonl(
            [
                {
                    "user_id": example.get("user_id"),
                    "date": example.get("date"),
                    "risk_label": example.get("risk_label"),
                    "evidence": example.get("evidence"),
                    "response": response,
                }
                for example, response in zip(eval_examples, responses)
            ],
            model_dir / "predictions.jsonl",
        )
        (model_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        (model_dir / "samples.md").write_text(render_samples(eval_examples, responses), encoding="utf-8")
        summary.append(metrics)

    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "summary.md").write_text(render_summary(summary), encoding="utf-8")
    print(render_summary(summary))


def load_registry(path: str | Path) -> dict[str, dict[str, Any]]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data.get("models", data)


def select_models(registry, model_keys, include_optional=False):
    selected = {}
    keys = model_keys or sorted(registry, key=lambda key: registry[key].get("priority", 999))
    for key in keys:
        cfg = registry[key]
        if cfg.get("role") == "optional" and not include_optional and model_keys is None:
            continue
        selected[key] = cfg
    return selected


def run_transformers_model(model_cfg, eval_examples, shot_examples, max_new_tokens, temperature):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = model_cfg["hf_id"]
    quantization_config = None
    if model_cfg.get("load_in_4bit", True):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=model_cfg.get("trust_remote_code", False))
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    responses = []
    for example in eval_examples:
        prompt = build_messages(example, shot_examples, model_cfg)
        inputs = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            **(model_cfg.get("chat_template_kwargs") or {}),
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=tokenizer.pad_token_id,
            )
        responses.append(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True))
    return responses


def run_vllm_model(
    model_cfg,
    eval_examples,
    shot_examples,
    max_new_tokens,
    temperature,
    max_model_len,
    gpu_memory_utilization,
    enforce_eager,
):
    """Runs batched inference with vLLM for fast bake-off/evaluation."""
    from vllm import LLM, SamplingParams

    model_id = model_cfg["hf_id"]
    llm_kwargs = {
        "model": model_id,
        "trust_remote_code": model_cfg.get("trust_remote_code", False),
        "dtype": "float16",
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": enforce_eager,
    }
    if model_cfg.get("vllm_quantization"):
        llm_kwargs["quantization"] = model_cfg["vllm_quantization"]
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    prompts = []
    for example in eval_examples:
        messages = build_messages(example, shot_examples, model_cfg)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **(model_cfg.get("chat_template_kwargs") or {}),
        )
        prompts.append(prompt)
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def build_messages(example, shots, model_cfg):
    messages = [{"role": "system", "content": UEBA_SYSTEM_PROMPT}]
    for shot in shots:
        messages.append({"role": "user", "content": build_ueba_prompt(shot["scenario"])})
        messages.append({"role": "assistant", "content": shot["response"]})
    messages.append({"role": "user", "content": build_ueba_prompt(example["scenario"])})
    return messages


def mock_response(example):
    evidence = "; ".join(example.get("evidence") or ["нет выраженных признаков риска"])
    return (
        f"Риск: {example.get('risk_label', 'normal')}\n"
        f"Признаки: {evidence}\n"
        "Обоснование: mock-ответ для проверки пайплайна без загрузки модели."
    )


def render_samples(examples, responses, limit=8):
    chunks = ["# Bake-off Samples\n"]
    for example, response in list(zip(examples, responses))[:limit]:
        chunks.append(f"## {example.get('user_id')} {example.get('date')}\n")
        chunks.append(f"Expected: `{example.get('risk_label')}`\n\n")
        chunks.append("```text\n" + example.get("scenario", "")[:1200] + "\n```\n")
        chunks.append("Response:\n\n```text\n" + str(response)[:1200] + "\n```\n")
    return "\n".join(chunks)


def render_summary(summary):
    lines = [
        "# Model Bake-off Summary",
        "",
        "| model | backend | status | accuracy | macro_f1 | recall_malicious | valid_format | evidence_hit | elapsed_sec |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            "| {model_key} | {backend} | {status} | {accuracy} | {macro_f1} | {recall_malicious} | "
            "{valid_format_rate} | {evidence_hit_rate} | {elapsed_sec} |".format(
                model_key=item.get("model_key"),
                backend=item.get("backend", "-"),
                status=item.get("status"),
                accuracy=item.get("accuracy", "-"),
                macro_f1=item.get("macro_f1", "-"),
                recall_malicious=item.get("recall_malicious", "-"),
                valid_format_rate=item.get("valid_format_rate", "-"),
                evidence_hit_rate=item.get("evidence_hit_rate", "-"),
                elapsed_sec=item.get("elapsed_sec", "-"),
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
