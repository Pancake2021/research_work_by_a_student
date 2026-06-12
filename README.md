<p align="center">
  <h1 align="center">SOC Narrative: Small LLMs for UEBA / Insider Threat Detection</h1>
  <p align="center">
    <em>Can small open-weight LLMs detect insider threats from user logs — and write a useful SOC investigation card?</em>
  </p>
  <p align="center">
    <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
    <a href="https://huggingface.co/Pankei"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Pankei-orange" alt="Hugging Face"></a>
    <a href="https://github.com/Pancake2021/research_work_by_a_student"><img src="https://img.shields.io/badge/GitHub-Repo-green" alt="GitHub"></a>
    <a href="https://pytorch.org"><img src="https://img.shields.io/badge/Framework-PyTorch%20%2B%20TRL-red" alt="Framework"></a>
  </p>
</p>

---

## Overview

Insider threats remain one of the hardest security problems: the attacker already has valid credentials and their actions blend into normal daily activity.

This project explores whether **small open-weight LLMs** (3B–14B parameters) can:
1. **Classify** user behavior as `normal`, `suspicious`, or `malicious`
2. **Explain** their decision with cited evidence from raw logs
3. **Generate** a structured SOC-style investigation card

We frame this as a **SOC Narrative** task: a model receives a user/day window of events from the **CERT Insider Threat Dataset R4.2** and must produce a structured answer with risk label, evidence flags, and reasoning.

**Key result**: **Few-shot Qwen3.5-9B achieves accuracy 0.86, macro F1 0.876, recall malicious 0.96** — no fine-tuning needed. GRPO strict128 further improves actionability (0.78 vs 0.68) at the cost of recall.

---

## Results

| Setting | Model | Accuracy | Macro F1 | Recall (malicious) | Valid Format | Actionability |
|---|---|---|---|---|---|---|
| Zero-shot | Qwen3-14B | 0.36 | 0.343 | 0.52 | 0.88 | 0.46 |
| Zero-shot | Qwen3.5-9B | 0.46 | 0.449 | 0.32 | 0.72 | 0.20 |
| **Few-shot** | **Qwen3.5-9B** | **0.86** | **0.876** | **0.96** | **0.92** | **0.84** |
| Few-shot | Qwen3-14B | 0.42 | 0.446 | 0.12 | 0.60 | 0.60 |
| Few-shot | Qwen3.5-35B-A3B (MoE) | 0.58 | 0.636 | 0.44 | 0.64 | 0.54 |
| SFT LoRA | Qwen3-14B | 0.74 | 0.735 | 0.88 | 0.68 | 0.68 |
| GRPO-32 | Qwen3-14B | 0.74 | 0.735 | 0.88 | 0.64 | 0.64 |
| **GRPO strict128** | **Qwen3-14B** | **0.84** | **0.839** | **0.76** | **0.78** | **0.78** |

**Balanced eval subset**: `dev_balanced_50` — 25 normal + 25 malicious, fixed across all experiments.

### Key Takeaways

- **Few-shot Qwen3.5-9B** is the best practical model — no training, high recall, excellent format compliance.
- **SFT LoRA** boosts recall malicious from 0.52 → 0.88 (Qwen3-14B) but reduces format quality.
- **GRPO strict128** recovers format and actionability (0.68 → 0.78) but recall drops to 0.76.
- RL fine-tuning (GRPO) on top of SFT does **not** universally improve classification — it trades recall for structure.

---

## Models on Hugging Face

LoRA adapters trained on `train_balanced_512` (256 normal + 256 malicious):

| Checkpoint | Base Model | Method | Metrics |
|---|---|---|---|
| [`soc-narrative-sft-qwen3-14b`](https://huggingface.co/Pankei/soc-narrative-sft-qwen3-14b) | Qwen/Qwen3-14B | SFT LoRA | Acc 0.74, Recall 0.88 |
| [`soc-narrative-grpo32-qwen3-14b`](https://huggingface.co/Pankei/soc-narrative-grpo32-qwen3-14b) | Qwen/Qwen3-14B | GRPO (32 steps) | Acc 0.74, Recall 0.88 |
| [`soc-narrative-grpo-strict128-qwen3-14b`](https://huggingface.co/Pankei/soc-narrative-grpo-strict128-qwen3-14b) | Qwen/Qwen3-14B | GRPO strict128 | Acc 0.84, Recall 0.76 |
| [`soc-narrative-sft-qwen3.5-9b`](https://huggingface.co/Pankei/soc-narrative-sft-qwen3.5-9b) | Qwen/Qwen3.5-9B | SFT LoRA | Acc 0.74, Recall 0.52 |

Evaluation dataset: [`soc-narrative-dev-balanced-50`](https://huggingface.co/datasets/Pankei/soc-narrative-dev-balanced-50)

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/Pancake2021/research_work_by_a_student.git
cd research_work_by_a_student
pip install -r requirements.txt

# 2. Run evaluation on the dev set (downloads model + dataset automatically)
python scripts/run_evaluation.py \
  --model-id Pankei/soc-narrative-sft-qwen3-14b \
  --dataset Pankei/soc-narrative-dev-balanced-50

# 3. Run a full bake-off
python scripts/model_bakeoff.py \
  --dataset-jsonl outputs/dev_balanced_50.jsonl \
  --registry configs/model_registry.yaml \
  --models qwen3_14b qwen3_5_9b \
  --limit 50
```

### Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pancake2021/research_work_by_a_student/blob/main/notebooks/soc_narrative_demo.ipynb)

---

## Project Structure

```
research_work_by_a_student/
├── configs/           # Experiment configs (model registry, tech stack, GRPO)
├── notebooks/         # Colab and demo notebooks
├── scripts/           # Training, evaluation, bake-off pipelines
├── src/
│   ├── data/          # CERT loader, scenario builder, preprocessing
│   ├── evaluation/    # UEBA metrics, error analysis
│   ├── models/        # Model loading, baselines
│   ├── rewards/       # Reward functions for GRPO (accuracy, format, evidence)
│   ├── training/      # GRPO, PPO, DPO trainers
│   └── visualization/ # Plots and charts
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Reward Functions

Three UEBA-specific reward functions for GRPO experiments:

| Reward | Signal | Range |
|---|---|---|
| `ueba_accuracy` | Correct risk label only | `{0.0, 1.0}` |
| `ueba_format` | Correct label + required fields present | `[0.0, 1.3]` |
| `ueba_evidence` | Label + format + evidence overlap – hallucination penalty | `[-0.5, 2.0]` |

---

## Tech Stack

| Component | Tool | Role |
|---|---|---|
| Dataset | pandas, datasets, CERT R4.2 | CSV loading, JSONL splits |
| Baseline | scikit-learn | Logistic Regression / RandomForest |
| Inference | transformers (primary), vLLM (optional) | Model evaluation |
| Training | TRL + PEFT + bitsandbytes | SFT, GRPO with LoRA |
| Quantization | bitsandbytes NF4 | 4-bit for 8GB GPU compatibility |
| Tracking | wandb + local outputs | Metrics, configs, predictions |

---

## Citation

```bibtex
@misc{pankeev2025socnarrative,
  title        = {SOC Narrative: Small LLMs for UEBA / Insider Threat Detection},
  author       = {Pankeev, Gleb},
  year         = {2026},
  howpublished = {\url{https://github.com/Pancake2021/research_work_by_a_student}},
  note         = {Undergraduate research project, Samara University}
}
```

---

## License

Apache 2.0. The CERT Insider Threat Dataset R4.2 is subject to [CERT's original license](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099).
