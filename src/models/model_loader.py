"""
model_loader.py
===============
Загрузка Qwen-2.5-1.5B-Instruct через Unsloth с 4-bit квантизацией и LoRA.

Использование:
    model, tokenizer = load_model()
    inference_model = prepare_for_inference(model)
"""

import os
from typing import Tuple, Optional

from src.data.data_utils import logger, check_gpu


# ──────────────────────────────────────────────────────────────────────────────
# Настройки по умолчанию
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_NAME   = os.getenv("MODEL_NAME", "unsloth/Qwen2.5-1.5B-Instruct")
DEFAULT_MAX_SEQ_LEN  = int(os.getenv("MAX_SEQ_LENGTH", 2048))
DEFAULT_LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "true").lower() == "true"

# LoRA гиперпараметры
LORA_RANK        = 16     # rank адаптера (обычно 8–64)
LORA_ALPHA       = 32     # scaling = alpha / rank
LORA_DROPOUT     = 0.0    # 0 даёт лучшие результаты при обучении
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ──────────────────────────────────────────────────────────────────────────────
# Загрузка модели
# ──────────────────────────────────────────────────────────────────────────────

def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    max_seq_length: int = DEFAULT_MAX_SEQ_LEN,
    load_in_4bit: bool = DEFAULT_LOAD_IN_4BIT,
    with_lora: bool = True,
    lora_rank: int = LORA_RANK,
):
    """
    Загружает модель и токенизатор.

    Args:
        model_name:     Имя модели (HuggingFace Hub или локальный путь)
        max_seq_length: Максимальная длина последовательности
        load_in_4bit:   Использовать 4-bit квантизацию (экономит ~50% VRAM)
        with_lora:      Добавить LoRA адаптеры (для обучения)
        lora_rank:      Ранг LoRA адаптера

    Returns:
        (model, tokenizer)
    """
    gpu_info = check_gpu()
    logger.info(f"Загрузка модели: {model_name}")
    logger.info(f"  4-bit: {load_in_4bit} | max_seq: {max_seq_length} | LoRA: {with_lora}")

    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=None,  # авто-определение
        )
        logger.info("  ✓ Unsloth: FastLanguageModel загружена")
    except ImportError:
        logger.warning("Unsloth недоступен — использую HuggingFace Transformers (медленнее)")
        model, tokenizer = _load_with_transformers(model_name, load_in_4bit, max_seq_length)

    # Настройка pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if with_lora:
        model = _add_lora_adapters(model, lora_rank)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Параметры: {total_params/1e9:.2f}B всего | {trainable_params/1e6:.1f}M обучаемых")

    return model, tokenizer


def _add_lora_adapters(model, lora_rank: int = LORA_RANK):
    """Добавляет LoRA адаптеры к загруженной модели."""
    try:
        from unsloth import FastLanguageModel
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",  # >30% экономия памяти
            random_state=42,
            use_rslora=False,
        )
    except ImportError:
        from peft import get_peft_model, LoraConfig, TaskType
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    logger.info(f"  ✓ LoRA адаптеры добавлены (rank={lora_rank})")
    return model


def _load_with_transformers(model_name: str, load_in_4bit: bool, max_seq_length: int):
    """Fallback: загрузка через HuggingFace Transformers без Unsloth."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    bnb_config = None
    if load_in_4bit:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except Exception:
            logger.warning("bitsandbytes недоступен, загружаем без квантизации")

    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=max_seq_length)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Подготовка к инференсу
# ──────────────────────────────────────────────────────────────────────────────

def prepare_for_inference(model):
    """
    Переводит модель в режим инференса (быстрее, меньше памяти).
    Только для Unsloth.
    """
    try:
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
        logger.info("Модель переведена в режим инференса (Unsloth)")
    except (ImportError, AttributeError):
        model.eval()
        logger.info("Модель переведена в eval() режим")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Сохранение и загрузка чекпоинтов
# ──────────────────────────────────────────────────────────────────────────────

def save_model(model, tokenizer, save_path: str) -> None:
    """Сохраняет LoRA адаптеры и токенизатор."""
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Модель сохранена: {save_path}")


def push_to_hub(model, tokenizer, repo_id: str) -> None:
    """Публикует модель на HuggingFace Hub."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN не задан — публикация невозможна")
        return
    model.push_to_hub(repo_id, token=hf_token)
    tokenizer.push_to_hub(repo_id, token=hf_token)
    logger.info(f"Модель опубликована: https://huggingface.co/{repo_id}")
