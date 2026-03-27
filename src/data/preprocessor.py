"""Prompt formatting and output parsing for behavior classification."""

from __future__ import annotations

import re
from typing import Optional

_LABEL_PATTERN = re.compile(r"\b(positive|negative)\b", re.IGNORECASE)
_REASONING_PATTERN = re.compile(
    r"(?:анализ|analysis|обоснование)\s*:\s*(.+)",
    flags=re.IGNORECASE | re.DOTALL,
)


def build_chat_prompt(text: str, tokenizer=None) -> str:
    """Build a consistent instruction prompt used for inference/training."""
    instruction = (
        "Проанализируй текст и выведи метку sentiment: positive или negative. "
        "Формат ответа: сначала 'Метка: <label>', затем 'Анализ: <краткое объяснение>'."
    )
    user_text = text.strip()

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": "Ты точный классификатор поведенческих паттернов."},
            {"role": "user", "content": f"{instruction}\n\nТекст:\n{user_text}"},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    return (
        "[SYSTEM] Ты точный классификатор поведенческих паттернов.\n"
        f"[USER] {instruction}\n\nТекст:\n{user_text}\n"
        "[ASSISTANT]"
    )


def parse_label(response_text: Optional[str]) -> Optional[str]:
    """Extract class label from free-form model response."""
    if not response_text:
        return None

    text = response_text.strip().lower()

    # Strong patterns first.
    for marker in ("метка:", "label:", "sentiment:"):
        if marker in text:
            tail = text.split(marker, 1)[1]
            match = _LABEL_PATTERN.search(tail)
            if match:
                return match.group(1).lower()

    # Fallback anywhere in the response.
    match = _LABEL_PATTERN.search(text)
    if match:
        return match.group(1).lower()

    return None


def parse_reasoning(response_text: Optional[str]) -> Optional[str]:
    """Extract reasoning section if present."""
    if not response_text:
        return None

    match = _REASONING_PATTERN.search(response_text)
    if match:
        reasoning = match.group(1).strip()
        return reasoning or None

    return None
