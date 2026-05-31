"""Prompt formatting and output parsing for behavior and UEBA classification."""

from __future__ import annotations

import re
from typing import Optional


LABEL_MAP = {
    "positive": "positive",
    "negative": "negative",
    "pos": "positive",
    "neg": "negative",
    "1": "positive",
    "0": "negative",
    "1.0": "positive",
    "0.0": "negative",
    "normal": "normal",
    "норма": "normal",
    "нормальный": "normal",
    "штатный": "normal",
    "suspicious": "suspicious",
    "susp": "suspicious",
    "подозрительный": "suspicious",
    "сомнительный": "suspicious",
    "malicious": "malicious",
    "mal": "malicious",
    "вредоносный": "malicious",
    "инцидент": "malicious",
    "атака": "malicious",
}

VALID_LABELS = {"positive", "negative"}
UEBA_LABELS = {"normal", "suspicious", "malicious"}

_LEGACY_LABEL_PATTERN = re.compile(r"\b(positive|negative|pos|neg)\b", re.IGNORECASE)
_UEBA_LABEL_PATTERN = re.compile(
    r"\b(normal|suspicious|malicious|норма|подозрительный|вредоносный)\b",
    re.IGNORECASE,
)
_REASONING_PATTERN = re.compile(
    r"(?:анализ|analysis|обоснование|rationale)\s*:\s*(.+)",
    flags=re.IGNORECASE | re.DOTALL,
)

SYSTEM_PROMPT = "Ты точный классификатор поведенческих паттернов."
UEBA_SYSTEM_PROMPT = (
    "Ты — SOC-аналитик, выполняющий UEBA-анализ пользовательского поведения. "
    "Классифицируй риск поведения по логам и кратко укажи признаки решения."
)


def format_example(row: dict) -> dict:
    """Convert raw row to prompt/response structure for training."""
    text = str(row.get("text", row.get("scenario", ""))).strip()
    label = normalize_label(row.get("label", row.get("risk_label", "")))
    reasoning = str(row.get("reasoning", row.get("rationale", ""))).strip()
    evidence = row.get("evidence", row.get("evidence_labels", []))

    if not label:
        raise ValueError(f"Unknown label: {row.get('label', row.get('risk_label'))!r}")

    if label in UEBA_LABELS:
        prompt = build_ueba_prompt(text)
        response = build_ueba_response(label, evidence, reasoning)
    else:
        prompt = build_prompt(text)
        response = build_response(label, reasoning)

    return {
        "prompt": prompt,
        "response": response,
        "label": label,
        "risk_label": label if label in UEBA_LABELS else "",
        "text": text,
        "scenario": text,
        "reasoning": reasoning,
        "evidence": evidence,
    }


def build_prompt(text: str) -> str:
    """Build legacy positive/negative instruction prompt."""
    return (
        "Проанализируй текст и выведи метку sentiment: positive или negative.\n"
        "Формат ответа: сначала 'Метка: <label>', затем 'Анализ: <краткое объяснение>'.\n\n"
        f"Текст:\n{text.strip()}"
    )


def build_response(label: str, reasoning: str = "") -> str:
    """Build legacy target answer."""
    parts = [f"Метка: {label}"]
    if reasoning:
        parts.append(f"Анализ: {reasoning}")
    return "\n".join(parts)


def build_ueba_prompt(scenario: str) -> str:
    """Build UEBA/insider-threat instruction prompt."""
    return (
        "Задача: анализ поведения пользователя в информационной системе\n"
        f"Сценарий:\n{scenario.strip()}\n\n"
        "Определи уровень риска поведения строго одним из классов: normal | suspicious | malicious.\n"
        "Дай ответ строго в формате:\n"
        "Риск: <normal|suspicious|malicious>\n"
        "Признаки: <2-4 кратких признака через ';'>\n"
        "Обоснование: <краткое объяснение>"
    )


def build_ueba_response(label: str, evidence=None, reasoning: str = "") -> str:
    """Build UEBA target answer."""
    evidence = evidence or []
    if isinstance(evidence, str):
        evidence_text = evidence
    else:
        evidence_text = "; ".join(str(item) for item in evidence if str(item).strip())
    parts = [f"Риск: {label}", f"Признаки: {evidence_text or 'нет выраженных признаков риска'}"]
    if reasoning:
        parts.append(f"Обоснование: {reasoning}")
    return "\n".join(parts)


def build_chat_prompt(text: str, tokenizer=None, task: str = "sentiment") -> str:
    """Build chat-template prompt when tokenizer supports it; otherwise fallback text."""
    user_prompt = build_ueba_prompt(text) if task == "ueba" else build_prompt(text)
    system_prompt = UEBA_SYSTEM_PROMPT if task == "ueba" else SYSTEM_PROMPT

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    return f"[SYSTEM] {system_prompt}\n[USER] {user_prompt}\n[ASSISTANT]"


def parse_label(response_text: Optional[str]) -> Optional[str]:
    """Extract positive/negative or UEBA risk label from free-form model response."""
    if not response_text:
        return None

    text = response_text.strip().lower()

    for marker in ("риск:", "risk:"):
        if marker in text:
            tail = text.split(marker, 1)[1]
            match = _UEBA_LABEL_PATTERN.search(tail)
            if match:
                return normalize_label(match.group(1))

    for marker in ("метка:", "label:", "sentiment:", "категория:"):
        if marker in text:
            tail = text.split(marker, 1)[1]
            match = _LEGACY_LABEL_PATTERN.search(tail)
            if match:
                return normalize_label(match.group(1))

    match = _UEBA_LABEL_PATTERN.search(text)
    if match:
        return normalize_label(match.group(1))
    match = _LEGACY_LABEL_PATTERN.search(text)
    if match:
        return normalize_label(match.group(1))
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


def parse_evidence(response_text: Optional[str]) -> list[str]:
    """Extract UEBA evidence items from 'Признаки:' section."""
    if not response_text:
        return []
    match = re.search(
        r"(?:признаки|evidence)\s*:\s*(.+?)(?:\n\s*(?:обоснование|анализ|rationale)\s*:|$)",
        response_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return []
    raw = match.group(1).strip()
    parts = re.split(r"[;\n,]+", raw)
    return [part.strip(" -\t") for part in parts if part.strip(" -\t")]


def normalize_label(label) -> Optional[str]:
    """Normalize known labels to canonical task labels."""
    return LABEL_MAP.get(str(label).strip().lower())
