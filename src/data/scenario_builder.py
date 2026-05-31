"""
scenario_builder.py
===================
Преобразование UEBA/CERT событий в текстовые сценарии для small LLM.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from src.data.preprocessor import build_ueba_prompt, build_ueba_response


RISK_LABELS = ("normal", "suspicious", "malicious")


@dataclass(frozen=True)
class ScenarioRecord:
    """Единая запись для обучения и оценки UEBA-моделей."""

    user_id: str
    date: str
    scenario: str
    risk_label: str
    evidence: list[str]
    source_events: list[dict[str, Any]]

    def to_example(self) -> dict[str, Any]:
        reasoning = make_rationale(self.risk_label, self.evidence)
        return {
            "user_id": self.user_id,
            "date": self.date,
            "scenario": self.scenario,
            "text": self.scenario,
            "prompt": build_ueba_prompt(self.scenario),
            "response": build_ueba_response(self.risk_label, self.evidence, reasoning),
            "label": self.risk_label,
            "risk_label": self.risk_label,
            "evidence": self.evidence,
            "reasoning": reasoning,
            "source_events": self.source_events,
        }


def build_scenario_record(
    user_id: str,
    date: str,
    events: list[dict[str, Any]],
    risk_label: str | None = None,
) -> ScenarioRecord:
    """Создает ScenarioRecord из событий одного пользователя за день/сессию."""
    normalized_events = [normalize_event(event) for event in events]
    features = extract_behavior_features(normalized_events)
    evidence = evidence_from_features(features)
    label = risk_label or weak_label_from_features(features)
    scenario = render_scenario(user_id, date, normalized_events, features, evidence)
    return ScenarioRecord(
        user_id=str(user_id),
        date=str(date),
        scenario=scenario,
        risk_label=label,
        evidence=evidence,
        source_events=normalized_events,
    )


def normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    """Приводит разнородные CERT CSV-строки к единой форме."""
    event_type = str(event.get("event_type") or event.get("type") or event.get("source") or "event")
    user_id = str(event.get("user_id") or event.get("user") or event.get("User") or "")
    timestamp = str(event.get("timestamp") or event.get("date") or event.get("Date") or "")
    pc = str(event.get("pc") or event.get("PC") or event.get("computer") or "")
    activity = str(event.get("activity") or event.get("Activity") or event.get("action") or "")
    normalized = dict(event)
    normalized.update(
        {
            "event_type": event_type.lower(),
            "user_id": user_id,
            "timestamp": timestamp,
            "date": timestamp[:10] if len(timestamp) >= 10 else timestamp,
            "pc": pc,
            "activity": activity.lower(),
        }
    )
    return normalized


def extract_behavior_features(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Считает компактные признаки поведения для baseline ML и сценариев."""
    event_list = list(events)
    by_type = Counter(event["event_type"] for event in event_list)
    hours = [_extract_hour(event.get("timestamp", "")) for event in event_list]
    after_hours = sum(1 for hour in hours if hour is not None and (hour < 7 or hour > 20))
    external_emails = sum(1 for event in event_list if _is_external_email(event))
    file_events = by_type.get("file", 0)
    usb_events = sum(1 for event in event_list if _is_usb_event(event))
    suspicious_urls = sum(1 for event in event_list if _is_suspicious_url(event))
    logon_failures = sum(1 for event in event_list if _is_logon_failure(event))
    total_events = len(event_list)
    return {
        "total_events": total_events,
        "event_type_counts": dict(by_type),
        "after_hours_events": after_hours,
        "external_emails": external_emails,
        "file_events": file_events,
        "usb_events": usb_events,
        "suspicious_urls": suspicious_urls,
        "logon_failures": logon_failures,
    }


def evidence_from_features(features: dict[str, Any]) -> list[str]:
    """Формирует evidence labels из признаков поведения."""
    evidence = []
    if features.get("after_hours_events", 0) >= 3:
        evidence.append("активность во внерабочее время")
    if features.get("usb_events", 0) > 0:
        evidence.append("использование съемного устройства")
    if features.get("file_events", 0) >= 25:
        evidence.append("массовые операции с файлами")
    if features.get("external_emails", 0) >= 3:
        evidence.append("несколько писем на внешние адреса")
    if features.get("suspicious_urls", 0) > 0:
        evidence.append("обращение к подозрительным web-ресурсам")
    if features.get("logon_failures", 0) >= 3:
        evidence.append("повторные неуспешные входы")
    return evidence


def weak_label_from_features(features: dict[str, Any]) -> str:
    """Weak-label fallback, если в CERT labels нет явной разметки."""
    evidence = evidence_from_features(features)
    high_risk = {
        "использование съемного устройства",
        "массовые операции с файлами",
        "несколько писем на внешние адреса",
        "обращение к подозрительным web-ресурсам",
    }
    high_risk_hits = sum(1 for item in evidence if item in high_risk)
    if high_risk_hits >= 3:
        return "malicious"
    if evidence:
        return "suspicious"
    return "normal"


def render_scenario(
    user_id: str,
    date: str,
    events: list[dict[str, Any]],
    features: dict[str, Any],
    evidence: list[str],
) -> str:
    """Рендерит короткий SOC-сценарий для LLM."""
    counts = features.get("event_type_counts", {})
    lines = [
        f"Пользователь: {user_id}",
        f"Период: {date}",
        (
            "Сводка событий: "
            f"всего {features.get('total_events', 0)}; "
            f"logon={counts.get('logon', 0)}, "
            f"device={counts.get('device', 0)}, "
            f"file={counts.get('file', 0)}, "
            f"email={counts.get('email', 0)}, "
            f"http={counts.get('http', 0)}."
        ),
    ]
    feature_line = (
        f"Внерабочее время: {features.get('after_hours_events', 0)}; "
        f"USB: {features.get('usb_events', 0)}; "
        f"файловые операции: {features.get('file_events', 0)}; "
        f"внешние письма: {features.get('external_emails', 0)}; "
        f"подозрительные URL: {features.get('suspicious_urls', 0)}; "
        f"ошибки входа: {features.get('logon_failures', 0)}."
    )
    lines.append(f"Поведенческие признаки: {feature_line}")
    if evidence:
        lines.append("Кандидатные признаки риска: " + "; ".join(evidence) + ".")
    lines.append("Последние события: " + " | ".join(summarize_events(events[-8:])))
    return "\n".join(lines)


def summarize_events(events: Iterable[dict[str, Any]]) -> list[str]:
    summaries = []
    for event in events:
        event_type = event.get("event_type", "event")
        timestamp = event.get("timestamp", "")
        activity = event.get("activity", "")
        target = (
            event.get("url")
            or event.get("to")
            or event.get("filename")
            or event.get("file")
            or event.get("pc")
            or ""
        )
        summary = " ".join(str(part) for part in [timestamp, event_type, activity, target] if part)
        summaries.append(summary[:180])
    return summaries or ["нет детальных событий"]


def make_rationale(label: str, evidence: list[str]) -> str:
    if label == "normal":
        return "Поведение не содержит выраженных признаков инсайдерской активности."
    if label == "suspicious":
        return "Найдены отдельные признаки риска, требующие проверки SOC-аналитиком."
    return "Набор признаков соответствует сценарию возможной подготовки или выполнения утечки данных."


def _extract_hour(timestamp: str) -> int | None:
    if not timestamp:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(timestamp[:19], fmt).hour
        except ValueError:
            continue
    match = __import__("re").search(r"\b([01]?\d|2[0-3]):[0-5]\d", timestamp)
    return int(match.group(1)) if match else None


def _is_usb_event(event: dict[str, Any]) -> bool:
    text = " ".join(str(value).lower() for value in event.values())
    return event.get("event_type") == "device" or "usb" in text or "removable" in text


def _is_external_email(event: dict[str, Any]) -> bool:
    if event.get("event_type") != "email":
        return False
    recipients = " ".join(str(event.get(key, "")) for key in ("to", "cc", "bcc"))
    return "@" in recipients and not any(domain in recipients.lower() for domain in ("dtaa.com", "company.com"))


def _is_suspicious_url(event: dict[str, Any]) -> bool:
    if event.get("event_type") != "http":
        return False
    url = str(event.get("url") or event.get("URL") or "").lower()
    markers = ("dropbox", "pastebin", "mega", "drive.google", "file", "upload", "share")
    return any(marker in url for marker in markers)


def _is_logon_failure(event: dict[str, Any]) -> bool:
    if event.get("event_type") != "logon":
        return False
    text = " ".join(str(value).lower() for value in event.values())
    return "fail" in text or "logoff" not in text and "denied" in text
