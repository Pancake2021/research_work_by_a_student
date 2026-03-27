#!/usr/bin/env python3
"""HTTP client for colab-mcp server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests


class ColabMCPClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def health(self) -> dict[str, Any]:
        return self._get_json("/health")

    def execute_cell(self, code: str, timeout: int = 3600, wait: bool = False) -> dict[str, Any]:
        payload = {"code": code, "timeout": timeout, "wait": wait}
        return self._post_json("/execute", payload)

    def upload_file(self, local_path: str, colab_path: str) -> dict[str, Any]:
        path = Path(local_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(local_path)

        with path.open("rb") as f:
            files = {"file": (path.name, f)}
            data = {"colab_path": colab_path}
            resp = self.session.post(self._url("/upload"), files=files, data=data, timeout=self.timeout)
        return self._raise_or_json(resp)

    def download_file(self, colab_path: str, local_path: str) -> dict[str, Any]:
        payload = {"colab_path": colab_path}
        resp = self.session.post(self._url("/download"), json=payload, timeout=self.timeout, stream=True)
        if resp.status_code >= 400:
            body = resp.text.strip()
            raise RuntimeError(f"HTTP {resp.status_code}: {body}")

        out_path = Path(local_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=2 * 1024 * 1024):
                if chunk:
                    f.write(chunk)

        return {"status": "ok", "local_path": str(out_path), "size_bytes": out_path.stat().st_size}

    def get_runtime_info(self) -> dict[str, Any]:
        return self._get_json("/runtime_info")

    def stream_logs(self, last_n_lines: int = 100) -> dict[str, Any]:
        params = {"last_n_lines": int(last_n_lines)}
        resp = self.session.get(self._url("/stream_logs"), params=params, timeout=self.timeout)
        return self._raise_or_json(resp)

    def interrupt_execution(self) -> dict[str, Any]:
        return self._post_json("/interrupt", {})

    def _get_json(self, path: str) -> dict[str, Any]:
        resp = self.session.get(self._url(path), timeout=self.timeout)
        return self._raise_or_json(resp)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self.session.post(self._url(path), json=payload, timeout=self.timeout)
        return self._raise_or_json(resp)

    @staticmethod
    def _raise_or_json(resp: requests.Response) -> dict[str, Any]:
        if resp.status_code >= 400:
            body = resp.text.strip()
            raise RuntimeError(f"HTTP {resp.status_code}: {body}")
        return resp.json()


def parse_json_lines(lines: list[str]) -> list[dict[str, Any]]:
    """Extract strict JSON lines from logs."""
    events: list[dict[str, Any]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Strip stream prefixes like [stdout] ...
        if stripped.startswith("[stdout] "):
            stripped = stripped[len("[stdout] "):]
        elif stripped.startswith("[stderr] "):
            stripped = stripped[len("[stderr] "):]

        if not (stripped.startswith("{") and stripped.endswith("}")):
            continue

        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events
