#!/usr/bin/env python3
"""Minimal MCP-like HTTP bridge for running commands in Google Colab.

Run this inside Colab (first cell) and expose via ngrok.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request, send_file


APP = Flask(__name__)
API_KEY = os.getenv("COLAB_MCP_API_KEY", "")
LOG_BUFFER_MAX = int(os.getenv("COLAB_MCP_LOG_BUFFER", "2000"))

_STATE_LOCK = threading.Lock()
_LOG_BUFFER = deque(maxlen=LOG_BUFFER_MAX)
_CURRENT_PROCESS: Optional[subprocess.Popen] = None
_CURRENT_EXEC_ID: Optional[str] = None
_CURRENT_STATUS: str = "idle"
_LAST_RESULT: dict = {}


def _require_auth() -> Optional[tuple]:
    if not API_KEY:
        return None
    token = request.headers.get("X-API-Key", "")
    if token != API_KEY:
        return jsonify({"status": "error", "error": "unauthorized"}), 401
    return None


def _append_log(line: str) -> None:
    _LOG_BUFFER.append({
        "ts": time.time(),
        "line": line.rstrip("\n"),
    })


def _runtime_info() -> dict:
    gpu_name = None
    gpu_ram_gb = None
    accelerator = "cpu"

    try:
        import torch

        if torch.cuda.is_available():
            accelerator = "gpu"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_ram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
    except Exception:
        pass

    if os.getenv("COLAB_TPU_ADDR"):
        accelerator = "tpu"

    # Colab VM reports (best-effort)
    ram_gb = None
    disk_gb = None
    try:
        import psutil

        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        disk_gb = round(psutil.disk_usage("/").total / (1024 ** 3), 2)
    except Exception:
        pass

    return {
        "status": "ok",
        "accelerator": accelerator,
        "gpu_name": gpu_name,
        "gpu_ram_gb": gpu_ram_gb,
        "ram_gb": ram_gb,
        "disk_gb": disk_gb,
        "execution_status": _CURRENT_STATUS,
        "execution_id": _CURRENT_EXEC_ID,
    }


def _reader_thread(stream, label: str) -> None:
    for line in iter(stream.readline, ""):
        _append_log(f"[{label}] {line}")


def _run_command(code: str, timeout: int, execution_id: str) -> None:
    global _CURRENT_PROCESS, _CURRENT_EXEC_ID, _CURRENT_STATUS, _LAST_RESULT

    cmd = ["bash", "-lc", code]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid if os.name != "nt" else None,
    )

    with _STATE_LOCK:
        _CURRENT_PROCESS = proc
        _CURRENT_EXEC_ID = execution_id
        _CURRENT_STATUS = "running"

    out_thread = threading.Thread(target=_reader_thread, args=(proc.stdout, "stdout"), daemon=True)
    err_thread = threading.Thread(target=_reader_thread, args=(proc.stderr, "stderr"), daemon=True)
    out_thread.start()
    err_thread.start()

    try:
        return_code = proc.wait(timeout=timeout)
        out_thread.join(timeout=1)
        err_thread.join(timeout=1)
        status = "ok" if return_code == 0 else "error"
        result = {
            "status": status,
            "execution_id": execution_id,
            "return_code": return_code,
        }
    except subprocess.TimeoutExpired:
        _interrupt_current_process("timeout")
        result = {
            "status": "error",
            "execution_id": execution_id,
            "return_code": None,
            "error": f"timeout_after_{timeout}s",
        }

    with _STATE_LOCK:
        _CURRENT_PROCESS = None
        _CURRENT_STATUS = "idle"
        _LAST_RESULT = result


def _run_command_sync(code: str, timeout: int) -> dict:
    proc = subprocess.run(
        ["bash", "-lc", code],
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return {
        "status": "ok" if proc.returncode == 0 else "error",
        "return_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _interrupt_current_process(reason: str = "manual") -> bool:
    global _CURRENT_PROCESS, _CURRENT_STATUS
    with _STATE_LOCK:
        proc = _CURRENT_PROCESS
    if not proc:
        return False

    try:
        if os.name != "nt":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        _CURRENT_STATUS = f"interrupted:{reason}"
        return True
    except Exception:
        return False


@APP.route("/health", methods=["GET"])
def health():
    auth = _require_auth()
    if auth:
        return auth
    return jsonify({"status": "ok", "service": "colab-mcp"})


@APP.route("/runtime_info", methods=["GET"])
def runtime_info():
    auth = _require_auth()
    if auth:
        return auth
    return jsonify(_runtime_info())


@APP.route("/execute", methods=["POST"])
def execute():
    auth = _require_auth()
    if auth:
        return auth

    body = request.get_json(force=True, silent=False) or {}
    code = body.get("code", "")
    timeout = int(body.get("timeout", 3600))
    wait = bool(body.get("wait", False))

    if not code.strip():
        return jsonify({"status": "error", "error": "empty_code"}), 400

    if wait:
        try:
            result = _run_command_sync(code, timeout)
            if result.get("stdout"):
                for line in result["stdout"].splitlines():
                    _append_log(f"[stdout] {line}")
            if result.get("stderr"):
                for line in result["stderr"].splitlines():
                    _append_log(f"[stderr] {line}")
            return jsonify(result)
        except subprocess.TimeoutExpired:
            return jsonify({
                "status": "error",
                "return_code": None,
                "stdout": "",
                "stderr": "",
                "error": f"timeout_after_{timeout}s",
            }), 408

    with _STATE_LOCK:
        if _CURRENT_STATUS == "running":
            return jsonify({"status": "error", "error": "execution_in_progress", "execution_id": _CURRENT_EXEC_ID}), 409

    execution_id = f"exec_{uuid.uuid4().hex[:10]}"
    _append_log(f"[system] starting execution {execution_id}")

    worker = threading.Thread(target=_run_command, args=(code, timeout, execution_id), daemon=True)
    worker.start()

    return jsonify({"status": "started", "execution_id": execution_id})


@APP.route("/stream_logs", methods=["GET"])
def stream_logs():
    auth = _require_auth()
    if auth:
        return auth

    last_n_lines = int(request.args.get("last_n_lines", 100))
    if last_n_lines < 1:
        last_n_lines = 1
    lines = list(_LOG_BUFFER)[-last_n_lines:]

    return jsonify({
        "status": "ok",
        "execution_status": _CURRENT_STATUS,
        "execution_id": _CURRENT_EXEC_ID,
        "lines": [entry["line"] for entry in lines],
        "last_result": _LAST_RESULT,
    })


@APP.route("/interrupt", methods=["POST"])
def interrupt():
    auth = _require_auth()
    if auth:
        return auth

    ok = _interrupt_current_process("manual")
    return jsonify({"status": "ok" if ok else "idle"})


@APP.route("/upload", methods=["POST"])
def upload_file():
    auth = _require_auth()
    if auth:
        return auth

    dst = request.form.get("colab_path", "")
    file = request.files.get("file")
    if not dst or file is None:
        return jsonify({"status": "error", "error": "missing_file_or_path"}), 400

    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    file.save(str(dst_path))
    return jsonify({"status": "ok", "colab_path": str(dst_path), "size_bytes": dst_path.stat().st_size})


@APP.route("/download", methods=["POST"])
def download_file():
    auth = _require_auth()
    if auth:
        return auth

    body = request.get_json(force=True, silent=False) or {}
    src = body.get("colab_path", "")
    if not src:
        return jsonify({"status": "error", "error": "missing_colab_path"}), 400

    src_path = Path(src)
    if not src_path.exists() or not src_path.is_file():
        return jsonify({"status": "error", "error": "file_not_found"}), 404

    return send_file(str(src_path), as_attachment=True, download_name=src_path.name)


def maybe_start_ngrok(port: int) -> None:
    authtoken = os.getenv("NGROK_AUTHTOKEN", "").strip()
    if not authtoken:
        print("NGROK_AUTHTOKEN not set; run local tunnel manually if needed.")
        return

    try:
        from pyngrok import ngrok

        ngrok.set_auth_token(authtoken)
        public_url = ngrok.connect(port)
        print(f"COLAB_MCP_URL={public_url}")
    except Exception as exc:
        print(f"Failed to start ngrok: {exc}")


def main() -> None:
    host = os.getenv("COLAB_MCP_HOST", "0.0.0.0")
    port = int(os.getenv("COLAB_MCP_PORT", "5000"))
    if os.getenv("COLAB_MCP_ENABLE_NGROK", "1") == "1":
        maybe_start_ngrok(port)
    APP.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
