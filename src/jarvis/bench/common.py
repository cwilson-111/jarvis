"""Shared benchmark plumbing: hardware facts, result files, memory sampling."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import httpx
import psutil

RESULTS_DIR = Path("docs/results")


def _sysctl(name: str) -> str:
    out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=False)
    return out.stdout.strip()


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
    }
    if sys.platform == "darwin":
        info["chip"] = _sysctl("machdep.cpu.brand_string")
        info["model"] = _sysctl("hw.model")
        info["macos"] = platform.mac_ver()[0]
    packages = ("sherpa-onnx", "kokoro-onnx", "openwakeword", "onnxruntime", "faster-whisper")
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    info["packages"] = versions
    return info


def ollama_info(host: str, model: str) -> dict[str, Any]:
    try:
        with httpx.Client(base_url=host, timeout=5.0) as client:
            version = client.get("/api/version").json().get("version")
            tags = client.get("/api/tags").json().get("models", [])
            running = client.get("/api/ps").json().get("models", [])
    except (httpx.HTTPError, ValueError):
        return {"reachable": False}
    wanted = {model, f"{model}:latest"}
    tag: dict[str, Any] = next((t for t in tags if t.get("name") in wanted), {})
    ps: dict[str, Any] = next((r for r in running if r.get("name") in wanted), {})
    return {
        "reachable": True,
        "version": version,
        "model": tag.get("name"),
        "digest": tag.get("digest"),
        "quantization": tag.get("details", {}).get("quantization_level"),
        "parameter_size": tag.get("details", {}).get("parameter_size"),
        "loaded_size_mb": round(ps["size"] / 2**20) if ps.get("size") else None,
        "loaded_vram_mb": round(ps["size_vram"] / 2**20) if ps.get("size_vram") else None,
    }


def save_result(name: str, data: dict[str, Any], directory: Path = RESULTS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    data = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"), **data}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_result(name: str, directory: Path = RESULTS_DIR) -> dict[str, Any] | None:
    path = directory / f"{name}.json"
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _ollama_processes() -> list[psutil.Process]:
    """Ollama's server plus its model runner (``llama-server`` in recent Ollama builds)."""
    procs = []
    for proc in psutil.process_iter(["name"]):
        name = (proc.info.get("name") or "").lower()
        if "ollama" in name or "llama-server" in name or "llama_server" in name:
            procs.append(proc)
    return procs


@dataclass
class MemorySampler:
    """Samples RSS of this process and of all Ollama processes every ``interval`` s.

    On macOS, Metal buffers holding the LLM weights are not part of the Ollama
    runner's RSS, so the model's resident size is taken from Ollama's own
    ``/api/ps`` report as well (see ``ollama_info``).
    """

    interval: float = 0.25
    peak_self_mb: float = 0.0
    peak_ollama_mb: float = 0.0
    peak_total_mb: float = 0.0
    min_available_mb: float = float("inf")
    swap_start_mb: float = 0.0
    swap_peak_mb: float = 0.0
    samples: int = 0
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def __enter__(self) -> MemorySampler:
        self.swap_start_mb = psutil.swap_memory().used / 2**20
        self._thread = threading.Thread(target=self._loop, name="mem", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def _loop(self) -> None:
        me = psutil.Process()
        ollama = _ollama_processes()
        last_refresh = time.monotonic()
        while not self._stop.is_set():
            if time.monotonic() - last_refresh > 5.0:
                ollama, last_refresh = _ollama_processes(), time.monotonic()
            own = me.memory_info().rss / 2**20
            other = 0.0
            for proc in ollama:
                try:
                    other += proc.memory_info().rss / 2**20
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            vm = psutil.virtual_memory()
            self.peak_self_mb = max(self.peak_self_mb, own)
            self.peak_ollama_mb = max(self.peak_ollama_mb, other)
            self.peak_total_mb = max(self.peak_total_mb, own + other)
            self.min_available_mb = min(self.min_available_mb, vm.available / 2**20)
            self.swap_peak_mb = max(self.swap_peak_mb, psutil.swap_memory().used / 2**20)
            self.samples += 1
            self._stop.wait(self.interval)

    def summary(self) -> dict[str, Any]:
        return {
            "peak_jarvis_rss_mb": round(self.peak_self_mb),
            "peak_ollama_rss_mb": round(self.peak_ollama_mb),
            "peak_combined_rss_mb": round(self.peak_total_mb),
            "min_system_available_mb": round(self.min_available_mb),
            "swap_used_start_mb": round(self.swap_start_mb),
            "swap_used_peak_mb": round(self.swap_peak_mb),
            "samples": self.samples,
            "interval_s": self.interval,
        }
