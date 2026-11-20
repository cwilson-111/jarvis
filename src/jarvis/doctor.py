"""`jarvis doctor`: is everything this machine needs actually in place?"""

from __future__ import annotations

import platform
from collections.abc import Callable
from dataclasses import dataclass

import httpx
import numpy as np

from jarvis.config import Config
from jarvis.models import REGISTRY


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    critical: bool = True


def _python() -> Check:
    machine = platform.machine()
    ok = not (platform.system() == "Darwin" and machine == "x86_64" and _apple_silicon())
    detail = f"{platform.python_version()} ({machine})"
    if not ok:
        detail += ": x86_64 Python under Rosetta on Apple Silicon; use `uv sync` (arm64)"
    return Check("python", ok, detail)


def _apple_silicon() -> bool:
    import subprocess

    out = subprocess.run(
        ["sysctl", "-n", "hw.optional.arm64"], capture_output=True, text=True, check=False
    )
    return out.stdout.strip() == "1"


def _models(cfg: Config) -> list[Check]:
    checks = []
    for model in REGISTRY:
        present = model.present(cfg.models_dir)
        detail = str(cfg.models_dir / model.path) if present else "missing: run `jarvis download`"
        checks.append(Check(f"model: {model.name}", present, detail))
    return checks


def _ollama(cfg: Config) -> list[Check]:
    try:
        with httpx.Client(base_url=cfg.llm.host, timeout=3.0) as client:
            version = client.get("/api/version").json().get("version", "?")
            tags = client.get("/api/tags").json().get("models", [])
    except (httpx.HTTPError, ValueError) as exc:
        return [Check("ollama", False, f"not reachable at {cfg.llm.host}: {exc}")]
    names = {str(m.get("name", "")) for m in tags}
    wanted = cfg.llm.model
    has_model = wanted in names or f"{wanted}:latest" in names
    return [
        Check("ollama", True, f"v{version} at {cfg.llm.host}"),
        Check(
            f"ollama model: {wanted}",
            has_model,
            "pulled" if has_model else f"missing: run `ollama pull {wanted}`",
        ),
    ]


def _audio() -> list[Check]:
    try:
        import sounddevice as sd
    except OSError as exc:  # PortAudio missing
        return [Check("audio", False, f"PortAudio unavailable: {exc}")]
    checks = []
    try:
        dev_in = sd.query_devices(kind="input")
        recording = sd.rec(int(0.5 * 16000), samplerate=16000, channels=1, dtype="float32")
        sd.wait()
        rms = float(np.sqrt(np.mean(np.square(recording))))
        # macOS hands back exact zeros when the app lacks microphone permission.
        ok = rms > 0.0
        detail = f"{dev_in['name']}: 0.5 s captured, RMS {rms:.5f}"
        if not ok:
            detail += (
                " (all zeros: grant microphone access to your terminal in System Settings ->"
                " Privacy & Security -> Microphone)"
            )
        checks.append(Check("microphone", ok, detail))
    except Exception as exc:
        checks.append(Check("microphone", False, f"{type(exc).__name__}: {exc}"))
    try:
        dev_out = sd.query_devices(kind="output")
        checks.append(Check("speaker", True, str(dev_out["name"])))
    except Exception as exc:
        checks.append(Check("speaker", False, f"{type(exc).__name__}: {exc}"))
    return checks


def _tts_runtime() -> Check:
    try:
        import espeakng_loader

        return Check("espeak-ng (Kokoro phonemizer)", True, str(espeakng_loader.get_library_path()))
    except Exception as exc:
        return Check("espeak-ng (Kokoro phonemizer)", False, f"{type(exc).__name__}: {exc}")


def run_checks(cfg: Config, *, audio: bool = True) -> list[Check]:
    steps: list[Callable[[], Check | list[Check]]] = [
        _python,
        lambda: _models(cfg),
        _tts_runtime,
        lambda: _ollama(cfg),
    ]
    if audio:
        steps.append(_audio)
    checks: list[Check] = []
    for step in steps:
        result = step()
        checks.extend(result if isinstance(result, list) else [result])
    return checks
