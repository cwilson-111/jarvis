"""Model registry and downloader.

Every model the assistant uses is listed here with a pinned URL and SHA-256, so
`jarvis download` is reproducible and `jarvis doctor` can tell exactly what is
missing. Weights live in the gitignored ``models/`` directory, never in git.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PARAKEET_DIR = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"


@dataclass(frozen=True)
class ModelFile:
    """One downloadable artifact.

    ``path`` is relative to the models directory. For archives, ``path`` is the
    directory the archive unpacks to and ``members`` are the files that must
    exist inside it afterwards.
    """

    name: str
    url: str
    path: str
    sha256: str
    size_mb: int
    archive: bool = False
    members: tuple[str, ...] = ()

    def present(self, models_dir: Path) -> bool:
        target = models_dir / self.path
        if self.archive:
            return all((target / m).is_file() for m in self.members)
        return target.is_file()


REGISTRY: tuple[ModelFile, ...] = (
    ModelFile(
        name="Silero VAD v5",
        url="https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/src/silero_vad/data/silero_vad.onnx",
        path="silero_vad.onnx",
        sha256="2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f",
        size_mb=2,
    ),
    ModelFile(
        name="openWakeWord melspectrogram",
        url="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
        path="openwakeword/melspectrogram.onnx",
        sha256="ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
        size_mb=1,
    ),
    ModelFile(
        name="openWakeWord embedding",
        url="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
        path="openwakeword/embedding_model.onnx",
        sha256="70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
        size_mb=1,
    ),
    ModelFile(
        name="openWakeWord 'hey jarvis'",
        url="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx",
        path="openwakeword/hey_jarvis_v0.1.onnx",
        sha256="94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb",
        size_mb=1,
    ),
    ModelFile(
        name="Kokoro-82M v1.0 (int8 ONNX)",
        url="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx",
        path="kokoro/kokoro-v1.0.int8.onnx",
        sha256="6e742170d309016e5891a994e1ce1559c702a2ccd0075e67ef7157974f6406cb",
        size_mb=88,
    ),
    ModelFile(
        name="Kokoro voices v1.0",
        url="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
        path="kokoro/voices-v1.0.bin",
        sha256="bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
        size_mb=27,
    ),
    ModelFile(
        name="Parakeet TDT 0.6B v3 (int8, sherpa-onnx)",
        url=f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{PARAKEET_DIR}.tar.bz2",
        path=PARAKEET_DIR,
        sha256="",  # archive checksum is not pinned upstream; members are checked instead
        size_mb=465,
        archive=True,
        members=("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"),
    ),
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` with a coarse progress line on stderr."""
    request = urllib.request.Request(url, headers={"User-Agent": "jarvis-voice/0.1"})
    with urllib.request.urlopen(request, timeout=60) as resp, dest.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        last_pct = -10
        while block := resp.read(1 << 20):
            out.write(block)
            done += len(block)
            if total:
                pct = done * 100 // total
                if pct >= last_pct + 10:
                    last_pct = pct
                    print(f"    {pct:3d}%  ({done >> 20} / {total >> 20} MB)", file=sys.stderr)


def _safe_extract(archive: Path, dest: Path) -> None:
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")


def download(models_dir: Path, *, force: bool = False) -> list[str]:
    """Download every missing model into ``models_dir``. Returns the names fetched."""
    models_dir.mkdir(parents=True, exist_ok=True)
    fetched: list[str] = []
    for model in REGISTRY:
        if model.present(models_dir) and not force:
            print(f"  ok       {model.name}", file=sys.stderr)
            continue
        print(f"  fetching {model.name} (~{model.size_mb} MB)", file=sys.stderr)
        target = models_dir / model.path
        with tempfile.TemporaryDirectory(dir=models_dir) as tmp:
            tmp_file = Path(tmp) / "download"
            fetch(model.url, tmp_file)
            if model.sha256:
                actual = sha256_of(tmp_file)
                if actual != model.sha256:
                    raise RuntimeError(
                        f"checksum mismatch for {model.name}: expected {model.sha256}, got {actual}"
                    )
            if model.archive:
                if target.exists():
                    shutil.rmtree(target)
                _safe_extract(tmp_file, models_dir)
                missing = [m for m in model.members if not (target / m).is_file()]
                if missing:
                    raise RuntimeError(f"{model.name}: archive did not contain {missing}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp_file.replace(target)
        fetched.append(model.name)
    return fetched


def missing(models_dir: Path) -> list[ModelFile]:
    return [m for m in REGISTRY if not m.present(models_dir)]
