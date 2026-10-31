"""Human-readable names for engines, projects and families.

Ground truth is the recorded Docker image (``narandill/<project>:<tag>``) and
the model path — never the cryptic config ids (``parakeet_cpp_0``…). Tables
show the original project + model names; charts get a short ``display`` label.
"""

from __future__ import annotations

# project (from image) -> engine family shown in the README cards
FAMILY_OF_PROJECT = {
    "nemotron-unified-asr": "parakeet.cpp",
    "parakeet-redux-stt": "parakeet.cpp",
    "phonon2-stt": "parakeet.cpp",
    "wyoming-faster-whisper": "faster-whisper",
    "kokoro-tts": "kokoro",
    "wyoming-piper1-tts": "piper1-tts",
    "phonon31m-tts": "phonon-31m",
    "paradee-tts": "paradee",
}

# Color = identity, per engine FAMILY, consistent across every chart.
# Real brand colors where they exist; the rest stay on the Okabe-Ito palette.
BRAND_COLORS = {
    "parakeet.cpp": "#76B900",  # NVIDIA green — the runtime carries NVIDIA parakeet models
    "piper1-tts": "#41BDF5",  # Home Assistant blue — Piper is HA's official TTS
}
# Okabe-Ito identity hues for families without a brand color
DEFAULT_FAMILY_COLORS = {
    "faster-whisper": "#E69F00",  # orange
    "kokoro": "#009E73",  # green
    "phonon-31m": "#CC79A7",  # pink
    "paradee": "#F0E442",  # yellow
}
FAMILY_COLORS = {**DEFAULT_FAMILY_COLORS, **BRAND_COLORS}

EMOJI = {
    "parakeet.cpp": "🦜",
    "faster-whisper": "⚡",
    "kokoro": "🗣️",
    "piper1-tts": "🎺",
    "phonon-31m": "🌊",
    "paradee": "✨",
}

# short identity description per family (cards show no benchmark numbers)
FAMILY_DESC = {
    "parakeet.cpp": "one parakeet.cpp runtime, several q8_0 GGUF models",
    "faster-whisper": "faster-whisper 1.2.1 on our CTranslate2 4.8.4 build",
    "kokoro": "Kokoro-82M v1.0, int8 ONNX on onnxruntime-gpu",
    "piper1-tts": "piper-tts 1.8.0 on our onnxruntime-gpu build",
    "phonon-31m": "gradium/phonon-31m-en via the ptts runtime — CPU-only by design",
    "paradee": "Paradee-8M v1.0 int8 ONNX — a Kokoro distillation",
}

# streaming behavior (from the project registry; shown in the family cards)
STREAMING = {
    "parakeet.cpp": "true buffered streaming (0.16–2.08 s chunks)",
    "faster-whisper": "Wyoming streaming",
    "kokoro": "create_stream()",
    "piper1-tts": "Wyoming streaming",
    "phonon-31m": "true streaming on CPU (307 MB)",
    "paradee": "sentence-chunked",
}


def project(row: dict) -> str:
    """Original project name, from the recorded image."""
    img = row.get("image") or ""
    name = img.split("/")[-1].split(":")[0]
    return name or row.get("engine", "?")


def family(row: dict) -> str:
    return FAMILY_OF_PROJECT.get(project(row), project(row))


def star(row: dict) -> str:
    """Star marker for the cohort winner (rank 1) — used in charts (font-safe)."""
    return "★ " if row.get("rank") == 1 else ""


MEDALS = {1: "🥇\u00a0", 2: "🥈\u00a0", 3: "🥉\u00a0"}


def medal(row: dict) -> str:
    """Medal emoji for the composite top 3 — used in tables (GitHub renders them)."""
    return MEDALS.get(row.get("rank"), "")


def _model_base(row: dict) -> str:
    m = str(row.get("model") or "")
    base = m.rsplit("/", 1)[-1]
    return base.removesuffix(".gguf").removesuffix(".onnx")


def model_pretty(row: dict) -> str:
    """Original model name for tables (extension stripped, baked models named)."""
    proj = project(row)
    base = _model_base(row)
    if proj == "paradee-tts":
        return "Paradee-8M-v1.0 int8 (baked)"
    if proj == "kokoro-tts":
        return "Kokoro-82M v1.0 int8 (model_uint8.onnx)"
    if proj == "parakeet-redux-stt" and "packed" in base:
        return "parakeet-redux packed (redux-packed.gguf)"
    return base or "—"


def display(row: dict) -> str:
    """Short chart label: original engine + model variant."""
    proj = project(row)
    model = str(row.get("model") or "").lower()
    backend = (row.get("requested_backend") or "").lower()
    if proj == "nemotron-unified-asr":
        return "nemotron-unified 0.6B q8_0"
    if proj == "parakeet-redux-stt":
        return "redux packed (CPU)" if "packed" in model else "parakeet-redux 178M q8_0"
    if proj == "phonon2-stt":
        return "phonon-2.5 q8_0"
    if proj == "wyoming-faster-whisper":
        return "faster-whisper tiny int8"
    if proj == "kokoro-tts":
        return "kokoro 82M int8"
    if proj == "phonon31m-tts":
        return "phonon-31m (CPU streaming)"
    if proj == "paradee-tts":
        return "paradee 8M int8 (GPU)" if backend == "cuda" else "paradee 8M int8 (CPU)"
    if proj == "wyoming-piper1-tts":
        return "piper1-tts libritts-r medium"
    base = _model_base(row) or proj
    return f"{proj} · {base}"
