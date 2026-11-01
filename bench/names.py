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
    "moonshine-streaming-stt": "moonshine",
    "canary-flash-stt": "canary",
    "parakeet-tdt-stt": "parakeet-tdt",
    "nemotron-streaming-stt": "nemotron-streaming",
    "pocket-tts": "pocket-tts",
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
    "moonshine": "#56B4E9",  # sky blue
    "canary": "#8a7f2f",  # dark yellow (F0E442 is taken by paradee)
    "parakeet-tdt": "#D55E00",  # vermillion
    "nemotron-streaming": "#0072B2",  # blue
    "pocket-tts": "#9467bd",  # purple
}
FAMILY_COLORS = {**DEFAULT_FAMILY_COLORS, **BRAND_COLORS}

# Color = identity of the MODEL (the chart label), consistent across every
# chart — the parakeet.cpp runtime carries several distinct models, so the
# family hue would collapse them. The same model in two variants keeps ONE
# hue: variant by marker shape (circle CPU / square GPU) + label.
MODEL_COLORS = {
    # STT
    "nemotron-unified 0.6B q8_0": "#76B900",  # NVIDIA green (brand)
    "phonon-2.5 q8_0": "#56B4E9",  # Okabe-Ito sky blue
    "parakeet-redux 178M q8_0": "#D55E00",  # Okabe-Ito vermillion
    "redux packed (CPU)": "#D55E00",  # same model as above — CPU variant
    "faster-whisper tiny int8": "#E69F00",  # Okabe-Ito orange
    # TTS
    "kokoro 82M int8": "#009E73",  # Okabe-Ito green
    "paradee 8M int8 (GPU)": "#F0E442",  # Okabe-Ito yellow
    "paradee 8M int8 (CPU)": "#F0E442",  # same model — backend variant
    "piper1-tts libritts-r medium": "#41BDF5",  # Home Assistant blue (brand)
    "phonon-31m (CPU)": "#CC79A7",  # Okabe-Ito pink
    "pocket-tts 100M": "#9467bd",  # purple
    # new STT
    "moonshine-streaming medium int8": "#56B4E9",  # sky blue
    "canary-180m-flash int8": "#8a7f2f",  # dark yellow
    "parakeet-tdt 0.6B v3 int8": "#D55E00",  # vermillion
    "nemotron-streaming 0.6B int8 (160 ms)": "#0072B2",  # blue
}


def color(row: dict) -> str:
    """Identity color for a benchmark row: per model, fallback to the family
    hue for configs not in MODEL_COLORS."""
    return MODEL_COLORS.get(display(row)) or FAMILY_COLORS.get(family(row), "#0072B2")


EMOJI = {
    "parakeet.cpp": "🦜",
    "faster-whisper": "⚡",
    "kokoro": "🗣️",
    "piper1-tts": "🎺",
    "phonon-31m": "🌊",
    "paradee": "✨",
    "moonshine": "🌙",
    "canary": "🐤",
    "parakeet-tdt": "🦜",
    "nemotron-streaming": "🔊",
    "pocket-tts": "🎒",
}

# Packages registry for the root README "Packages" tables (bench/readme.py
# fills in image name + version from each package's build.sh, so versions can
# never drift). Key = directory/image name under packages/<task>/; table rows
# sort by (family, name). Golden rule: register every new package here.
PACKAGES = {
    "wyoming-faster-whisper": {
        "what": "faster-whisper 1.2.1 on our CTranslate2 4.8.4 (sm_87+sm_110)",
        "streaming": "wyoming streaming",
    },
    "nemotron-unified-asr": {
        "what": "nvidia nemotron-unified-asr 0.6B (q8_0) on parakeet.cpp",
        "streaming": "**true buffered** (0.16–2.08 s)",
    },
    "parakeet-redux-stt": {
        "what": "moondream parakeet-redux 178M (q8_0) GPU + packed CPU",
        "streaming": "VAD-chunked (opt-in)",
    },
    "phonon2-stt": {
        "what": "fermion phonon-2.5 (q8_0) GPU + packed CPU",
        "streaming": "VAD-chunked (opt-in)",
    },
    "wyoming-piper1-tts": {
        "what": "piper-tts 1.8.0 on our onnxruntime-gpu 1.31.1",
        "streaming": "wyoming streaming",
    },
    "kokoro-tts": {
        "what": "Kokoro-82M v1.0 int8 ONNX on onnxruntime-gpu (sm_87+sm_110a)",
        "streaming": "`create_stream()`",
    },
    "phonon31m-tts": {
        "what": "gradium/phonon-31m-en via `ptts` — **CPU-only by design** (307 MB)",
        "streaming": "**true streaming** (TTFA ~76 ms p95)",
    },
    "paradee-tts": {
        "what": "Paradee-8M v1.2 int8 ONNX — a Kokoro distillation (4 voices)",
        "streaming": "sentence-chunked",
    },
    "moonshine-streaming-stt": {
        "what": "Moonshine Streaming Medium 245M via moonshine-voice (official ORT runtime)",
        "streaming": "**native incremental** (live partials)",
    },
    "canary-flash-stt": {
        "what": "NVIDIA Canary-180M-Flash int8 (en/de/es/fr + X→EN) via sherpa-onnx",
        "streaming": "VAD-chunked (opt-in)",
    },
    "parakeet-tdt-stt": {
        "what": "NVIDIA Parakeet TDT 0.6B v3 int8 (25 languages) via sherpa-onnx",
        "streaming": "VAD-chunked (opt-in)",
    },
    "nemotron-streaming-stt": {
        "what": "NVIDIA Nemotron Speech Streaming EN 0.6B int8 (160 ms lookahead) via sherpa-onnx",
        "streaming": "**true native** (cache-aware, graph-chunked)",
    },
    "pocket-tts": {
        "what": "Kyutai Pocket TTS 100M (official pkg) — streaming + voice cloning, 7 languages",
        "streaming": "**true streaming** (~200 ms TTFA)",
    },
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
        return "phonon-31m (CPU)"
    if proj == "paradee-tts":
        return "paradee 8M int8 (GPU)" if backend == "cuda" else "paradee 8M int8 (CPU)"
    if proj == "wyoming-piper1-tts":
        return "piper1-tts libritts-r medium"
    if proj == "moonshine-streaming-stt":
        return "moonshine-streaming medium int8"
    if proj == "canary-flash-stt":
        return "canary-180m-flash int8"
    if proj == "parakeet-tdt-stt":
        return "parakeet-tdt 0.6B v3 int8"
    if proj == "nemotron-streaming-stt":
        return "nemotron-streaming 0.6B int8 (160 ms)"
    if proj == "pocket-tts":
        return "pocket-tts 100M"
    base = _model_base(row) or proj
    return f"{proj} · {base}"
