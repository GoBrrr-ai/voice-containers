# syntax=docker/dockerfile:1
# =============================================================================
# pocket-tts 0.1.0 — Kyutai Pocket TTS 100M (CC-BY-4.0)
#
# The OFFICIAL kyutai `pocket-tts` package (pure-python wheel + torch CPU):
# native streaming with adaptive chunks (~200 ms TTFA), zero-shot voice
# cloning, 7 languages (en/fr/de/pt/it/es/nl), infinitely long inputs.
# Also available: tiny English models (english_18m/7m/2m_2026-10, down to
# 1.9M params at 25x RT) via --language — see README.
#
# Model weights download at build time (warm-run) and live in /opt/hf-cache;
# the image runs fully offline afterwards. CPU-only — torch CPU aarch64.
# =============================================================================
FROM ubuntu:24.04
LABEL org.opencontainers.image.title="pocket-tts" \
      org.opencontainers.image.description="Kyutai Pocket TTS 100M via the official pocket-tts CLI (streaming, voice cloning) on torch-cpu; JP 7.2.1" \
      org.opencontainers.image.source="https://github.com/kyutai-labs/pocket-tts"
# Weights CC-BY-4.0 (kyutai/pocket-tts); code MIT.

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    # PyPI aarch64: pure-python wheel; torch is pulled as a CPU wheel.
    && /opt/venv/bin/pip install --no-cache-dir pocket-tts==4.0.1

# oneDNN's aarch64 JIT (Xbyak_aarch64) aborts on the Thor CPU with
# "label is redefined" (torch 2.15.0) — disable the oneDNN path globally.
# Re-test with a newer torch before removing (E9). NOTE: must go to the
# STDLIB dir — a venv site-packages sitecustomize.py is shadowed by
# Ubuntu's apport hook in /usr/lib/python3.12.
RUN printf '%s\n' \
    '# apport hook (Ubuntu default) + oneDNN aarch64 JIT workaround' \
    'try:' \
    '    import apport_python_hook' \
    'except ImportError:' \
    '    pass' \
    'else:' \
    '    apport_python_hook.install()' \
    '' \
    'try:' \
    '    import torch' \
    '    torch.backends.mkldnn.enabled = False' \
    'except Exception:' \
    '    pass' \
    > /usr/lib/python3.12/sitecustomize.py

# Bake the models: a real generation downloads + caches weights and voices
# under HF_HOME (offline OK afterwards). Two language bundles:
#   english             — the 100M flagship (RTF 0.13 GPU / 0.39 CPU on Thor)
#   english_18m_2026-10 — the Orin-budget find: 111.7 MB fp32 safetensors,
#     "up to 25× realtime on one CPU thread" (kyutai 2026-10-20); 7m/2m
#     exist but 18m is the quality/speed sweet spot for the 0.8 GB budget.
# The shipped english.yaml pins the HF revisions (model.safetensors@
# db04f0e6…, without-voice-cloning a7056f50…, tokenizer@00eac05e…) — a
# pocket-tts version bump ships a new yaml, which is the re-pin signal.
# `--voice alba` pins the voice to the english default (= what the smoke
# uses); a manifest records exactly what was baked into the layer.
ENV HF_HOME=/opt/hf-cache \
    XDG_CACHE_HOME=/opt/hf-cache/xdg
RUN /opt/venv/bin/pocket-tts generate --text "Warm up the Pocket TTS cache." \
        --output-path /tmp/warm.wav --language english --voice alba \
    && /opt/venv/bin/pocket-tts generate --text "Warm up the eighteen million cache." \
        --output-path /tmp/warm18.wav --language english_18m_2026-10 --voice alba \
    && find /opt/hf-cache -type f -exec sha256sum {} + | sort -k2 > /opt/hf-cache/BAKE_MANIFEST.sha256 \
    && du -sh /opt/hf-cache && rm -f /tmp/warm.wav /tmp/warm18.wav

# GPU-capable: the torch cu132 wheel ships sm_110 SASS (Thor) and sm_80
# cubins (native on Orin sm_87 via within-major 8.0→8.7 binary compat —
# no JIT). Driver injection on an ubuntu base needs both vars (nvidia/cuda
# bases bake them in); without them libcuda never appears and torch fails
# with "Found no NVIDIA driver". `pocket-tts --device cuda` / .to("cuda").
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV PATH="/opt/venv/bin:${PATH}"

EXPOSE 8000
CMD ["pocket-tts", "--help"]
