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
    # torch pinned to 2.14.1: the 2.15.0+cu132 wheel's bundled oneDNN
    # v3.13.3 aborts on Neoverse V3AE (Xbyak_aarch64 "label is redefined")
    # during generation — 2.14.1's oneDNN is fine (verified on-device).
    # Re-test when NVIDIA ships a fixed aarch64 wheel (E9).
    && /opt/venv/bin/pip install --no-cache-dir pocket-tts==4.0.1 'torch==2.14.1'

# Bake the model: a real generation downloads + caches weights and voices
# under HF_HOME (offline OK afterwards). Default language bundle: english.
# The shipped english.yaml pins the HF revisions (model.safetensors@
# db04f0e6…, without-voice-cloning a7056f50…, tokenizer@00eac05e…) — a
# pocket-tts version bump ships a new yaml, which is the re-pin signal.
# `--voice alba` pins the voice to the english default (= what the smoke
# uses); a manifest records exactly what was baked into the layer.
ENV HF_HOME=/opt/hf-cache \
    XDG_CACHE_HOME=/opt/hf-cache/xdg
RUN /opt/venv/bin/pocket-tts generate --text "Warm up the Pocket TTS cache." \
        --output-path /tmp/warm.wav --language english --voice alba \
    && find /opt/hf-cache -type f -exec sha256sum {} + | sort -k2 > /opt/hf-cache/BAKE_MANIFEST.sha256 \
    && du -sh /opt/hf-cache && rm -f /tmp/warm.wav

ENV PATH="/opt/venv/bin:${PATH}"

EXPOSE 8000
CMD ["pocket-tts", "--help"]
