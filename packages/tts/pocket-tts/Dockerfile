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

# Bake the model: a real generation downloads + caches weights and voices
# under HF_HOME (offline OK afterwards). Default language bundle: english.
ENV HF_HOME=/opt/hf-cache \
    XDG_CACHE_HOME=/opt/hf-cache/xdg
RUN /opt/venv/bin/pocket-tts generate --text "Warm up the Pocket TTS cache." \
        --output /tmp/warm.wav --language english \
    && du -sh /opt/hf-cache && rm -f /tmp/warm.wav

ENV PATH="/opt/venv/bin:${PATH}"

EXPOSE 8000
CMD ["pocket-tts", "--help"]
