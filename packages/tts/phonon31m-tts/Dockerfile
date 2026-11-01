# syntax=docker/dockerfile:1
# =============================================================================
# phonon31m-tts 0.1.0 — gradium/phonon-31m-en (Apache-2.0, 31M params)
#
# Pure-CPU image (plain ubuntu:24.04, NO CUDA) running the `ptts` 0.6.0
# aarch64 manylinux wheel (Rust "xn" engine). True streaming via
# tts.stream() — TTFA p95 28.4 ms (Open TTS LB #1 on CPU), RTFx 16.26.
#
# Baked: model.q8.gguf (64.5 MB) + default-voice.safetensors + tokenizer files.
# RAM ~0.10-0.15 GB. Ships ptts (CLI) and tts-stream-server (chunked HTTP).
# =============================================================================
FROM ubuntu:24.04 AS models
# Pinned: the upstream default voice silently changed to "Reuben" on
# 2026-10-22 without a version bump — resolve/92741c5b freezes it.
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /models/phonon-31m-en && cd /models/phonon-31m-en \
    && for F in model.q8.gguf default-voice.safetensors config.json generation_config.json \
                chat_template.jinja tokenizer.json tokenizer.model tokenizer.5.0.0.json \
                tokenizer_config.json processor_config.json; do \
        curl -fL --retry 5 -O "https://huggingface.co/gradium/phonon-31m-en/resolve/92741c5b07d085ff840e26fea8ce43110b87852f/$F"; \
    done

FROM ubuntu:24.04
LABEL org.opencontainers.image.title="phonon31m-tts" \
      org.opencontainers.image.description="gradium/phonon-31m-en (31M, EN) via ptts 0.6.0; CPU-only, true streaming, TTFA ~28ms" \
      org.opencontainers.image.source="https://github.com/gradium-ai/xn-ptts"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=models /models /models
COPY rootfs/ /

# A78AE (Orin) and Thor both run this; the wheel is manylinux-generic.
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir ptts==0.6.0 \
    && chmod +x /usr/local/bin/tts-stream-server /usr/local/bin/ptts-say

ENV PATH="/opt/venv/bin:${PATH}" \
    PTTS_MODEL=/models/phonon-31m-en \
    PTTS_LANG=en \
    PTTS_QUANT=q8 \
    TTS_PORT=8880

EXPOSE 8880
CMD ["tts-stream-server"]
