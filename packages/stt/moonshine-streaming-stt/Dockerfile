# syntax=docker/dockerfile:1
# =============================================================================
# moonshine-streaming-stt 0.1.0 — Moonshine Streaming Medium 245M (MIT)
#
# Native incremental streaming ASR (English) via the OFFICIAL moonshine-voice
# runtime: aarch64 manylinux wheel on PyPI (bundles the ORT C++ core), model
# = .ort flatbuffer 5-graph bundle (encoder/adapter/cross_kv/decoder_kv/
# frontend) from download.moonshine.ai, quantized medium-streaming-en.
# Open-ASR-avg WER 6.65% (float reference). CPU-only by design — no CUDA
# needed, runs on A78AE (Orin) and Thor alike.
#
# ~270 MB of model files + ~50-300 MB runtime. Ships `moonshine-transcribe`
# (offline per file) — for live partials use the streaming API
# (Transcriber.create_stream / MicTranscriber, see README).
# =============================================================================
FROM ubuntu:24.04 AS models
# Pinned stage dir: the quantized medium-streaming-en bundle (arch id 5),
# digests recorded 2026-11-01. spelling_cnn.ort deliberately absent — it
# lives under model/spelling-en/ upstream (404s here) and the helper does
# not use alphanumeric fusion. Every fetch is fatal + digest-checked.
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /models/moonshine-streaming-en && cd /models/moonshine-streaming-en \
    && for F in adapter.ort cross_kv.ort decoder_kv.ort encoder.ort \
                frontend.model.ort frontend.weights.ort \
                streaming_config.json tokenizer.bin; do \
        curl -fsSL --retry 5 -O "https://download.moonshine.ai/model/medium-streaming-en/quantized_26_08_21/$F" || exit 1; \
    done \
    && printf '%s  %s\n' \
        3f2a287def57cc094367a0eec3c4f5fc36a32ec420e86764b696920991b20281 adapter.ort \
        642f6e21cd305be79342207c6f9e6b681d469d55bc48c72b27b84846fb71fd1e cross_kv.ort \
        193bb366492b74fc4ad338c6778e8d8eb916aaa11b5aa264f9057f4db7759486 decoder_kv.ort \
        12915e76ebac7dd287c5ea63965d06103a53ba1ce242a4a34f318f3958c60c37 encoder.ort \
        95768855c70c8251eeecc05fedf69999da1b8ab16f605c9f457fd3354b0ad6b5 frontend.model.ort \
        5ac941f490cbe035b335b99a414cc393d62d4c6f9f2423495b286870d271d709 frontend.weights.ort \
        28e83b7a28e91472692a035e0dae3116422ae43aeb2bef5ed822c44ce89b88af streaming_config.json \
        6884b35fd6377d4c4d32336a0bc152f36b64d1e45b6503683cdc238250a8472d tokenizer.bin \
        > /tmp/moonshine.sha256 \
    && sha256sum --check --strict /tmp/moonshine.sha256 \
    && rm /tmp/moonshine.sha256

FROM ubuntu:24.04
LABEL org.opencontainers.image.title="moonshine-streaming-stt" \
      org.opencontainers.image.description="Moonshine Streaming Medium 245M via moonshine-voice 0.1.5; native incremental streaming, CPU-only; JP 7.2.1" \
      org.opencontainers.image.source="https://github.com/moonshine-ai/moonshine"
# MIT (runtime + weights). Ubuntu 24.04 glibc 2.39 satisfies the manylinux_2_34 wheel.

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=models /models /models
COPY rootfs/ /

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    # Pinned: 0.1.5 ships aarch64 manylinux_2_31/2_34 wheels (ORT core inside).
    && /opt/venv/bin/pip install --no-cache-dir moonshine-voice==0.1.5 soundfile soxr \
    && chmod +x /usr/local/bin/moonshine-transcribe

ENV PATH="/opt/venv/bin:${PATH}" \
    MOONSHINE_MODEL_DIR=/models/moonshine-streaming-en \
    MOONSHINE_MODEL_ARCH=5

CMD ["moonshine-transcribe", "--help"]
