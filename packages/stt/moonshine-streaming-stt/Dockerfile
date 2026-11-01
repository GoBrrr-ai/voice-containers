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
# Pinned stage dir: the quantized medium-streaming-en bundle (arch id 5).
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /models/moonshine-streaming-en && cd /models/moonshine-streaming-en \
    && for F in adapter.ort cross_kv.ort decoder_kv.ort encoder.ort \
                frontend.model.ort frontend.weights.ort spelling_cnn.ort \
                streaming_config.json tokenizer.bin; do \
        curl -fL --retry 5 -O "https://download.moonshine.ai/model/medium-streaming-en/quantized_26_08_21/$F"; \
    done

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
    && /opt/venv/bin/pip install --no-cache-dir moonshine-voice==0.1.5 soundfile \
    && chmod +x /usr/local/bin/moonshine-transcribe

ENV PATH="/opt/venv/bin:${PATH}" \
    MOONSHINE_MODEL_DIR=/models/moonshine-streaming-en \
    MOONSHINE_MODEL_ARCH=5

CMD ["moonshine-transcribe", "--help"]
