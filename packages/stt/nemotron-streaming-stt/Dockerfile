# syntax=docker/dockerfile:1
# =============================================================================
# nemotron-streaming-stt 0.1.0 — NVIDIA Nemotron Speech Streaming EN 0.6B
#
# Cache-aware streaming FastConformer RNNT (English), the best streaming ASR
# design in its class. INT8 sherpa-onnx export, 160 ms lookahead latency
# profile BAKED into the graph — sherpa's stream API does the internal
# chunking; never pre-chunk the audio yourself (wrong boundaries silently
# drop audio). Encoder 653 MB int8.
# NVIDIA Open Model License. CPU-only (no aarch64 CUDA-13 sherpa build).
#
# NOTE: we also ship nemotron-unified-asr (parakeet.cpp GGUF) — this is the
# STREAMING variant via sherpa-onnx, a different model + runtime.
# =============================================================================
FROM ubuntu:24.04 AS models
# Pinned k2-fsa asr-models release asset: 160 ms latency profile, int8.
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates bzip2 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /tmp/nemostream && cd /tmp/nemostream \
    && curl -fL --retry 5 -o nemostream.tar.bz2 \
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25.tar.bz2" \
    && tar -xjf nemostream.tar.bz2 \
    && mkdir -p /models/nemotron-streaming-160ms \
    && find . -name "encoder.int8.onnx" -exec cp {} /models/nemotron-streaming-160ms/ \; \
    && find . -name "decoder.int8.onnx" -exec cp {} /models/nemotron-streaming-160ms/ \; \
    && find . -name "joiner.int8.onnx" -exec cp {} /models/nemotron-streaming-160ms/ \; \
    && find . -name "tokens.txt" -exec cp {} /models/nemotron-streaming-160ms/ \; \
    && rm -rf /tmp/nemostream

FROM ubuntu:24.04
LABEL org.opencontainers.image.title="nemotron-streaming-stt" \
      org.opencontainers.image.description="NVIDIA Nemotron Speech Streaming EN 0.6B int8 (160 ms lookahead) via sherpa-onnx 2.0.1; native cache-aware streaming, CPU; JP 7.2.1" \
      org.opencontainers.image.source="https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b"
# NVIDIA Open Model License; sherpa-onnx Apache-2.0.

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=models /models /models
COPY rootfs/ /

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    # >= 1.12.22 required for this model; 2.0.1 is current.
    && /opt/venv/bin/pip install --no-cache-dir sherpa-onnx==2.0.1 soundfile \
    && chmod +x /usr/local/bin/nemotron-stream-transcribe

ENV PATH="/opt/venv/bin:${PATH}" \
    NEMOSTREAM_ENCODER=/models/nemotron-streaming-160ms/encoder.int8.onnx \
    NEMOSTREAM_DECODER=/models/nemotron-streaming-160ms/decoder.int8.onnx \
    NEMOSTREAM_JOINER=/models/nemotron-streaming-160ms/joiner.int8.onnx \
    NEMOSTREAM_VOCAB=/models/nemotron-streaming-160ms/tokens.txt

CMD ["nemotron-stream-transcribe", "--help"]
