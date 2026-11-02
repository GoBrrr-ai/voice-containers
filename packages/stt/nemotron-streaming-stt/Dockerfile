# syntax=docker/dockerfile:1
# =============================================================================
# nemotron-streaming-stt 0.2.0 — NVIDIA Nemotron Speech Streaming EN 0.6B
#
# Cache-aware streaming FastConformer RNNT (English), the best streaming ASR
# design in its class. INT8 sherpa-onnx export, 160 ms lookahead latency
# profile BAKED into the graph — sherpa's stream API does the internal
# chunking; never pre-chunk the audio yourself (wrong boundaries silently
# drop audio). Encoder 653 MB int8.
# NVIDIA Open Model License. GPU via the sherpa-onnx GPU wheel (CUDA EP:
# sm_87 + sm_110a native SASS via our ORT 1.31.1) — ONE image serves
# provider=cpu AND provider=cuda.
#
# NOTE: we also ship nemotron-unified-asr (parakeet.cpp GGUF) — this is the
# STREAMING variant via sherpa-onnx, a different model + runtime.
# =============================================================================
# DECOUPLING CONTRACT: these two ARGs are the only external artifact
# references in this file; override to re-point either.
ARG BASE_IMAGE=nvidia/cuda:13.2.1-runtime-ubuntu24.04
ARG WHEELS_IMAGE=narandill/edge-wheels:cu132

FROM ubuntu:24.04 AS models
# Pinned k2-fsa asr-models release asset: 160 ms latency profile, int8.
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates bzip2 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /tmp/nemostream && cd /tmp/nemostream \
    && curl -fL --retry 5 -o nemostream.tar.bz2 \
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25.tar.bz2" \
    && echo "0ae73a41cd51599dc7cac9ac083d9d35de53d762ca45923505fde47a3751814b  nemostream.tar.bz2" | sha256sum --check - \
    && tar -xjf nemostream.tar.bz2 \
    && mkdir -p /models/nemotron-streaming-160ms \
    && for F in encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt; do \
        cp "sherpa-onnx-nemotron-speech-streaming-en-0.6b-160ms-int8-2026-04-25/$F" /models/nemotron-streaming-160ms/; \
    done \
    && test -f /models/nemotron-streaming-160ms/encoder.int8.onnx \
    && test -f /models/nemotron-streaming-160ms/tokens.txt \
    && rm -rf /tmp/nemostream

# DECOUPLING CONTRACT: these two ARGs are the only external artifact
# references in the runtime stage; override to re-point either.
FROM ${WHEELS_IMAGE} AS wheels
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="nemotron-streaming-stt" \
      org.opencontainers.image.description="NVIDIA Nemotron Speech Streaming EN 0.6B int8 (160 ms lookahead) via sherpa-onnx 2.0.1 (GPU wheel: CUDA EP sm_87+sm_110a native + CPU); native cache-aware streaming; JP 7.2.1" \
      org.opencontainers.image.source="https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b"
# NVIDIA Open Model License; sherpa-onnx Apache-2.0.

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv libcudnn9-cuda-13 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=models /models /models
COPY --from=wheels /wheels/sherpa_onnx-*.whl /wheels/ort-libs/ /tmp/sherpa/
COPY rootfs/ /

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    # GPU-capable sherpa-onnx wheel from builders/wheels-cu132 — linked
    # against OUR CUDA-13 ORT (sm_87+sm_110a SASS in providers_cuda.so), so
    # ONE image serves provider=cpu and provider=cuda. cuBLAS ships with the
    # CUDA runtime base; cuDNN 9 via apt; ORT .so files on LD_LIBRARY_PATH.
    && /opt/venv/bin/pip install --no-cache-dir /tmp/sherpa/sherpa_onnx-*.whl soundfile \
    && mkdir -p /opt/ort/lib \
    && cp -a /tmp/sherpa/libonnxruntime*.so* /opt/ort/lib/ \
    && rm -rf /tmp/sherpa \
    && chmod +x /usr/local/bin/nemotron-stream-transcribe

ENV PATH="/opt/venv/bin:${PATH}" \
    LD_LIBRARY_PATH=/opt/ort/lib \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NEMOSTREAM_ENCODER=/models/nemotron-streaming-160ms/encoder.int8.onnx \
    NEMOSTREAM_DECODER=/models/nemotron-streaming-160ms/decoder.int8.onnx \
    NEMOSTREAM_JOINER=/models/nemotron-streaming-160ms/joiner.int8.onnx \
    NEMOSTREAM_VOCAB=/models/nemotron-streaming-160ms/tokens.txt

CMD ["nemotron-stream-transcribe", "--help"]
