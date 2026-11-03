# =============================================================================
# canary-flash-stt 0.2.0 — NVIDIA Canary-180M-Flash (CC-BY-4.0)
#
# FastConformer encoder + 4-layer decoder (en/de/es/fr, ASR + X↔EN), ~182M
# params. INT8 sherpa-onnx export (encoder 132.7 MB + decoder 74.4 MB) on the
# GPU-capable sherpa wheel — ONE image serves provider=cpu AND provider=cuda
# (CUDA EP: sm_87 + sm_110a native SASS via our ORT 1.31.1). Per-utterance
# after VAD chunking: no native streaming. LibriSpeech-other 2.87 WER, SPGI 1.95.
# =============================================================================
# DECOUPLING CONTRACT: these two ARGs are the only external artifact
# references in this file; override to re-point either.
ARG BASE_IMAGE=nvidia/cuda:13.2.1-runtime-ubuntu24.04
ARG WHEELS_IMAGE=narandill/edge-wheels:cu132

FROM ubuntu:24.04 AS models
# Pinned k2-fsa asr-models release asset (int8, en-es-de-fr).
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates bzip2 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /tmp/canary && cd /tmp/canary \
    && curl -fL --retry 5 -o canary.tar.bz2 \
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2" \
    && echo "7a38ed8b13f014ad632b09ff8d22e0c6f1359dd046af9235d281dfae841b9ab9  canary.tar.bz2" | sha256sum --check - \
    && tar -xjf canary.tar.bz2 \
    && mkdir -p /models/canary-flash \
    && for F in encoder.int8.onnx decoder.int8.onnx tokens.txt; do \
        cp "sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/$F" /models/canary-flash/; \
    done \
    && rm -rf /tmp/canary

# DECOUPLING CONTRACT: these two ARGs are the only external artifact
# references in the runtime stage; override to re-point either.
FROM ${WHEELS_IMAGE} AS wheels
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="canary-flash-stt" \
      org.opencontainers.image.description="NVIDIA Canary-180M-Flash int8 via sherpa-onnx 2.0.1 (GPU wheel: CUDA EP sm_87+sm_110a native + CPU); VAD-chunked offline ASR + X↔EN translation; JP 7.2.1" \
      org.opencontainers.image.source="https://huggingface.co/soniqo/Canary-180M-Flash-ONNX"
# Weights CC-BY-4.0 (upstream nvidia/canary-180m-flash); sherpa-onnx Apache-2.0.

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
    # ONE image serves provider=cpu and provider=cuda. v2.0.1 fixes canary
    # lang2id derivation. cuBLAS ships with the CUDA runtime base; cuDNN 9
    # via apt; the ORT .so files land on LD_LIBRARY_PATH.
    && /opt/venv/bin/pip install --no-cache-dir /tmp/sherpa/sherpa_onnx-*.whl soundfile \
    && mkdir -p /opt/ort/lib \
    && cp -a /tmp/sherpa/libonnxruntime*.so* /opt/ort/lib/ \
    && rm -rf /tmp/sherpa \
    && chmod +x /usr/local/bin/canary-transcribe /usr/local/bin/voice-server

ENV PATH="/opt/venv/bin:${PATH}" \
    LD_LIBRARY_PATH=/opt/ort/lib \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CANARY_ENCODER=/models/canary-flash/encoder.int8.onnx \
    CANARY_DECODER=/models/canary-flash/decoder.int8.onnx \
    CANARY_VOCAB=/models/canary-flash/tokens.txt \
    CANARY_SRC_LANG=en \
    CANARY_TGT_LANG=en

# voice-api v1 (docs/voice-api.md); the transcribe CLI stays available
EXPOSE 8080
CMD ["voice-server"]
