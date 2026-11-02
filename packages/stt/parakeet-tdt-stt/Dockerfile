# syntax=docker/dockerfile:1
# =============================================================================
# parakeet-tdt-stt 0.2.0 — NVIDIA Parakeet TDT 0.6B v3 (CC-BY-4.0)
#
# Token-and-Duration Transducer, 25 languages — the most accurate model in
# its class (LS-clean ~1.8% WER class). INT8 sherpa-onnx export (encoder
# 652 MB + decoder 11.8 + joiner 6.4 MB) on the GPU-capable sherpa wheel —
# ONE image serves provider=cpu AND provider=cuda (CUDA EP: sm_87 + sm_110a
# native SASS via our ORT 1.31.1). ~0.65-0.8 GB loaded. Offline per
# utterance — VAD-chunk long audio upstream (pattern in README).
# =============================================================================
# DECOUPLING CONTRACT: these two ARGs are the only external artifact
# references in this file; override to re-point either.
ARG BASE_IMAGE=nvidia/cuda:13.2.1-runtime-ubuntu24.04
ARG WHEELS_IMAGE=narandill/edge-wheels:cu132

FROM ubuntu:24.04 AS models
# Pinned k2-fsa asr-models release asset (int8, 25 languages).
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates bzip2 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /tmp/tdt && cd /tmp/tdt \
    && curl -fL --retry 5 -o tdt.tar.bz2 \
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2" \
    && echo "5793d0fd397c5778d2cf2126994d58e9d56b1be7c04d13c7a15bb1b4eafb16bf  tdt.tar.bz2" | sha256sum --check - \
    && tar -xjf tdt.tar.bz2 \
    && mkdir -p /models/parakeet-tdt-v3 \
    && for F in encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt; do \
        cp "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/$F" /models/parakeet-tdt-v3/; \
    done \
    && rm -rf /tmp/tdt

# DECOUPLING CONTRACT: these two ARGs are the only external artifact
# references in the runtime stage; override to re-point either.
FROM ${WHEELS_IMAGE} AS wheels
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="parakeet-tdt-stt" \
      org.opencontainers.image.description="NVIDIA Parakeet TDT 0.6B v3 int8 via sherpa-onnx 2.0.1 (GPU wheel: CUDA EP sm_87+sm_110a native + CPU); 25 languages, VAD-chunked offline ASR; JP 7.2.1" \
      org.opencontainers.image.source="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3"
# Weights CC-BY-4.0; sherpa-onnx Apache-2.0.

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
    && chmod +x /usr/local/bin/parakeet-tdt-transcribe

ENV PATH="/opt/venv/bin:${PATH}" \
    LD_LIBRARY_PATH=/opt/ort/lib \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TDT_ENCODER=/models/parakeet-tdt-v3/encoder.int8.onnx \
    TDT_DECODER=/models/parakeet-tdt-v3/decoder.int8.onnx \
    TDT_JOINER=/models/parakeet-tdt-v3/joiner.int8.onnx \
    TDT_VOCAB=/models/parakeet-tdt-v3/tokens.txt

CMD ["parakeet-tdt-transcribe", "--help"]
