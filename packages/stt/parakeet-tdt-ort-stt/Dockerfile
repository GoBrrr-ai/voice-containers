# syntax=docker/dockerfile:1
# =============================================================================
# parakeet-tdt-ort-stt 0.1.0 — NVIDIA parakeet-tdt-0.6b-v3 int4 ONNX (CC-BY-4.0)
#
# The Orin-budget ORT row for the same 0.6B v3 TDT our sherpa int8
# (parakeet-tdt-stt) and parakeet.cpp q4_k (parakeet-tdt-cpp-stt) rows serve:
#   encoder-model.int4.onnx   391 MB — MatMulNBits int4 weights, CUDA EP
#                             (block_size=64, K/N % 64 == 0 — CUDA-eligible)
#   decoder_joint-model.int8  18 MB  — DYNAMIC int8 QOperator -> CPU EP only
#                             (DynamicQuantizeLSTM/MatMulInteger have no CUDA kernels)
#   nemo128.onnx              138 KB — the 128-bin mel featurizer graph
#   vocab.txt                 8193 tokens, <blk>=8192 last
# Total ~409 MB weights (efederici: "6x smaller than fp32, half the quant
# degradation of int8"). Decode loop = NeMo GreedyTDTInfer port
# (task/research/parakeet-int4-ort-driver/, end-to-end verified vs the
# jfk.flac reference); mixed-EP by design: encoder on CUDA, decoder on CPU.
# =============================================================================
# DECOUPLING CONTRACT: these two ARGs are the only external artifact
# references in this file; override to re-point either.
ARG BASE_IMAGE=nvidia/cuda:13.2.1-runtime-ubuntu24.04
ARG WHEELS_IMAGE=narandill/edge-wheels:cu132

FROM ubuntu:24.04 AS models
# Pinned: efederici/parakeet-tdt-0.6b-v3-onnx-int4 @ 9828da0021d81df726b3d5d
# b2d6d04354942749 (2026-04-17). nemo128.onnx is the fp32 featurizer (the
# int8 twin exists but fp32 keeps the conv state carry exact).
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /models/parakeet-int4 && cd /models/parakeet-int4 \
    && for F in encoder-model.int4.onnx decoder_joint-model.int8.onnx \
                nemo128.onnx vocab.txt config.json; do \
        curl -fsSL --retry 5 -O \
            "https://huggingface.co/efederici/parakeet-tdt-0.6b-v3-onnx-int4/resolve/9828da0021d81df726b3d5ddb2d6d04354942749/$F" \
            || exit 1; \
    done \
    && test -s encoder-model.int4.onnx && test -s vocab.txt

FROM ${WHEELS_IMAGE} AS wheels
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="parakeet-tdt-ort-stt" \
      org.opencontainers.image.description="NVIDIA parakeet-tdt-0.6b-v3 int4 MatMulNBits (391 MB) on our onnxruntime-gpu 1.31.1 CUDA EP (sm_87+sm_110a); decoder_joint dynamic-int8 on CPU; TDT greedy decode" \
      org.opencontainers.image.source="https://huggingface.co/efederici/parakeet-tdt-0.6b-v3-onnx-int4"
# Weights CC-BY-4.0 (nvidia/parakeet-tdt-0.6b-v3); export Apache-2.0 (efederici).

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv libsndfile1 libcudnn9-cuda-13 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=models /models /models
COPY --from=wheels /wheels/onnxruntime_gpu-*.whl /tmp/
COPY rootfs/ /

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir "numpy<3" soundfile \
    && /opt/venv/bin/pip install --no-cache-dir /tmp/onnxruntime_gpu-*.whl \
    && rm -rf /tmp/onnxruntime_gpu-*.whl \
    && chmod +x /usr/local/bin/tdt-ort-transcribe /usr/local/bin/voice-server

ENV PATH="/opt/venv/bin:${PATH}" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TDT_ORT_MODEL_DIR=/models/parakeet-int4 \
    TDT_ORT_PROVIDER=cuda

# voice-api v1 (docs/voice-api.md); the transcribe CLI stays available
EXPOSE 8080
CMD ["voice-server"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4).status == 200 else 1)" || exit 1
