# syntax=docker/dockerfile:1
# =============================================================================
# moonshine-streaming-stt 0.2.0 — Moonshine Streaming Medium 245M (MIT)
#
# TWO runtimes, one image:
#  - CPU: official moonshine-voice 0.1.5 — its .ort 5-graph bundle is fused
#    to com.microsoft CPU-only ops (CPU-only by design, upstream
#    docs/execution-providers.md).
#  - GPU: Masterx 5-graph plain-ai.onnx FP32 export driven directly on our
#    onnxruntime-gpu 1.31.1 CUDA EP (sm_87 + sm_110a native SASS). The int8
#    twins would thrash the CUDA EP (DynamicQuantizeLinear has no CUDA
#    kernel; MatMulInteger alone does) — fp32 is the GPU config. Protocol +
#    driver: task/research/moonshine-gpu-protocol/summary.md.
# =============================================================================
# DECOUPLING CONTRACT: these two ARGs are the only external artifact
# references in this file; override to re-point either.
ARG BASE_IMAGE=nvidia/cuda:13.2.1-runtime-ubuntu24.04
ARG WHEELS_IMAGE=narandill/edge-wheels:cu132

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

FROM ubuntu:24.04 AS models-onnx
# Pinned: Masterx/moonshine-streaming-medium-ONNX @ 3c59ebafc67091803df5297
# 32365a259b0cc7b8a (2026-10-05) — fp32 5-graph set, MIT, token-for-token
# vs transformers per upstream README. frontend is fp32-only by design
# ("convolution state carry should stay exact").
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /models/moonshine-onnx-fp32 && cd /models/moonshine-onnx-fp32 \
    && for F in frontend.onnx encoder.onnx adapter.onnx cross_kv.onnx \
                decoder_kv.onnx streaming_config.json tokenizer.json; do \
        curl -fsSL --retry 5 -O \
            "https://huggingface.co/Masterx/moonshine-streaming-medium-ONNX/resolve/3c59ebafc67091803df529732365a259b0cc7b8a/$F" \
            || exit 1; \
    done \
    && printf '%s  %s\n' \
        bd97db4c43438034ca78dc444eed6bdfba03ea7a6e5f3c9c0d05693acf687a6c frontend.onnx \
        9696eb966b76abffcbab8d3e0829032faee14d7d2535f78dcc59d456b6bfe16c encoder.onnx \
        a3e71a3fdca1459b7eb751d85c8781e76c552b305b4e061dba5854ae9aeaf85e adapter.onnx \
        b12501cb46dbc269ac42daaff06ec92075c555f6d308b8e2f2bdeaa9df8404ed cross_kv.onnx \
        0241a270d277d45e7d66502f57ceb7d72b33802461cd37e2f250377e745f6116 decoder_kv.onnx \
        bc7f1dfbffeb99049afc93fab4d0ee32473961d56e857f46c3f75a03f2e6effd streaming_config.json \
        28bbc54a47dfa33673affa26010f23ac452685b18c76e81451d9b3ea25a54d54 tokenizer.json \
        > /tmp/moonshine-onnx.sha256 \
    && sha256sum --check --strict /tmp/moonshine-onnx.sha256 \
    && rm /tmp/moonshine-onnx.sha256

# DECOUPLING CONTRACT: these two ARGs are the only external artifact
# references in the runtime stage; override to re-point either.
FROM ${WHEELS_IMAGE} AS wheels
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="moonshine-streaming-stt" \
      org.opencontainers.image.description="Moonshine Streaming Medium 245M; CPU via moonshine-voice 0.1.5, GPU via fp32 ONNX on onnxruntime-gpu 1.31.1 (CUDA EP sm_87+sm_110a native); JP 7.2.1" \
      org.opencontainers.image.source="https://github.com/moonshine-ai/moonshine"
# MIT (runtimes + weights). Ubuntu 24.04 glibc 2.39 satisfies the manylinux_2_34 wheel.

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv libsndfile1 libcudnn9-cuda-13 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=models /models /models
COPY --from=models-onnx /models/moonshine-onnx-fp32 /models/moonshine-onnx-fp32
COPY --from=wheels /wheels/onnxruntime_gpu-*.whl /tmp/
COPY rootfs/ /

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    # Pinned: 0.1.5 ships aarch64 manylinux_2_31/2_34 wheels (ORT core inside).
    && /opt/venv/bin/pip install --no-cache-dir moonshine-voice==0.1.5 soundfile soxr \
    # GPU path: fp32 5-graph driver on our CUDA-13 ORT (sm_87+sm_110a SASS);
    # tokenizers = HF BPE decoder for the Masterx tokenizer.json.
    && /opt/venv/bin/pip install --no-cache-dir "tokenizers>=0.20,<1" \
    && /opt/venv/bin/pip install --no-cache-dir /tmp/onnxruntime_gpu-*.whl \
    && rm -rf /tmp/onnxruntime_gpu-*.whl \
    && chmod +x /usr/local/bin/moonshine-transcribe /usr/local/bin/moonshine-ort-transcribe /usr/local/bin/voice-server

ENV PATH="/opt/venv/bin:${PATH}" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    MOONSHINE_MODEL_DIR=/models/moonshine-streaming-en \
    MOONSHINE_MODEL_ARCH=5 \
    MOONSHINE_ONNX_DIR=/models/moonshine-onnx-fp32

# voice-api v1 (docs/voice-api.md); the transcribe CLI stays available
EXPOSE 8080
CMD ["voice-server"]
