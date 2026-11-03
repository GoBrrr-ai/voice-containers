# syntax=docker/dockerfile:1
# =============================================================================
# kokoro-tts 0.2.0 — hexgrad/Kokoro-82M v1.0 via kokoro-onnx 0.6.1 (Apache-2.0)
#
# Ships the proven graph: model_uint8.onnx (static QLinear int8 — quantized
# weights with FLOAT32 COMPUTE). MEASURED 2026-11-03: the fp16 AND q4f16
# fused exports (onnx-community AND thehwh1teagle, both) crash kokoro-onnx
# on long utterances — "zero-size array to reduction operation maximum"
# in the chunk pipeline for 441/444 TTS-manifest paragraphs (short clips
# fine). Root cause: fp16 compute underflows the internal duration path
# on long chunks; our uint8 graph's fp32 compute is immune. Do not bake
# the fp16 graphs until upstream dur handling is fp32-safe.
#
# ONNX + voices-v1.0.bin on OUR onnxruntime-gpu
# 1.31.1 build (sm_87 + sm_90a + sm_110a SASS, sm_110a PTX). CUDA EP first,
# CPU EP fallback. Streaming: kokoro.create_stream() async generator
# (phoneme-chunked); RAM ~0.15-0.25 GB.
#
# NOTE: kokoro-onnx installed with --no-deps and our ort wheel force-installed
# LAST, so the [gpu] extra can never clobber our sm_87/sm_110 build.
# =============================================================================
ARG BASE_IMAGE=nvidia/cuda:13.2.1-runtime-ubuntu24.04
# Shared onnxruntime-gpu 1.31.1 wheel image (sm_87 + sm_90a + sm_110a SASS,
# sm_110a PTX; 75-minute source build, do NOT rebuild casually).
# DECOUPLING CONTRACT: this ARG is the only external artifact reference in this
# file — override it, or vendor the wheel and replace the COPY, to go standalone.
ARG ORT_WHEEL_IMAGE=narandill/edge-wheels:cu132

# --- model fetch stage --------------------------------------------------------
# Pinned to the onnx-community repo revision (unchanged since 2025-02-08):
# a resolve/main curl would silently pick up any future re-export.
# voices-v1.0.bin is a GitHub release asset (re-uploaded 2026-08-19 with the
# fixed graph contract) — verified by digest instead of a URL pin.
FROM ${BASE_IMAGE} AS models
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/kokoro \
    && curl -fL --retry 5 -o /opt/kokoro/model_uint8.onnx \
        "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/1939ad2a8e416c0acfeecc08a694d14ef25f2231/onnx/model_uint8.onnx" \
    && curl -fL --retry 5 -o /opt/kokoro/voices-v1.0.bin \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin" \
    && echo "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d  /opt/kokoro/voices-v1.0.bin" | sha256sum --check -

# --- final ---------------------------------------------------------------------
# Shared ort wheel enters as a stage (BuildKit does not expand vars in --from);
# override ORT_WHEEL_IMAGE to re-point it.
FROM ${ORT_WHEEL_IMAGE} AS ort-wheels

FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="kokoro-tts" \
      org.opencontainers.image.description="Kokoro-82M v1.0 int8 ONNX on kokoro-onnx 0.6.1 + onnxruntime-gpu 1.31.1 (sm_87+sm_110); JP 7.2.1 CUDA 13.2 SBSA" \
      org.opencontainers.image.source="https://github.com/thewh1teagle/kokoro-onnx"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv libcudnn9-cuda-13 libespeak-ng1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ort-wheels /wheels/onnxruntime_gpu-*.whl /tmp/
COPY --from=models /opt/kokoro /opt/kokoro
COPY rootfs/ /

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir \
        "numpy>=2.0.2" soundfile espeakng-loader phonemizer \
    # kokoro-onnx without deps: its ort requirement is satisfied by OUR wheel below
    && /opt/venv/bin/pip install --no-cache-dir --no-deps kokoro-onnx==0.6.1 \
    # GPU runtime LAST so it wins over anything above
    && /opt/venv/bin/pip install --no-cache-dir --force-reinstall /tmp/onnxruntime_gpu-*.whl \
    && rm -rf /tmp/onnxruntime* \
    && chmod +x /usr/local/bin/kokoro-say /usr/local/bin/kokoro-stream-demo

ENV PATH="/opt/venv/bin:${PATH}" \
    KOKORO_MODEL=/opt/kokoro/model_uint8.onnx \
    KOKORO_VOICES=/opt/kokoro/voices-v1.0.bin \
    KOKORO_VOICE=af_heart \
    KOKORO_LANG=en-us \
    KOKORO_SPEED=1.0

# Library image: use kokoro-say / kokoro-stream-demo, or import kokoro_onnx.
CMD ["kokoro-say", "--help"]
