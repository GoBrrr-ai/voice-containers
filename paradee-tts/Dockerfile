# syntax=docker/dockerfile:1
# =============================================================================
# paradee-tts 0.1.0 — sahilmahendrakar/Paradee-8M-v1.0 (Apache-2.0, 8.07M)
#
# 8.6 MB int8 ONNX distillation of Kokoro-82M (UTMOS 4.41, EN WER 0.71).
# Runs on OUR onnxruntime-gpu 1.31.1 (sm_87 + sm_110a); CPU fallback is the
# safe default (int8 graph = 18x realtime on one thread).
#
# No native streaming in the graph (whole-utterance synthesis, TTFA ~1 s) —
# the paradee-stream helper implements sentence-chunked streaming. Model is
# baked by warming the paradee HF cache at build time (no runtime download).
# =============================================================================
ARG BASE_IMAGE=nvidia/cuda:13.2.1-runtime-ubuntu24.04
# Shared onnxruntime-gpu 1.31.1 wheel image (sm_87 + sm_90a + sm_110a SASS,
# sm_110a PTX; 75-minute source build, do NOT rebuild casually).
# DECOUPLING CONTRACT: this ARG is the only external artifact reference in this
# file — override it, or vendor the wheel and replace the COPY, to go standalone.
ARG ORT_WHEEL_IMAGE=narandill/edge-wheels:cu132

# Shared ort wheel enters as a stage (BuildKit does not expand vars in --from);
# override ORT_WHEEL_IMAGE to re-point it.
FROM ${ORT_WHEEL_IMAGE} AS ort-wheels

# --- paradee source (git stays out of the final image) ------------------------
FROM ${BASE_IMAGE} AS paradee-src
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --depth 1 https://github.com/sahilmahendrakar/paradee /src/paradee

FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="paradee-tts" \
      org.opencontainers.image.description="Paradee-8M-v1.0 int8 ONNX (Kokoro distillation) on onnxruntime-gpu 1.31.1 (sm_87+sm_110); JP 7.2.1 CUDA 13.2 SBSA" \
      org.opencontainers.image.source="https://github.com/sahilmahendrakar/paradee"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv libcudnn9-cuda-13 libespeak-ng1 libsndfile1 espeak-ng-data \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ort-wheels /wheels/onnxruntime_gpu-*.whl /tmp/
COPY --from=paradee-src /src/paradee /tmp/paradee
COPY rootfs/ /

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir "numpy<3" soundfile \
    # paradee's own dep set = misaki (plain) + spacy + phonemizer-fork +
    # espeakng-loader + num2words — deliberately NOT misaki[en], whose extra
    # pulls spacy-curated-transformers -> torch
    && /opt/venv/bin/pip install --no-cache-dir /tmp/paradee \
    # GPU runtime LAST so it wins over the CPU ort paradee drags in
    && /opt/venv/bin/pip install --no-cache-dir --force-reinstall /tmp/onnxruntime_gpu-*.whl \
    && rm -rf /tmp/onnxruntime* /tmp/paradee \
    && chmod +x /usr/local/bin/paradee-say /usr/local/bin/paradee-stream

# Bake the model: warm-run a real synthesis for Paradee-8M-v1.0 (int8) so the
# onnx weights AND misaki's G2P dictionary data land in the caches (offline OK).
# Runs on the CPU EP (build stage has no GPU need).
ENV HF_HOME=/opt/hf-cache \
    XDG_CACHE_HOME=/opt/hf-cache/xdg \
    PARADEE_MODEL_VERSION=1.0
RUN /opt/venv/bin/paradee "Warm up the Paradee cache." -o /tmp/warm.wav --model-version 1.0 \
    && du -sh /opt/hf-cache && rm -f /tmp/warm.wav

ENV PATH="/opt/venv/bin:${PATH}"

# Library image: use paradee-say / paradee-stream, or import paradee.
CMD ["paradee-say", "--help"]
