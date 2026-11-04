# syntax=docker/dockerfile:1
# =============================================================================
# paradee-tts 0.2.0 — sahilmahendrakar/Paradee-8M-v1.2 (Apache-2.0, 7.88M)
#
# 8.5 MB int8 ONNX distillation of Kokoro-82M — FOUR voices: af_heart,
# bf_isabella, am_puck, bm_fable (af_heart: UTMOS 4.40, WER 5.8%).
# Runs on OUR onnxruntime-gpu 1.31.1 (sm_87 + sm_110a); CPU fallback is the
# safe default (int8 graph ≈ 62x realtime on one thread, upstream card).
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
# Pinned: not on PyPI; tag v1.2 == main HEAD == aae3cc4 (2026-10-25).
FROM ${BASE_IMAGE} AS paradee-src
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --depth 1 --branch v1.2 https://github.com/sahilmahendrakar/paradee /src/paradee

FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="paradee-tts" \
      org.opencontainers.image.description="Paradee-8M-v1.2 int8 ONNX (Kokoro distillation, 4 voices) on onnxruntime-gpu 1.31.1 (sm_87+sm_110); JP 7.2.1 CUDA 13.2 SBSA" \
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
    # GPU runtime must be the ONLY ort: the CPU wheel paradee drags in ships
    # its own onnxruntime_pybind11_state.*.so which shadows the GPU one
    # (silent CPU fallback) — so remove both dist names before installing ours
    && /opt/venv/bin/pip uninstall -y onnxruntime onnxruntime-gpu \
    && /opt/venv/bin/pip install --no-cache-dir --force-reinstall /tmp/onnxruntime_gpu-*.whl \
    && rm -rf /tmp/onnxruntime* /tmp/paradee \
    && chmod +x /usr/local/bin/paradee-say /usr/local/bin/paradee-stream /usr/local/bin/voice-server

# Bake the model: warm-run real syntheses for Paradee-8M-v1.2 (int8) so the
# onnx weights AND misaki's G2P dictionary data land in the caches (offline
# OK). Two voices on purpose — af_heart (American) + bf_isabella (British):
# misaki's G2P dictionaries load lazily PER ACCENT, and the bf_/bm_ voices
# need the British one.
# Runs on the CPU EP (build stage has no GPU need).
ENV HF_HOME=/opt/hf-cache \
    XDG_CACHE_HOME=/opt/hf-cache/xdg \
    PARADEE_MODEL_VERSION=1.2
RUN /opt/venv/bin/paradee "Warm up the Paradee cache." -o /tmp/warm1.wav --model-version 1.2 --voice af_heart \
    && /opt/venv/bin/paradee "Warm up the Paradee cache." -o /tmp/warm2.wav --model-version 1.2 --voice bf_isabella \
    && du -sh /opt/hf-cache && rm -f /tmp/warm1.wav /tmp/warm2.wav

ENV PATH="/opt/venv/bin:${PATH}"

# Library image: use paradee-say / paradee-stream, or import paradee.
# voice-api v1 (docs/voice-api.md); see the package README for the CLIs
EXPOSE 8880
CMD ["voice-server"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8880/health', timeout=4).status == 200 else 1)" || exit 1
