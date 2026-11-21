# syntax=docker/dockerfile:1
# =============================================================================
# faster-whisper-stt — OpenAI-protocol STT (voice-api v1, docs/voice-api.md).
# faster-whisper 1.2.1 on OUR CTranslate2 4.8.4 build
# (sm_87 + sm_110 SASS/PTX, CUDA 13.2 SBSA); stdlib HTTP server, SSE partials.
# =============================================================================
ARG BASE_IMAGE=nvidia/cuda:13.2.1-runtime-ubuntu24.04
# DECOUPLING CONTRACT: only external artifact reference (shared wheels builder).
ARG WHEELS_IMAGE=narandill/edge-wheels:cu132

# Shared wheels enter as a stage (BuildKit does not expand vars in --from).
FROM ${WHEELS_IMAGE} AS wheels

FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="faster-whisper-stt" \
      org.opencontainers.image.description="STT voice-api server: faster-whisper 1.2.1 on CTranslate2 4.8.4 (sm_87+sm_110); JP 7.2.1 CUDA 13.2 SBSA" \
      org.opencontainers.image.source="https://github.com/SYSTRAN/faster-whisper"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv libcudnn9-cuda-13 libgomp1 libssl3t64 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=wheels /wheels/ctranslate2-4.8.4-cp312-cp312-linux_aarch64.whl /tmp/
# CT2's python wheel links the C++ lib built in the builder; ship it too.
COPY --from=wheels /opt/ct2-dist/lib/ /usr/local/lib/
COPY rootfs/ /

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir \
        "faster-whisper==1.2.1" \
        "av==15.1.0" \
    # GPU runtime LAST so it wins over any CPU wheel faster-whisper drags in
    && /opt/venv/bin/pip install --no-cache-dir --force-reinstall /tmp/ctranslate2-4.8.4-cp312-cp312-linux_aarch64.whl \
    && rm -rf /tmp/*.whl \
    && chmod +x /usr/local/bin/voice-server \
    && ldconfig

# Strip the CUDA forward-compat pair the cu13.2 base ships at
# /usr/local/cuda/compat{,_orin}: nvidia-container-toolkit 1.19.1 injects a
# createContainer hook (`enable-cuda-compat`) into every container create and
# that hook panics parsing these libs' ELF notes (slice [:73] vs cap 71) —
# docker run dies with exit 125 before the entrypoint, no logs. The 595.58.03
# libs are never newer than the fleet's 595.78 driver, so the hook could only
# no-op; libcuda.so.1 is host-injected at run time regardless.
# Post-mortem + repro: docs/postmortem-cuda-compat-hook.md
RUN rm -rf /usr/local/cuda/compat /usr/local/cuda/compat_orin

ENV PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/data/hf-cache \
    WHISPER_PORT=8090 \
    WHISPER_MODEL=tiny-int8 \
    WHISPER_BEAM_SIZE=1 \
    WHISPER_COMPUTE_TYPE=default \
    WHISPER_LANGUAGE=en \
    WHISPER_DEVICE=cuda \
    WHISPER_DATA_DIR=/data/models/faster-whisper

EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8090/health', timeout=4).status == 200 else 1)" || exit 1
CMD ["voice-server"]
