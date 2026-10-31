# syntax=docker/dockerfile:1
# =============================================================================
# wyoming-faster-whisper (standalone) — the STT half of the former combined wyoming image.
# wyoming 1.10.2 + wyoming-faster-whisper 3.8.1 + faster-whisper 1.2.1 on OUR
# CTranslate2 4.8.4 build (sm_87 + sm_110 SASS/PTX, CUDA 13.2 SBSA).
# Same WHISPER_* env contract as the combined image -> HA config drop-in.
# =============================================================================
ARG BASE_IMAGE=nvidia/cuda:13.2.1-runtime-ubuntu24.04
# DECOUPLING CONTRACT: only external artifact reference (shared wheels builder).
ARG WHEELS_IMAGE=narandill/edge-wheels:cu132

# Shared wheels enter as a stage (BuildKit does not expand vars in --from).
FROM ${WHEELS_IMAGE} AS wheels

FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="wyoming-faster-whisper" \
      org.opencontainers.image.description="Wyoming STT server: faster-whisper 1.2.1 on CTranslate2 4.8.4 (sm_87+sm_110); JP 7.2.1 CUDA 13.2 SBSA" \
      org.opencontainers.image.source="https://github.com/OHF-Voice/wyoming-faster-whisper"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv libcudnn9-cuda-13 libgomp1 libssl3t64 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=wheels /wheels/ctranslate2-4.8.4-cp312-cp312-linux_aarch64.whl /tmp/
COPY --from=wheels /wheels/wyoming_faster_whisper-3.8.1-py3-none-any.whl /tmp/
COPY --from=wheels /wheels/pysilero_vad-3.4.0-cp39-abi3-manylinux_2_28_aarch64.whl /tmp/
# CT2's python wheel links the C++ lib built in the builder; ship it too.
COPY --from=wheels /opt/ct2-dist/lib/ /usr/local/lib/
COPY rootfs/ /

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir \
        "wyoming[zeroconf]==1.10.2" \
        "faster-whisper==1.2.1" \
        "av==15.1.0" \
        /tmp/pysilero_vad-3.4.0-cp39-abi3-manylinux_2_28_aarch64.whl \
    && /opt/venv/bin/pip install --no-cache-dir /tmp/wyoming_faster_whisper-3.8.1-py3-none-any.whl \
    # GPU runtime LAST so it wins over any CPU wheel faster-whisper drags in
    && /opt/venv/bin/pip install --no-cache-dir --force-reinstall /tmp/ctranslate2-4.8.4-cp312-cp312-linux_aarch64.whl \
    && rm -rf /tmp/*.whl \
    && chmod +x /usr/local/bin/run-whisper.sh \
    && ldconfig

ENV PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/data/hf-cache \
    WHISPER_PORT=10300 \
    WHISPER_MODEL=tiny-int8 \
    WHISPER_BEAM_SIZE=1 \
    WHISPER_COMPUTE_TYPE=default \
    WHISPER_LANGUAGE=en \
    WHISPER_DEVICE=cuda \
    WHISPER_DATA_DIR=/data/models/faster-whisper \
    WHISPER_ZEROCONF_NAME=wyoming-faster-whisper \
    WHISPER_DEBUG=true

EXPOSE 10300
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD bash -c 'exec 3<>/dev/tcp/localhost/10300' || exit 1
ENTRYPOINT ["/usr/local/bin/run-whisper.sh"]
