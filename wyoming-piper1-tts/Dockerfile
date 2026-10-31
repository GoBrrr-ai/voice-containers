# syntax=docker/dockerfile:1
# =============================================================================
# wyoming-piper1-tts (standalone) — the TTS half of the former combined wyoming image.
# wyoming 1.10.2 + wyoming-piper 2.5.2 + piper-tts 1.8.0 on OUR onnxruntime-gpu
# 1.31.1 build (sm_87 + sm_90a + sm_110a SASS, sm_110a PTX).
# Same PIPER_* env contract as the combined image -> HA config drop-in.
# =============================================================================
ARG BASE_IMAGE=nvidia/cuda:13.2.1-runtime-ubuntu24.04
ARG WHEELS_IMAGE=narandill/edge-wheels:cu132

FROM ${WHEELS_IMAGE} AS wheels

FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="wyoming-piper1-tts" \
      org.opencontainers.image.description="Wyoming TTS server: piper-tts 1.8.0 (piper1-gpl) on onnxruntime-gpu 1.31.1 (sm_87+sm_110); JP 7.2.1 CUDA 13.2 SBSA" \
      org.opencontainers.image.source="https://github.com/OHF-Voice/wyoming-piper"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv libcudnn9-cuda-13 libgomp1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=wheels /wheels/onnxruntime_gpu-1.31.1-cp312-cp312-linux_aarch64.whl /tmp/
COPY --from=wheels /wheels/piper_tts-1.8.0-cp39-abi3-linux_aarch64.whl /tmp/
COPY --from=wheels /wheels/wyoming_piper-2.5.2-py3-none-any.whl /tmp/
COPY rootfs/ /

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir "wyoming[zeroconf]==1.10.2" \
    && /opt/venv/bin/pip install --no-cache-dir \
        /tmp/piper_tts-1.8.0-cp39-abi3-linux_aarch64.whl \
        /tmp/wyoming_piper-2.5.2-py3-none-any.whl \
    # GPU runtime LAST (piper's wheel strips its ort dep; PyPI ort must not win)
    && /opt/venv/bin/pip install --no-cache-dir --force-reinstall /tmp/onnxruntime_gpu-1.31.1-cp312-cp312-linux_aarch64.whl \
    && rm -rf /tmp/*.whl \
    && chmod +x /usr/local/bin/run-piper.sh \
    && ldconfig

ENV PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/data/hf-cache \
    PIPER_PORT=10200 \
    PIPER_VOICE=en_US-libritts_r-medium \
    PIPER_USE_CUDA=true \
    PIPER_CACHE=/data/models/piper \
    PIPER_ZEROCONF_NAME=wyoming-piper-tts \
    PIPER_DEBUG=true

EXPOSE 10200
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD bash -c 'exec 3<>/dev/tcp/localhost/10200' || exit 1
ENTRYPOINT ["/usr/local/bin/run-piper.sh"]
