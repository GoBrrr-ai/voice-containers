# syntax=docker/dockerfile:1
# =============================================================================
# piper1-tts — OpenAI-protocol TTS (voice-api v1, docs/voice-api.md).
# piper-tts 1.8.0 (piper1-gpl) on OUR onnxruntime-gpu 1.31.1 build
# (sm_87 + sm_90a + sm_110a SASS, sm_110a PTX); stdlib HTTP server, streamed
# per-sentence pcm chunks.
# =============================================================================
ARG BASE_IMAGE=nvidia/cuda:13.2.1-runtime-ubuntu24.04
ARG WHEELS_IMAGE=narandill/edge-wheels:cu132

FROM ${WHEELS_IMAGE} AS wheels

FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="piper1-tts" \
      org.opencontainers.image.description="TTS voice-api server: piper-tts 1.8.0 (piper1-gpl) on onnxruntime-gpu 1.31.1 (sm_87+sm_110); JP 7.2.1 CUDA 13.2 SBSA" \
      org.opencontainers.image.source="https://github.com/OHF-Voice/piper1-gpl"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv libcudnn9-cuda-13 libgomp1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=wheels /wheels/onnxruntime_gpu-1.31.1-cp312-cp312-linux_aarch64.whl /tmp/
COPY --from=wheels /wheels/piper_tts-1.8.0-cp39-abi3-linux_aarch64.whl /tmp/
COPY rootfs/ /

# Bake the default voice: the server must never phone HF at boot (airgap
# rule) and a cold download delays Ready past any sane health window.
# TODO(post-vllm): the 0.1.0 tag currently carries this as an overlay layer
# (see task/wiki/experiments.md 2026-11-03); until the wheels builder is
# rebuilt, refresh it with:
#   docker cp <warmed tts container>:/data/models/piper/*.onnx* /tmp/bake/
#   FROM narandill/piper1-tts:0.2.0  +  COPY *.onnx* /data/models/piper/
RUN mkdir -p /data/models/piper \
    && python3 -c "import urllib.request as u; [u.urlretrieve(f'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/{f}', f'/data/models/piper/{f}') for f in ('en_US-libritts_r-medium.onnx','en_US-libritts_r-medium.onnx.json')]" \
    && python3 -c "import urllib.request as u; u.urlretrieve('https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json', '/data/models/piper/voices.json')"

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir \
        /tmp/piper_tts-1.8.0-cp39-abi3-linux_aarch64.whl \
    # GPU runtime LAST (piper's wheel strips its ort dep; PyPI ort must not win)
    && /opt/venv/bin/pip install --no-cache-dir --force-reinstall /tmp/onnxruntime_gpu-1.31.1-cp312-cp312-linux_aarch64.whl \
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
    PIPER_PORT=8091 \
    PIPER_VOICE=en_US-libritts_r-medium \
    PIPER_USE_CUDA=true \
    PIPER_CACHE=/data/models/piper

EXPOSE 8091
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8091/health', timeout=4).status == 200 else 1)" || exit 1
CMD ["voice-server"]
