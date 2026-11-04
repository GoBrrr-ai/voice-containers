# syntax=docker/dockerfile:1
# =============================================================================
# nemotron-unified-asr 0.2.0 — nvidia/nemotron-labs-unified-asr-0.6b (OpenMDW)
#
# Two baked variants, selected by env/opts.model:
#   nemotron-labs-unified-asr-0.6b-q8_0.gguf  (939 MB — GPU/CUDA0, default)
#   nemotron-labs-unified-asr-0.6b-q4_k.gguf  (674 MB — the Orin Nano 8 GB
#     voice-budget variant; env-swap only, same binary; mudler validates
#     the NVIDIA quants at WER 0 against NeMo)
#
# Runs on mudler/parakeet.cpp (GGML) with native sm_87 (AGX Orin) + sm_110a
# (AGX Thor) cubins + compute_110 PTX baked by the shared builder image.
#
# Streaming: buffered "unified" streaming — `--stream --stream-latency <ms>`
# with 160/240/.../2080 ms presets; parakeet.cpp's parity tests show the
# streaming loop is token-identical to NeMo's.
# =============================================================================
ARG BASE_IMAGE=nvidia/cuda:13.2.1-runtime-ubuntu24.04
# Shared parakeet.cpp runtime dist (GGML, sm_87+sm_110a native cubins + PTX).
# DECOUPLING CONTRACT: this ARG is the only external artifact reference in this
# file — override it (own registry build / vendored dist) and nothing else changes.
ARG PARAKEET_DIST_IMAGE=narandill/edge-builder:parakeet-cpp

# --- model fetch stage (keeps curl out of the final image) -------------------
FROM ${BASE_IMAGE} AS models
# Pinned GGUF repo revision — resolve/main would silently pick up new quants.
ARG GGUF_BASE_URL=https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/07a08a3a405d20c5434b9a1442f6e6cf571f57ad
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /models \
    && curl -fL --retry 5 -o /models/nemotron-labs-unified-asr-0.6b-q8_0.gguf \
        "${GGUF_BASE_URL}/nemotron-labs-unified-asr-0.6b-q8_0.gguf" \
    && curl -fL --retry 5 -o /models/nemotron-labs-unified-asr-0.6b-q4_k.gguf \
        "${GGUF_BASE_URL}/nemotron-labs-unified-asr-0.6b-q4_k.gguf" \
    && curl -fL --retry 5 -o /models/silero-vad-f16.gguf \
        "${GGUF_BASE_URL}/silero-vad-f16.gguf"

# --- final -------------------------------------------------------------------
# --- shared runtime dist (stage form: BuildKit does not expand vars in --from) ---
FROM ${PARAKEET_DIST_IMAGE} AS parakeet-dist

# --- final -------------------------------------------------------------------
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="nemotron-unified-asr" \
      org.opencontainers.image.description="nvidia/nemotron-labs-unified-asr-0.6b q8_0 GGUF on parakeet.cpp; sm_87+sm_110 (JP 7.2.1, CUDA 13.2 SBSA)" \
      org.opencontainers.image.source="https://github.com/mudler/parakeet.cpp"

COPY --from=parakeet-dist /opt/parakeet-dist /opt/parakeet-dist
COPY --from=models /models /models
COPY rootfs/ /

# libgomp for the ggml/parakeet binaries (OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 python3 \
    && rm -rf /var/lib/apt/lists/*

RUN echo "/opt/parakeet-dist/lib" > /etc/ld.so.conf.d/parakeet-dist.conf \
    && echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda-lib64.conf \
    && chmod +x /usr/local/bin/parakeet-entrypoint.sh /usr/local/bin/stt-transcribe \
    && ldconfig

ENV PATH="/opt/parakeet-dist/bin:${PATH}" \
    PARAKEET_MODEL=/models/nemotron-labs-unified-asr-0.6b-q8_0.gguf \
    PARAKEET_VAD_MODEL=/models/silero-vad-f16.gguf \
    # Nemotron unified streaming: on by default (0.16..2.08 s presets)
    PARAKEET_STREAM=1 \
    PARAKEET_STREAM_LATENCY=0.56 \
    PARAKEET_LANG=en \
    PARAKEET_HOST=0.0.0.0 \
    PARAKEET_PORT=8080

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/parakeet-entrypoint.sh"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD bash -c 'exec 3<>/dev/tcp/127.0.0.1/8080' || exit 1
