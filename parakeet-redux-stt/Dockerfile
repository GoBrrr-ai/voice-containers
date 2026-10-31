# syntax=docker/dockerfile:1
# =============================================================================
# parakeet-redux-stt 0.1.0 — moondream/parakeet-redux (CC-BY-4.0, 25 EU langs)
#
# Two baked variants, selected by env:
#   redux-q8_0.gguf    (~898 MB, dequantized — GPU/CUDA0, default)
#   redux-packed.gguf  (~203 MB, packed ternary — CPU dotprod, PARAKEET_DEVICE=cpu)
# Runs on mudler/parakeet.cpp with native sm_87 + sm_110a cubins + compute_110 PTX.
#
# Streaming: offline TDT model — chunked pseudo-streaming via `--vad`
# (silero cuts long audio at >=0.2 s pauses); true streaming lives in the
# nemotron-unified-asr / nemotron-3.5 images (same binary).
# =============================================================================
ARG BASE_IMAGE=nvidia/cuda:13.2.1-runtime-ubuntu24.04
# Shared parakeet.cpp runtime dist (GGML, sm_87+sm_110a native cubins + PTX).
# DECOUPLING CONTRACT: this ARG is the only external artifact reference in this
# file — override it (own registry build / vendored dist) and nothing else changes.
ARG PARAKEET_DIST_IMAGE=futureproofhomes/edge-builder:parakeet-cpp

FROM ${BASE_IMAGE} AS models
ARG GGUF_BASE_URL=https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /models \
    && curl -fL --retry 5 -o /models/redux-q8_0.gguf \
        "${GGUF_BASE_URL}/redux-q8_0.gguf" \
    && curl -fL --retry 5 -o /models/redux-packed.gguf \
        "${GGUF_BASE_URL}/redux-packed.gguf" \
    && curl -fL --retry 5 -o /models/silero-vad-f16.gguf \
        "${GGUF_BASE_URL}/silero-vad-f16.gguf"

# --- shared runtime dist (stage form: BuildKit does not expand vars in --from) ---
FROM ${PARAKEET_DIST_IMAGE} AS parakeet-dist

# --- final -------------------------------------------------------------------
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="parakeet-redux-stt" \
      org.opencontainers.image.description="moondream/parakeet-redux (178M ternary) q8_0+packed GGUF on parakeet.cpp; sm_87+sm_110 (JP 7.2.1, CUDA 13.2 SBSA)" \
      org.opencontainers.image.source="https://github.com/mudler/parakeet.cpp"

COPY --from=parakeet-dist /opt/parakeet-dist /opt/parakeet-dist
COPY --from=models /models /models
COPY rootfs/ /

# libgomp for the ggml/parakeet binaries (OpenMP runtime)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN echo "/opt/parakeet-dist/lib" > /etc/ld.so.conf.d/parakeet-dist.conf \
    && echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda-lib64.conf \
    && chmod +x /usr/local/bin/parakeet-entrypoint.sh /usr/local/bin/stt-transcribe \
    && ldconfig

ENV PATH="/opt/parakeet-dist/bin:${PATH}" \
    PARAKEET_MODEL=/models/redux-q8_0.gguf \
    # ultra-light CPU variant: docker run -e PARAKEET_DEVICE=cpu -e PARAKEET_MODEL=/models/redux-packed.gguf ...
    # VAD chunking is OPT-IN (assistant pipeline gates audio upstream);
    # enable for long-file/batch mode: -e PARAKEET_VAD_MODEL=/models/silero-vad-f16.gguf
    PARAKEET_VAD_MODEL="" \
    PARAKEET_HOST=0.0.0.0 \
    PARAKEET_PORT=8080

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/parakeet-entrypoint.sh"]
