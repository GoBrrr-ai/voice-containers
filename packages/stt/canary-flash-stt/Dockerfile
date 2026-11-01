# syntax=docker/dockerfile:1
# =============================================================================
# canary-flash-stt 0.1.0 — NVIDIA Canary-180M-Flash (CC-BY-4.0)
#
# FastConformer encoder + 4-layer decoder (en/de/es/fr, ASR + X->EN), ~182M
# params. INT8 sherpa-onnx export (encoder 132.7 MB + decoder 74.4 MB) on the
# CPU EP — the strongest ASR here that also has an easy GPU path (FP16 ONNX,
# see README). Per-utterance after VAD chunking: no native streaming.
# LibriSpeech-other 2.87 WER, SPGI 1.95.
# =============================================================================
FROM ubuntu:24.04 AS models
# Pinned k2-fsa asr-models release asset (int8, en-es-de-fr).
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates bzip2 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /tmp/canary && cd /tmp/canary \
    && curl -fL --retry 5 -o canary.tar.bz2 \
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2" \
    && tar -xjf canary.tar.bz2 \
    && mkdir -p /models/canary-flash \
    && for F in encoder.int8.onnx decoder.int8.onnx tokens.txt; do \
        cp "sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/$F" /models/canary-flash/; \
    done \
    && rm -rf /tmp/canary

FROM ubuntu:24.04
LABEL org.opencontainers.image.title="canary-flash-stt" \
      org.opencontainers.image.description="NVIDIA Canary-180M-Flash int8 via sherpa-onnx 2.0.1; VAD-chunked offline ASR + X->EN translation, CPU; JP 7.2.1" \
      org.opencontainers.image.source="https://huggingface.co/soniqo/Canary-180M-Flash-ONNX"
# Weights CC-BY-4.0 (upstream nvidia/canary-180m-flash); sherpa-onnx Apache-2.0.

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=models /models /models
COPY rootfs/ /

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    # PyPI aarch64 CPU wheel; v2.0.1 fixes canary lang2id derivation.
    && /opt/venv/bin/pip install --no-cache-dir sherpa-onnx==2.0.1 soundfile \
    && chmod +x /usr/local/bin/canary-transcribe

ENV PATH="/opt/venv/bin:${PATH}" \
    CANARY_ENCODER=/models/canary-flash/encoder.int8.onnx \
    CANARY_DECODER=/models/canary-flash/decoder.int8.onnx \
    CANARY_VOCAB=/models/canary-flash/tokens.txt \
    CANARY_SRC_LANG=en \
    CANARY_TGT_LANG=en

CMD ["canary-transcribe", "--help"]
