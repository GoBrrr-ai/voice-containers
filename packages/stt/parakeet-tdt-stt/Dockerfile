# syntax=docker/dockerfile:1
# =============================================================================
# parakeet-tdt-stt 0.1.0 — NVIDIA Parakeet TDT 0.6B v3 (CC-BY-4.0)
#
# Token-and-Duration Transducer, 25 languages — the most accurate model in
# its class (LS-clean ~1.8% WER class). INT8 sherpa-onnx export (encoder
# 652 MB + decoder 11.8 + joiner 6.4 MB) on the CPU EP; ~0.65-0.8 GB loaded.
# Offline per utterance — VAD-chunk long audio upstream (pattern in README).
# A GGUF for our parakeet.cpp runtime also exists (f16 1.44 GB / q8_0 941 MB)
# if a GPU variant is ever wanted — see README.
# =============================================================================
FROM ubuntu:24.04 AS models
# Pinned k2-fsa asr-models release asset (int8, 25 languages).
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates bzip2 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /tmp/tdt && cd /tmp/tdt \
    && curl -fL --retry 5 -o tdt.tar.bz2 \
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2" \
    && tar -xjf tdt.tar.bz2 \
    && mkdir -p /models/parakeet-tdt-v3 \
    && for F in encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt; do \
        cp "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/$F" /models/parakeet-tdt-v3/; \
    done \
    && rm -rf /tmp/tdt

FROM ubuntu:24.04
LABEL org.opencontainers.image.title="parakeet-tdt-stt" \
      org.opencontainers.image.description="NVIDIA Parakeet TDT 0.6B v3 int8 via sherpa-onnx 2.0.1; 25 languages, VAD-chunked offline ASR, CPU; JP 7.2.1" \
      org.opencontainers.image.source="https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3"
# Weights CC-BY-4.0; sherpa-onnx Apache-2.0.

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=models /models /models
COPY rootfs/ /

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir sherpa-onnx==2.0.1 soundfile \
    && chmod +x /usr/local/bin/parakeet-tdt-transcribe

ENV PATH="/opt/venv/bin:${PATH}" \
    TDT_ENCODER=/models/parakeet-tdt-v3/encoder.int8.onnx \
    TDT_DECODER=/models/parakeet-tdt-v3/decoder.int8.onnx \
    TDT_JOINER=/models/parakeet-tdt-v3/joiner.int8.onnx \
    TDT_VOCAB=/models/parakeet-tdt-v3/tokens.txt

CMD ["parakeet-tdt-transcribe", "--help"]
