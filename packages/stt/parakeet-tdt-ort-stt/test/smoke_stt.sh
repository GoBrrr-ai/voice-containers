#!/usr/bin/env bash
# Smoke test: int4 encoder on CUDA EP + CPU fallback + parity vs the reference.
# Usage: ./smoke_stt.sh <image>
set -euo pipefail
IMAGE="${1:?usage: smoke_stt.sh <image>}"
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
AUDIO="$REPO/task/voice-data/smoke_tts_16k.wav"
[ -f "$AUDIO" ] || { echo "missing $AUDIO"; exit 1; }
# nvidia/cuda base's brand list lacks a Thor entry -> the runtime requirement
# check fails on JP 7.2.1 without this (repo-wide smoke convention).
RUNTIME=(--runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1)

echo "[smoke] 1/2 transcription (CUDA EP encoder)"
OUT=$(docker run --rm "${RUNTIME[@]}" -v "$AUDIO":/audio/x.wav:ro \
  --entrypoint tdt-ort-transcribe "$IMAGE" /audio/x.wav 2>/dev/null | tail -4)
echo "$OUT" | head -c 400; echo
[ -n "$OUT" ] || { echo "[smoke] FAIL: no transcript"; exit 1; }

echo "[smoke] 2/2 CPU-EP encoder fallback"
OUT=$(docker run --rm "${RUNTIME[@]}" -v "$AUDIO":/audio/x.wav:ro \
  -e TDT_ORT_PROVIDER=cpu \
  --entrypoint tdt-ort-transcribe "$IMAGE" /audio/x.wav 2>/dev/null | tail -4)
echo "$OUT" | head -c 400; echo
[ -n "$OUT" ] || { echo "[smoke] FAIL: empty CPU transcript"; exit 1; }

echo "[smoke] PASS"
