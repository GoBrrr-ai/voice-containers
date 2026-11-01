#!/usr/bin/env bash
# CPU smoke test (no GPU needed): offline transcription of the 16 kHz test wav.
# Usage: ./smoke_stt.sh <image>
set -euo pipefail
IMAGE="${1:?usage: smoke_stt.sh <image>}"
AUDIO="/home/narandill/edge-containers/task/voice-data/smoke_tts_16k.wav"
[ -f "$AUDIO" ] || { echo "missing $AUDIO"; exit 1; }

echo "[smoke] 1/1 transcription (parakeet-tdt-transcribe, 25-language int8)"
OUT=$(docker run --rm -v "$AUDIO":/audio/x.wav:ro \
  --entrypoint parakeet-tdt-transcribe "$IMAGE" /audio/x.wav 2>/dev/null | tail -4)
echo "$OUT" | head -c 400; echo
[ -n "$OUT" ] || { echo "[smoke] FAIL: no transcript"; exit 1; }

echo "[smoke] PASS"
