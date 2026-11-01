#!/usr/bin/env bash
# CPU smoke test (no GPU needed): offline transcription of the 16 kHz test wav.
# Usage: ./smoke_stt.sh <image>
set -euo pipefail
IMAGE="${1:?usage: smoke_stt.sh <image>}"
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
AUDIO="$REPO/task/voice-data/smoke_tts_16k.wav"
[ -f "$AUDIO" ] || { echo "missing $AUDIO"; exit 1; }

echo "[smoke] 1/2 transcription en->en (canary-transcribe)"
OUT=$(docker run --rm -v "$AUDIO":/audio/x.wav:ro \
  --entrypoint canary-transcribe "$IMAGE" /audio/x.wav 2>/dev/null | tail -4)
echo "$OUT" | head -c 400; echo
[ -n "$OUT" ] || { echo "[smoke] FAIL: no transcript"; exit 1; }

echo "[smoke] 2/2 translation en->de (canary-transcribe --tgt de)"
OUT=$(docker run --rm -v "$AUDIO":/audio/x.wav:ro \
  --entrypoint canary-transcribe "$IMAGE" /audio/x.wav --tgt de 2>/dev/null | tail -4)
echo "$OUT" | head -c 400; echo
[ -n "$OUT" ] || { echo "[smoke] FAIL: no translation"; exit 1; }

echo "[smoke] PASS"
