#!/usr/bin/env bash
# CPU smoke test (no GPU needed): chunked streaming transcription (160 ms
# profile baked into the graph).
# Usage: ./smoke_stt.sh <image>
set -euo pipefail
IMAGE="${1:?usage: smoke_stt.sh <image>}"
AUDIO="/home/narandill/edge-containers/task/voice-data/smoke_tts_16k.wav"
[ -f "$AUDIO" ] || { echo "missing $AUDIO"; exit 1; }

echo "[smoke] 1/2 chunked streaming transcription (160 ms chunks)"
OUT=$(docker run --rm -v "$AUDIO":/audio/x.wav:ro \
  --entrypoint nemotron-stream-transcribe "$IMAGE" /audio/x.wav 2>/dev/null | tail -4)
echo "$OUT" | head -c 400; echo
[ -n "$OUT" ] || { echo "[smoke] FAIL: no transcript"; exit 1; }

echo "[smoke] 2/2 wrong-chunk guard (must refuse, not silently drop audio)"
docker run --rm -v "$AUDIO":/audio/x.wav:ro \
  --entrypoint nemotron-stream-transcribe "$IMAGE" /audio/x.wav --chunk-ms 80 >/dev/null 2>&1 \
  && { echo "[smoke] FAIL: 80 ms chunking should be refused"; exit 1; } || true

echo "[smoke] PASS"
