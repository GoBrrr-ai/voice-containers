#!/usr/bin/env bash
# CPU smoke test (no GPU needed): streaming transcription (160 ms latency
# profile is baked into the graph; sherpa chunks internally).
# Usage: ./smoke_stt.sh <image>
set -euo pipefail
IMAGE="${1:?usage: smoke_stt.sh <image>}"
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
AUDIO="$REPO/task/voice-data/smoke_tts_16k.wav"
[ -f "$AUDIO" ] || { echo "missing $AUDIO"; exit 1; }
# nvidia/cuda base's brand list lacks a Thor entry -> the runtime requirement
# check fails on JP 7.2.1 without this (repo-wide smoke convention).
RUNTIME=(--runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1)

echo "[smoke] 1/2 streaming transcription (nemotron-stream-transcribe)"
OUT=$(docker run --rm "${RUNTIME[@]}" -v "$AUDIO":/audio/x.wav:ro \
  --entrypoint nemotron-stream-transcribe "$IMAGE" /audio/x.wav 2>/dev/null | tail -4)
echo "$OUT" | head -c 400; echo
[ -n "$OUT" ] || { echo "[smoke] FAIL: no transcript"; exit 1; }

echo "[smoke] 2/2 voice-server SSE partials (stream=true)"
CID=$(docker run -d "${RUNTIME[@]}" -e NEMOSTREAM_PROVIDER=cpu \
  -p 127.0.0.1:18095:8080 "$IMAGE")
trap 'docker rm -f "$CID" >/dev/null 2>&1' EXIT
sleep 6
SSE=$(curl -sN -X POST "http://127.0.0.1:18095/v1/audio/transcriptions?stream=true" \
  -F file=@"$AUDIO" || true)
docker rm -f "$CID" >/dev/null 2>&1; trap - EXIT
echo "$SSE" | grep -q "^event: partial" || { echo "[smoke] FAIL: no partial events"; exit 1; }
echo "$SSE" | grep -q "^event: final" || { echo "[smoke] FAIL: no final event"; exit 1; }
echo "$SSE" | grep "^data:" | tail -1 | head -c 200; echo

echo "[smoke] PASS"
