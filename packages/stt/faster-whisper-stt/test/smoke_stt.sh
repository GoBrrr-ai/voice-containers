#!/usr/bin/env bash
# GPU smoke test (Thor): OpenAI-protocol STT round trip, plain + SSE stream.
# Retries until the server is ready (first boot downloads the model into the
# /data volume cache). Usage: ./smoke_stt.sh <image>
set -euo pipefail
IMAGE="${1:?usage: smoke_stt.sh <image>}"
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
AUDIO="$REPO/task/voice-data/smoke_tts_16k.wav"
[ -f "$AUDIO" ] || { echo "missing $AUDIO"; exit 1; }
NAME="fw-stt-smoke-$$"
PORT="18090"
RUNTIME=(--runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1)

for _ in $(seq 1 90); do
  USED=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
  [ "${USED:-0}" -lt 100000 ] && break
  echo "[smoke] GPU busy (${USED} MiB in use) - sleeping 20s"; sleep 20
done
echo "[smoke] GPU check: ${USED:-0} MiB in use by co-tenants (proceeding)"

docker ps -aq --filter "name=fw-stt-smoke" | xargs -r docker rm -f >/dev/null 2>&1 || true
docker run -d --name "$NAME" "${RUNTIME[@]}" -p "${PORT}:8090" \
  -v "$REPO/task/voice-data:/data" "$IMAGE" >/dev/null
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT

echo "[smoke] waiting for /health (first boot downloads the model)"
ok=""
for _ in $(seq 1 90); do
  if curl -sf -m 5 "http://localhost:${PORT}/health" >/dev/null 2>&1; then ok=1; break; fi
  sleep 5
done
[ -n "$ok" ] || { echo "[smoke] FAIL: server never went healthy"; docker logs "$NAME" 2>&1 | tail -8; exit 1; }

echo "[smoke] 1/2 transcription (multipart, model echo)"
OUT=$(curl -sf -m 300 -X POST "http://localhost:${PORT}/v1/audio/transcriptions" \
  -F "file=@${AUDIO};type=audio/wav" 2>/dev/null)
echo "$OUT" | head -c 300; echo
TEXT=$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('text',''))" "$OUT" 2>/dev/null || true)
[ -n "$TEXT" ] || { echo "[smoke] FAIL: empty transcript"; exit 1; }

echo "[smoke] 2/2 streaming partials (stream=true -> SSE)"
SSE=$(curl -sf -N -m 300 -X POST "http://localhost:${PORT}/v1/audio/transcriptions?stream=true" \
  -H "Content-Type: audio/wav" --data-binary @"$AUDIO" 2>/dev/null)
FINAL=$(printf '%s\n' "$SSE" | sed -n 's/^data: //p' | tail -1)
python3 -c "import json,sys; d=json.loads(sys.argv[1]); assert d.get('text'), 'empty final' " "$FINAL" \
  || { echo "[smoke] FAIL: no SSE final"; exit 1; }
echo "[smoke] sse final: $(printf '%s' "$FINAL" | head -c 120)"
echo "[smoke] PASS"
