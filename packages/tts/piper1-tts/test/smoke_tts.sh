#!/usr/bin/env bash
# GPU smoke test (Thor): OpenAI-protocol TTS round trip, wav + streamed pcm.
# The voice is baked into the image — no downloads, ready in seconds.
# Usage: ./smoke_tts.sh <image>
set -euo pipefail
IMAGE="${1:?usage: smoke_tts.sh <image>}"
OUT="$(mktemp -d /tmp/piper-smoke-XXXXXX)"
NAME="piper-tts-smoke-$$"
PORT="18091"
RUNTIME=(--runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1)

for _ in $(seq 1 90); do
  USED=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
  [ "${USED:-0}" -lt 100000 ] && break
  echo "[smoke] GPU busy (${USED} MiB in use) - sleeping 20s"; sleep 20
done
echo "[smoke] GPU check: ${USED:-0} MiB in use by co-tenants (proceeding)"

docker ps -aq --filter "name=piper-tts-smoke" | xargs -r docker rm -f >/dev/null 2>&1 || true
docker run -d --name "$NAME" "${RUNTIME[@]}" -p "${PORT}:8091" "$IMAGE" >/dev/null
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT

echo "[smoke] waiting for /health"
ok=""
for _ in $(seq 1 60); do
  if curl -sf -m 5 "http://localhost:${PORT}/health" >/dev/null 2>&1; then ok=1; break; fi
  sleep 5
done
[ -n "$ok" ] || { echo "[smoke] FAIL: server never went healthy"; docker logs "$NAME" 2>&1 | tail -8; exit 1; }

echo "[smoke] 1/2 synthesis (wav)"
curl -sf -m 300 -X POST "http://localhost:${PORT}/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello from Jetson Thor.","response_format":"wav"}' -o "$OUT/speech.wav"
S=$(stat -c%s "$OUT/speech.wav"); echo "[smoke] wav bytes: $S"
[ "$S" -gt 30000 ] || { echo "[smoke] FAIL: wav too small"; exit 1; }

echo "[smoke] 2/2 streamed pcm (chunked transfer)"
curl -sf -m 300 -X POST "http://localhost:${PORT}/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"input":"Streaming check. Second sentence flows while the first still plays.","response_format":"pcm"}' \
  -o "$OUT/speech.pcm"
P=$(stat -c%s "$OUT/speech.pcm"); echo "[smoke] pcm bytes: $P"
[ "$P" -gt 30000 ] || { echo "[smoke] FAIL: pcm too small"; exit 1; }
echo "[smoke] PASS (artifacts in $OUT)"
