#!/usr/bin/env bash
# GPU smoke test (Thor): CUDA EP check + batch + streaming synthesis.
# Waits (never kills) if the GPU is busy. Usage: ./smoke_tts.sh <image>
set -euo pipefail
IMAGE="${1:?usage: smoke_tts.sh <image>}"
OUT="$(mktemp -d /tmp/kokoro-smoke-XXXXXX)"
RUNTIME=(--runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1)

# Co-tenants are fine (125GB Thor); wait only under real memory pressure.
# NEVER kill anyone's processes.
for i in $(seq 1 90); do
  USED=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
  [ "${USED:-0}" -lt 100000 ] && break
  echo "[smoke] GPU busy (${USED} MiB in use) - sleeping 20s"; sleep 20
done
echo "[smoke] GPU check: ${USED:-0} MiB in use by co-tenants (proceeding)"

echo "[smoke] 1/2 streaming synthesis (create_stream) + TTFA"
docker run --rm "${RUNTIME[@]}" -v "$OUT":/data \
  --entrypoint kokoro-stream-demo "$IMAGE" \
  "Hello from Jetson Thor. Kokoro is speaking on the GPU." -o /data/kokoro_stream.wav

echo "[smoke] 2/2 batch synthesis (create) + provider check"
docker run --rm "${RUNTIME[@]}" -v "$OUT":/data \
  --entrypoint kokoro-say "$IMAGE" \
  "Batch synthesis check." -o /data/kokoro_batch.wav

S=$(stat -c%s "$OUT/kokoro_stream.wav"); B=$(stat -c%s "$OUT/kokoro_batch.wav")
echo "[smoke] sizes: stream=$S batch=$B"
[ "$S" -gt 30000 ] && [ "$B" -gt 30000 ] && echo "[smoke] PASS" || { echo "[smoke] FAIL: wav too small"; exit 1; }
