#!/usr/bin/env bash
# GPU smoke test (Thor): wyoming tts round trip (retries until the server is
# ready — first boot downloads the model into the /data volume).
set -euo pipefail
IMAGE="${1:?usage: smoke_piper.sh <image>}"
AUDIO="/home/narandill/voice-containers/task/voice-data/smoke_tts_16k.wav"
[ -f "$AUDIO" ] || { echo "missing $AUDIO"; exit 1; }
NAME_SUFFIX="$$"
NAME="wyoming-piper1-tts-smoke-$$"
RUNTIME=(--runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1)

for _ in $(seq 1 90); do
  USED=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
  [ "${USED:-0}" -lt 100000 ] && break
  echo "[smoke] GPU busy (${USED} MiB in use) - sleeping 20s"; sleep 20
done
echo "[smoke] GPU check: ${USED:-0} MiB in use by co-tenants (proceeding)"

docker ps -aq --filter "name=wyoming-piper1-tts-smoke" | xargs -r docker rm -f >/dev/null 2>&1 || true
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" "${RUNTIME[@]}" -e PIPER_ZEROCONF_NAME=wpts-$NAME_SUFFIX \
  -v /home/narandill/voice-containers/task/voice-data:/data "$IMAGE" >/dev/null
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT

echo "[smoke] tts round trip (retries until ready; first boot downloads the model)"
ok=""
for _ in $(seq 1 60); do
  if OUT=$(timeout 300 docker exec -i "$NAME" /opt/venv/bin/python - \
      < "$(dirname "$0")/wyoming_roundtrip.py" tts /tmp/piper_smoke.wav "Hello from Jetson Thor." 2>/dev/null) && [ -n "$OUT" ]; then
    ok="$OUT"; break
  fi
  sleep 5
done
[ -n "$ok" ] || { echo "[smoke] FAIL: round trip never succeeded"; docker logs "$NAME" 2>&1 | tail -8; exit 1; }
echo "[smoke] tts result: $ok"
SZ=$(docker exec "$NAME" stat -c%s /tmp/piper_smoke.wav 2>/dev/null || echo 0)
echo "[smoke] wav bytes: $SZ"
[ "$SZ" -gt 30000 ] || { echo "[smoke] FAIL: wav too small"; exit 1; }
echo "[smoke] PASS"
