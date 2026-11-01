#!/usr/bin/env bash
# GPU smoke test (Thor): wyoming stt round trip (retries until the server is
# ready — first boot downloads the model into the /data volume).
set -euo pipefail
IMAGE="${1:?usage: smoke_whisper.sh <image>}"
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
AUDIO="$REPO/task/voice-data/smoke_tts_16k.wav"
[ -f "$AUDIO" ] || { echo "missing $AUDIO"; exit 1; }
NAME_SUFFIX="$$"
NAME="wyoming-faster-whisper-smoke-$$"
RUNTIME=(--runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1)

for i in $(seq 1 90); do
  USED=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
  [ "${USED:-0}" -lt 100000 ] && break
  echo "[smoke] GPU busy (${USED} MiB in use) - sleeping 20s"; sleep 20
done
echo "[smoke] GPU check: ${USED:-0} MiB in use by co-tenants (proceeding)"

docker ps -aq --filter "name=wyoming-faster-whisper-smoke" | xargs -r docker rm -f >/dev/null 2>&1 || true
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" "${RUNTIME[@]}" -e WHISPER_ZEROCONF_NAME=wfws-$NAME_SUFFIX \
  -v /home/narandill/edge-containers/task/voice-data:/data "$IMAGE" >/dev/null
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT

echo "[smoke] stt round trip (retries until ready; first boot downloads the model)"
ok=""
for i in $(seq 1 60); do
  if OUT=$(timeout 300 docker exec -i "$NAME" /opt/venv/bin/python - \
      < "$(dirname "$0")/wyoming_roundtrip.py" stt /data/smoke_tts_16k.wav 2>/dev/null) && [ -n "$OUT" ]; then
    ok="$OUT"; break
  fi
  sleep 5
done
[ -n "$ok" ] || { echo "[smoke] FAIL: round trip never succeeded"; docker logs "$NAME" 2>&1 | tail -8; exit 1; }
echo "[smoke] stt result: $ok"
echo "[smoke] PASS"
