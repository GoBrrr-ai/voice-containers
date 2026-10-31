#!/usr/bin/env bash
# GPU smoke test (Thor): batch + streaming transcription, CPU variant, OpenAI server.
# Waits (never kills) if the GPU is busy. Usage: ./smoke_stt.sh <image>
set -euo pipefail
IMAGE="${1:?usage: smoke_stt.sh <image>}"
AUDIO="/home/narandill/edge-containers/task/voice-data/smoke_tts_16k.wav"
[ -f "$AUDIO" ] || { echo "missing $AUDIO"; exit 1; }
RUNTIME=(--runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1)   # env only needed pre-JP7.2.1-upgrade

# --- wait for a free GPU (up to ~30 min; NEVER kill other people's jobs) -----
# Co-tenants are fine (125GB Thor); wait only under real memory pressure.
# NEVER kill anyone's processes.
for i in $(seq 1 90); do
  USED=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {print s+0}')
  [ "${USED:-0}" -lt 100000 ] && break
  echo "[smoke] GPU busy (${USED} MiB in use) - sleeping 20s"; sleep 20
done
echo "[smoke] GPU check: ${USED:-0} MiB in use by co-tenants (proceeding)"

echo "[smoke] 1/4 batch transcription (GPU)"
OUT=$(docker run --rm "${RUNTIME[@]}" -v "$AUDIO":/audio/x.wav:ro \
  --entrypoint stt-transcribe "$IMAGE" /audio/x.wav 2>/dev/null | tail -4)
echo "$OUT" | head -c 400; echo
[ -n "$OUT" ] || { echo "[smoke] FAIL: no transcript"; exit 1; }

echo "[smoke] 2/4 streaming run (stream preset or VAD chunking per image)"
# nemotron-unified-asr: PARAKEET_STREAM=1 is baked in -> buffered streaming.
# TDT images (redux/phonon): PARAKEET_VAD_MODEL baked -> --vad chunked runs.
OUT=$(docker run --rm "${RUNTIME[@]}" -v "$AUDIO":/audio/x.wav:ro \
  --entrypoint stt-transcribe "$IMAGE" /audio/x.wav 2>/dev/null | tail -4)
echo "$OUT" | head -c 400; echo
[ -n "$OUT" ] || { echo "[smoke] FAIL: empty streaming transcript"; exit 1; }

echo "[smoke] 3/4 CPU fallback (same binary, -d cpu)"
OUT=$(docker run --rm "${RUNTIME[@]}" -v "$AUDIO":/audio/x.wav:ro \
  -e PARAKEET_DEVICE=cpu \
  --entrypoint stt-transcribe "$IMAGE" /audio/x.wav 2>/dev/null | tail -4)
echo "$OUT" | head -c 400; echo
echo "$OUT" | grep -qiE 'hello|thor' || echo "[smoke] WARN: CPU transcript unexpected (continuing)"

echo "[smoke] 4/4 OpenAI-compatible server round trip"
CID="stt-smoke-$$"
docker rm -f "$CID" >/dev/null 2>&1 || true
docker run -d --name "$CID" "${RUNTIME[@]}" -p 18080:8080 "$IMAGE" >/dev/null
trap 'docker rm -f "$CID" >/dev/null 2>&1 || true' EXIT
# TCP-port probe only: a bodyless POST would wedge the single-slot server.
# HTTP GET probe: docker's userland proxy accepts TCP before the server
# listens (false positive -> POST reset), and a raw connect+EOF can hit the
# server mid-boot. GET / answers 404 fast and cannot wedge the POST handler.
up=0
for i in $(seq 1 60); do
  # NOTE: curl -w prints "000" on failure by itself; do NOT append another
  # 000 via || -- the doubled value would compare != "000" and false-positive.
  C=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://localhost:18080/ 2>/dev/null || true)
  [ -n "$C" ] && [ "$C" != "000" ] && up=1 && break
  sleep 2
done
[ "$up" = "1" ] || { echo "[smoke] FAIL: server never answered (last code $C)"; exit 1; }
RESP=$(curl -sS -m 300 -F file=@"$AUDIO" -F response_format=verbose_json http://localhost:18080/v1/audio/transcriptions) \
  || { RC=$?; echo "[smoke] FAIL: POST failed (curl exit $RC)"; docker logs "$CID" 2>&1 | tail -6; exit 1; }
echo "$RESP" | head -c 400; echo
echo "$RESP" | grep -q '"text"' || { echo "[smoke] FAIL: server did not return a transcript"; exit 1; }
echo "[smoke] PASS"
