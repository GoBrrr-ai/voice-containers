#!/usr/bin/env bash
# CPU smoke test (no GPU needed): CLI synthesis + streamed server + TTFA probe.
# Usage: ./smoke_tts.sh <image>
set -euo pipefail
IMAGE="${1:?usage: smoke_tts.sh <image>}"
OUT="$(mktemp -d /tmp/phonon-smoke-XXXXXX)"

echo "[smoke] 1/3 CLI synthesis (ptts-say)"
docker run --rm -v "$OUT":/data --entrypoint ptts-say "$IMAGE" \
  "Hello from Jetson Thor." -o /data/phonon_cli.wav
S=$(stat -c%s "$OUT/phonon_cli.wav"); echo "[smoke] wav: $S bytes"
[ "$S" -gt 50000 ] || { echo "[smoke] FAIL: wav too small"; exit 1; }

echo "[smoke] 2/3 streaming python API (tts.stream) + TTFA"
docker run --rm -i -v "$OUT":/data --entrypoint /opt/venv/bin/python "$IMAGE" - <<'PY'
import time
from ptts import TTS
tts = TTS("/models/phonon-31m-en", lang="en", quant="q8")
t0 = time.perf_counter(); n = 0; first = None; sr = None
with tts.stream("Streaming check on Jetson.") as audio:
    sr = audio.sample_rate
    for pcm in audio:
        if first is None:
            first = time.perf_counter() - t0
            print(f"TTFA: {first * 1000:.0f} ms")
        n += len(pcm)
print(f"streaming OK: {n} samples, {n / sr:.1f}s @ {sr} Hz")
PY

echo "[smoke] 3/3 HTTP streaming server"
CID="phonon-smoke-$$"
docker rm -f "$CID" >/dev/null 2>&1 || true
docker run -d --name "$CID" -p 18880:8880 "$IMAGE" >/dev/null
trap 'docker rm -f "$CID" >/dev/null 2>&1 || true' EXIT
sleep 3
curl -s -m 120 "http://localhost:18880/say?text=Server%20streaming%20check." -o "$OUT/phonon_http.wav"
S=$(stat -c%s "$OUT/phonon_http.wav"); echo "[smoke] http wav: $S bytes"
[ "$S" -gt 50000 ] || { echo "[smoke] FAIL: http wav too small"; exit 1; }
echo "[smoke] PASS (artifacts in $OUT)"
