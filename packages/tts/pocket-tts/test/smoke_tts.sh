#!/usr/bin/env bash
# CPU smoke test (no GPU needed): CLI generation + python library probe.
# Usage: ./smoke_tts.sh <image>
set -euo pipefail
IMAGE="${1:?usage: smoke_tts.sh <image>}"
OUT="$(mktemp -d /tmp/pocket-smoke-XXXXXX)"

echo "[smoke] 1/2 CLI generation (pocket-tts generate)"
docker run --rm -v "$OUT":/data --entrypoint pocket-tts "$IMAGE" \
  generate --text "Hello from Jetson Thor." --output /data/pocket_cli.wav
S=$(stat -c%s "$OUT/pocket_cli.wav"); echo "[smoke] wav: $S bytes"
[ "$S" -gt 50000 ] || { echo "[smoke] FAIL: wav too small"; exit 1; }

echo "[smoke] 2/2 python library (TTSModel.load_model + generate_audio)"
docker run --rm -i --entrypoint /opt/venv/bin/python "$IMAGE" - <<'PY'
import scipy.io.wavfile
from pocket_tts import TTSModel

m = TTSModel.load_model()
voice_state = m.get_state_for_audio_prompt("alba")
audio = m.generate_audio(voice_state, "Streaming check on Jetson.")
scipy.io.wavfile.write("/tmp/check.wav", m.sample_rate, audio.numpy())
print(f"python API OK: {len(audio)} samples @ {m.sample_rate} Hz")
PY

echo "[smoke] PASS (artifacts in $OUT)"
