#!/usr/bin/env bash
# CPU smoke test (no GPU needed): offline transcription of the 16 kHz test wav.
# Usage: ./smoke_stt.sh <image>
set -euo pipefail
IMAGE="${1:?usage: smoke_stt.sh <image>}"
AUDIO="/home/narandill/edge-containers/task/voice-data/smoke_tts_16k.wav"
[ -f "$AUDIO" ] || { echo "missing $AUDIO"; exit 1; }

echo "[smoke] 1/2 transcription (moonshine-transcribe, medium streaming bundle)"
OUT=$(docker run --rm -v "$AUDIO":/audio/x.wav:ro \
  --entrypoint moonshine-transcribe "$IMAGE" /audio/x.wav 2>/dev/null | tail -4)
echo "$OUT" | head -c 400; echo
[ -n "$OUT" ] || { echo "[smoke] FAIL: no transcript"; exit 1; }

echo "[smoke] 2/2 streaming API smoke (create_stream + add_audio)"
docker run --rm -i -v "$AUDIO":/audio/x.wav:ro --entrypoint /opt/venv/bin/python "$IMAGE" - <<'PY'
import os, soundfile as sf
from moonshine_voice import Transcriber
pcm, sr = sf.read("/audio/x.wav", dtype="float32")
if pcm.ndim > 1: pcm = pcm.mean(axis=1)
if sr != 16000:
    import soxr; pcm = soxr.resample(pcm, sr, 16000)
t = Transcriber(model_path=os.environ["MOONSHINE_MODEL_DIR"],
                model_arch=int(os.environ["MOONSHINE_MODEL_ARCH"]))
s = t.create_stream()
s.add_audio(pcm)
print("streaming API OK:", type(s).__name__)
PY

echo "[smoke] PASS"
