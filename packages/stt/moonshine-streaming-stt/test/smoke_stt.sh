#!/usr/bin/env bash
# CPU smoke test (no GPU needed): offline transcription of the 16 kHz test wav.
# Usage: ./smoke_stt.sh <image>
set -euo pipefail
IMAGE="${1:?usage: smoke_stt.sh <image>}"
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
AUDIO="$REPO/task/voice-data/smoke_tts_16k.wav"
[ -f "$AUDIO" ] || { echo "missing $AUDIO"; exit 1; }

echo "[smoke] 1/2 transcription (moonshine-transcribe, medium streaming bundle)"
OUT=$(docker run --rm -v "$AUDIO":/audio/x.wav:ro \
  --entrypoint moonshine-transcribe "$IMAGE" /audio/x.wav 2>/dev/null | tail -4)
echo "$OUT" | head -c 400; echo
[ -n "$OUT" ] || { echo "[smoke] FAIL: no transcript"; exit 1; }

echo "[smoke] 2/2 streaming API smoke (create_stream + add_audio)"
docker run --rm -i -v "$AUDIO":/audio/x.wav:ro --entrypoint /opt/venv/bin/python "$IMAGE" - <<'PY'
import os, soundfile as sf
from moonshine_voice import ModelArch, Transcriber
pcm, sr = sf.read("/audio/x.wav", dtype="float32")
if pcm.ndim > 1: pcm = pcm.mean(axis=1)
if sr != 16000:
    import soxr; pcm = soxr.resample(pcm, sr, 16000)
t = Transcriber(model_path=os.environ["MOONSHINE_MODEL_DIR"],
                model_arch=ModelArch.MEDIUM_STREAMING)
s = t.create_stream()
s.start()
s.add_audio(pcm.tolist(), 16000)
out = s.stop()
text = " ".join(l.text for l in out.lines).strip()
assert text, "empty streaming transcript"
print("streaming API OK:", text[:80])
PY

echo "[smoke] PASS"
