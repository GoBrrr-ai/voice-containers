#!/usr/bin/env bash
# Build shared builders + every engine image. Fails fast; layer-cached after first run.
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECTS="${PROJECTS:-packages/llm/llama-cpp packages/stt/wyoming-faster-whisper packages/stt/nemotron-unified-asr packages/stt/parakeet-redux-stt packages/stt/phonon2-stt packages/stt/moonshine-streaming-stt packages/stt/canary-flash-stt packages/stt/parakeet-tdt-stt packages/stt/parakeet-tdt-cpp-stt packages/stt/parakeet-tdt-ort-stt packages/stt/nemotron-streaming-stt packages/tts/wyoming-piper1-tts packages/tts/kokoro-tts packages/tts/phonon31m-tts packages/tts/paradee-tts packages/tts/pocket-tts}"
echo "== builders (cached)"
(cd builders/parakeet-cpp && ./build.sh)
(cd builders/wheels-cu132 && ./build.sh)
for p in $PROJECTS; do
  echo "== build $p"
  (cd "$p" && ./build.sh)
done
echo "== ALL BUILT"
