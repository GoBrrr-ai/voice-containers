#!/usr/bin/env bash
# Build shared builders + every engine image. Fails fast; layer-cached after first run.
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECTS="${PROJECTS:-llama-cpp wyoming-faster-whisper wyoming-piper1-tts nemotron-unified-asr parakeet-redux-stt phonon2-stt kokoro-tts phonon31m-tts paradee-tts}"
echo "== builders (cached)"
(cd builders/parakeet-cpp && ./build.sh)
(cd builders/wheels-cu132 && ./build.sh)
for p in $PROJECTS; do
  echo "== build $p"
  (cd "$p" && ./build.sh)
done
echo "== ALL BUILT"
