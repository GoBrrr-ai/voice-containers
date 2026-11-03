#!/usr/bin/env bash
# Verify arch (cuobjdump) + run the GPU smoke suite for every project.
# Waits for free VRAM when co-tenants run; NEVER kills other processes.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="${DOCKER_REPO:-narandill}"
SMOKE="${SMOKE:-1}"   # SMOKE=0 → arch verification only
PROJECTS="${PROJECTS:-packages/llm/llama-cpp packages/stt/wyoming-faster-whisper packages/stt/nemotron-unified-asr packages/stt/parakeet-redux-stt packages/stt/phonon2-stt packages/stt/moonshine-streaming-stt packages/stt/canary-flash-stt packages/stt/parakeet-tdt-stt packages/stt/parakeet-tdt-cpp-stt packages/stt/nemotron-streaming-stt packages/tts/wyoming-piper1-tts packages/tts/kokoro-tts packages/tts/phonon31m-tts packages/tts/paradee-tts packages/tts/pocket-tts}"
fail=0
for p in $PROJECTS; do
  # Single source of truth: the VERSION default in each build.sh (the inner
  # most-specific `:-X` — llama-cpp nests it inside LLAMA_CPP_VERSION).
  v="$(grep -oE 'VERSION:-[0-9.]+' "$p/build.sh" 2>/dev/null | head -1 | sed 's/.*VERSION:-//')"
  v="${v:-0.1.0}"; img="$REPO/$p:$v"
  echo "== verify $p ($img)"
  if [ -x "$p/test/verify_arch.sh" ]; then "$p/test/verify_arch.sh" "$img" || fail=1
  else echo "  (no arch verification — CPU-only or none needed)"; fi
  if [ "$SMOKE" = "1" ]; then
    for s in "$p"/test/smoke_*.sh; do
      [ -e "$s" ] || { echo "  (no smoke scripts)"; break; }
      echo "== smoke $p ($(basename "$s"))"
      (cd "$p" && ./test/"$(basename "$s")" "$img") || fail=1
    done
  fi
done
if [ "$fail" = "0" ]; then echo "SWEEP PASS"; else echo "SWEEP FAIL"; exit 1; fi
