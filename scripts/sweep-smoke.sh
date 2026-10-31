#!/usr/bin/env bash
# Verify arch (cuobjdump) + run the GPU smoke suite for every project.
# Waits for free VRAM when co-tenants run; NEVER kills other processes.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="${DOCKER_REPO:-narandill}"
SMOKE="${SMOKE:-1}"   # SMOKE=0 → arch verification only
PROJECTS="${PROJECTS:-llama-cpp wyoming-faster-whisper wyoming-piper1-tts nemotron-unified-asr parakeet-redux-stt phonon2-stt kokoro-tts phonon31m-tts paradee-tts}"
fail=0
declare -A VER=(
  [llama-cpp]=0.8.0 [wyoming-faster-whisper]=0.1.0 [wyoming-piper1-tts]=0.1.0
  [nemotron-unified-asr]=0.1.0 [parakeet-redux-stt]=0.1.0 [phonon2-stt]=0.1.0
  [kokoro-tts]=0.1.0 [phonon31m-tts]=0.1.0 [paradee-tts]=0.1.0
)
for p in $PROJECTS; do
  v="${VER[$p]:-0.1.0}"; img="$REPO/$p:$v"
  echo "== verify $p ($img)"
  if [ -x "$p/test/verify_arch.sh" ]; then "$p/test/verify_arch.sh" "$img" || fail=1
  else echo "  (no arch verification — CPU-only or none needed)"; fi
  if [ "$SMOKE" = "1" ] && ls "$p"/test/smoke_*.sh >/dev/null 2>&1; then
    echo "== smoke $p"
    (cd "$p" && ./test/smoke_*.sh "$img") || fail=1
  fi
done
if [ "$fail" = "0" ]; then echo "SWEEP PASS"; else echo "SWEEP FAIL"; exit 1; fi
