#!/usr/bin/env bash
# wyoming-piper TTS service (fork env-var contract)
set -euo pipefail

flags=()
# voice is BAKED into the image (/data/models/piper) — default OFF so boot
# never phones HF; set true to refresh the index/voices from HF
[ "${PIPER_UPDATE_VOICES:-false}" = "true" ] && flags+=("--update-voices")
[ "${PIPER_DEBUG:-}" = "true" ] && flags+=("--debug")
[ "${PIPER_USE_CUDA:-true}" = "true" ] && flags+=("--use-cuda")
[ -n "${PIPER_SPEAKER:-}" ] && flags+=("--speaker" "${PIPER_SPEAKER}")
[ "${PIPER_OFFLINE:-}" = "true" ] && flags+=("--local-files-only")

exec python3 -m wyoming_piper \
    --uri "tcp://0.0.0.0:${PIPER_PORT:-10200}" \
    --voice "${PIPER_VOICE:-en_US-libritts_r-medium}" \
    --length-scale "${PIPER_LENGTH_SCALE:-1.0}" \
    --noise-scale "${PIPER_NOISE_SCALE:-0.667}" \
    --noise-w "${PIPER_NOISE_W:-0.333}" \
    --data-dir "${PIPER_CACHE:-/data/models/piper}" \
    --download-dir "${PIPER_CACHE:-/data/models/piper}" \
    --zeroconf "${PIPER_ZEROCONF_NAME:-wyoming-piper-tts}" \
    "${flags[@]}"
