#!/usr/bin/env bash
# wyoming-faster-whisper STT service (fork env-var contract)
set -euo pipefail

flags=()
[ "${WHISPER_DEBUG:-}" = "true" ] && flags+=("--debug")
[ "${WHISPER_OFFLINE:-}" = "true" ] && flags+=("--local-files-only")
[ -n "${WHISPER_INITIAL_PROMPT:-}" ] && flags+=("--initial-prompt" "${WHISPER_INITIAL_PROMPT}")
[ -n "${WHISPER_VAD_FILTER:-}" ] && flags+=("--vad-filter")
[ -n "${WHISPER_HASS_TOKEN:-}" ] && flags+=("--hass-token" "${WHISPER_HASS_TOKEN}")

exec python3 -m wyoming_faster_whisper \
    --uri "tcp://0.0.0.0:${WHISPER_PORT:-10300}" \
    --model "${WHISPER_MODEL:-tiny-int8}" \
    --beam-size "${WHISPER_BEAM_SIZE:-1}" \
    --compute-type "${WHISPER_COMPUTE_TYPE:-default}" \
    --language "${WHISPER_LANGUAGE:-en}" \
    --zeroconf "${WHISPER_ZEROCONF_NAME:-wyoming-faster-whisper}" \
    --device "${WHISPER_DEVICE:-cuda}" \
    --data-dir "${WHISPER_DATA_DIR:-/data/models/faster-whisper}" \
    --download-dir "${WHISPER_DATA_DIR:-/data/models/faster-whisper}" \
    --stt-library "${WHISPER_STT_LIBRARY:-faster-whisper}" \
    "${flags[@]}"
