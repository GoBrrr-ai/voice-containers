#!/bin/sh
# OpenAI-compatible ASR server (POST /v1/audio/transcriptions, WAV uploads).
# One model, one request at a time — the voice-api surface is docs/voice-api.md.
# Env: PARAKEET_MODEL (required), PARAKEET_HOST, PARAKEET_PORT,
#      PARAKEET_THREADS, PARAKEET_CONCURRENCY, PARAKEET_EXTRA_ARGS.
set -eu
MODEL="${PARAKEET_MODEL:?PARAKEET_MODEL is required (baked default should apply)}"
set -- /opt/parakeet-dist/bin/parakeet-server \
    --model "$MODEL" \
    --host "${PARAKEET_HOST:-0.0.0.0}" \
    --port "${PARAKEET_PORT:-8080}" \
    ${PARAKEET_THREADS:+--threads "$PARAKEET_THREADS"} \
    ${PARAKEET_CONCURRENCY:+--concurrency "$PARAKEET_CONCURRENCY"} \
    ${PARAKEET_EXTRA_ARGS:-} "$@"
exec "$@"
