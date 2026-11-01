#!/usr/bin/env bash
# Build nemotron-streaming-stt (CPU-only) for Jetson Thor + Orin.
# Usage: ./build.sh [--push]
set -euo pipefail
cd "$(dirname "$0")"
VERSION="${VERSION:-0.1.0}"
REPO="${DOCKER_REPO:-narandill}"

TAGS=(
  -t "${REPO}/nemotron-streaming-stt:${VERSION}"
  -t "${REPO}/nemotron-streaming-stt:latest"
)

echo "Building ${REPO}/nemotron-streaming-stt:${VERSION}"
docker build "${TAGS[@]}" .

if [ "${1:-}" = "--push" ]; then
  docker push "${REPO}/nemotron-streaming-stt:${VERSION}"
  docker push "${REPO}/nemotron-streaming-stt:latest"
fi

echo "Done. Run e.g.:"
echo "  docker run --rm -v /data:/data --entrypoint nemotron-stream-transcribe ${REPO}/nemotron-streaming-stt:${VERSION} /audio/x.wav"
