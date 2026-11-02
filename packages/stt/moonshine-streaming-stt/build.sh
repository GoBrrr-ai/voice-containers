#!/usr/bin/env bash
# Build moonshine-streaming-stt (CPU-only) for Jetson Thor + Orin.
# Usage: ./build.sh [--push]
set -euo pipefail
cd "$(dirname "$0")"
VERSION="${VERSION:-0.2.0}"
REPO="${DOCKER_REPO:-narandill}"

TAGS=(
  -t "${REPO}/moonshine-streaming-stt:${VERSION}"
  -t "${REPO}/moonshine-streaming-stt:latest"
)

echo "Building ${REPO}/moonshine-streaming-stt:${VERSION}"
docker build "${TAGS[@]}" .

if [ "${1:-}" = "--push" ]; then
  docker push "${REPO}/moonshine-streaming-stt:${VERSION}"
  docker push "${REPO}/moonshine-streaming-stt:latest"
fi

echo "Done. Run e.g.:"
echo "  docker run --rm -v /data:/data --entrypoint moonshine-transcribe ${REPO}/moonshine-streaming-stt:${VERSION} /audio/x.wav"
