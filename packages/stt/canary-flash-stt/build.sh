#!/usr/bin/env bash
# Build canary-flash-stt (CPU-only) for Jetson Thor + Orin.
# Usage: ./build.sh [--push]
set -euo pipefail
cd "$(dirname "$0")"
VERSION="${VERSION:-0.2.0}"
REPO="${DOCKER_REPO:-narandill}"

TAGS=(
  -t "${REPO}/canary-flash-stt:${VERSION}"
  -t "${REPO}/canary-flash-stt:latest"
)

echo "Building ${REPO}/canary-flash-stt:${VERSION}"
docker build "${TAGS[@]}" .

if [ "${1:-}" = "--push" ]; then
  docker push "${REPO}/canary-flash-stt:${VERSION}"
  docker push "${REPO}/canary-flash-stt:latest"
fi

echo "Done. Run e.g.:"
echo "  docker run --rm -v /data:/data --entrypoint canary-transcribe ${REPO}/canary-flash-stt:${VERSION} /audio/x.wav"
