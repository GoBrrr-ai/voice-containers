#!/usr/bin/env bash
# Build parakeet-tdt-stt (CPU-only) for Jetson Thor + Orin.
# Usage: ./build.sh [--push]
set -euo pipefail
cd "$(dirname "$0")"
VERSION="${VERSION:-0.2.0}"
REPO="${DOCKER_REPO:-narandill}"

TAGS=(
  -t "${REPO}/parakeet-tdt-stt:${VERSION}"
  -t "${REPO}/parakeet-tdt-stt:latest"
)

echo "Building ${REPO}/parakeet-tdt-stt:${VERSION}"
docker build "${TAGS[@]}" .

if [ "${1:-}" = "--push" ]; then
  docker push "${REPO}/parakeet-tdt-stt:${VERSION}"
  docker push "${REPO}/parakeet-tdt-stt:latest"
fi

echo "Done. Run e.g.:"
echo "  docker run --rm -v /data:/data --entrypoint parakeet-tdt-transcribe ${REPO}/parakeet-tdt-stt:${VERSION} /audio/x.wav"
