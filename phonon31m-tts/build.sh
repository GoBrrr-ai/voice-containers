#!/usr/bin/env bash
# Build phonon31m-tts (CPU-only) for Jetson Thor + Orin (no CUDA needed).
# Usage: ./build.sh [--push]
set -euo pipefail
cd "$(dirname "$0")"
VERSION="${VERSION:-0.1.0}"
REPO="${DOCKER_REPO:-narandill}"

TAGS=(
  -t "${REPO}/phonon31m-tts:${VERSION}"
  -t "${REPO}/phonon31m-tts:latest"
)

echo "Building ${REPO}/phonon31m-tts:${VERSION}"
docker build "${TAGS[@]}" .

if [ "${1:-}" = "--push" ]; then
  docker push "${REPO}/phonon31m-tts:${VERSION}"
  docker push "${REPO}/phonon31m-tts:latest"
fi

echo "Done. Run e.g.:"
echo "  docker run --rm -p 8880:8880 ${REPO}/phonon31m-tts:${VERSION}"
echo "  # or CLI: docker run --rm -v /data:/data --entrypoint ptts-say ${REPO}/phonon31m-tts:${VERSION} \"Hello\" -o /data/x.wav"
