#!/usr/bin/env bash
# Build pocket-tts (CPU-only) for Jetson Thor + Orin.
# Usage: ./build.sh [--push]
set -euo pipefail
cd "$(dirname "$0")"
VERSION="${VERSION:-0.1.0}"
REPO="${DOCKER_REPO:-narandill}"

TAGS=(
  -t "${REPO}/pocket-tts:${VERSION}"
  -t "${REPO}/pocket-tts:latest"
)

echo "Building ${REPO}/pocket-tts:${VERSION}"
docker build "${TAGS[@]}" .

if [ "${1:-}" = "--push" ]; then
  docker push "${REPO}/pocket-tts:${VERSION}"
  docker push "${REPO}/pocket-tts:latest"
fi

echo "Done. Run e.g.:"
echo "  docker run --rm -v /data:/data --entrypoint pocket-tts ${REPO}/pocket-tts:${VERSION} generate --text \"Hello\" --output-path /data/x.wav"
echo "  docker run --rm -p 8000:8000 ${REPO}/pocket-tts:${VERSION} serve"
