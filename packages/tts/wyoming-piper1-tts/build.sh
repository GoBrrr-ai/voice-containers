#!/usr/bin/env bash
# Build wyoming-piper1-tts for Jetson Thor + Orin on JP 7.2.1 (one image, both boards).
# Usage: ./build.sh [--push]
set -euo pipefail
cd "$(dirname "$0")"
VERSION="${VERSION:-0.1.1}"
REPO="${DOCKER_REPO:-narandill}"

TAGS=(
  -t "${REPO}/wyoming-piper1-tts:${VERSION}"
  -t "${REPO}/wyoming-piper1-tts:latest"
)

echo "Building ${REPO}/wyoming-piper1-tts:${VERSION}"
docker build "${TAGS[@]}" .

if [ "${1:-}" = "--push" ]; then
  docker push "${REPO}/wyoming-piper1-tts:${VERSION}"
  docker push "${REPO}/wyoming-piper1-tts:latest"
fi

echo "Done. Run e.g.:"
echo "  docker run --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 --rm -p 10200:10200 -v /data/piper:/data ${REPO}/wyoming-piper1-tts:${VERSION}"
