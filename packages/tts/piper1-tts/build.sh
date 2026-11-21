#!/usr/bin/env bash
# Build piper1-tts for Jetson Thor + Orin on JP 7.2.1 (one image, both boards).
# Usage: ./build.sh [--push]
set -euo pipefail
cd "$(dirname "$0")"
VERSION="${VERSION:-0.2.0}"
REPO="${DOCKER_REPO:-narandill}"

TAGS=(
  -t "${REPO}/piper1-tts:${VERSION}"
  -t "${REPO}/piper1-tts:latest"
)

echo "Building ${REPO}/piper1-tts:${VERSION}"
docker build "${TAGS[@]}" .

if [ "${1:-}" = "--push" ]; then
  docker push "${REPO}/piper1-tts:${VERSION}"
  docker push "${REPO}/piper1-tts:latest"
fi

echo "Done. Run e.g.:"
echo "  docker run --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 --rm -p 8091:8091 -v /data/piper:/data ${REPO}/piper1-tts:${VERSION}"
