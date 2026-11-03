#!/usr/bin/env bash
# Build <image> for Jetson Thor + Orin on JP 7.2.1 (one image, both boards).
# Usage: ./build.sh [--push]
set -euo pipefail
cd "$(dirname "$0")"
VERSION="${VERSION:-0.1.0}"
REPO="${DOCKER_REPO:-narandill}"
NAME="$(basename "$(pwd)")"

TAGS=(
  -t "${REPO}/${NAME}:${VERSION}"
  -t "${REPO}/${NAME}:latest"
)

echo "Building ${REPO}/${NAME}:${VERSION}"
docker build "${TAGS[@]}" .

if [ "${1:-}" = "--push" ]; then
  docker push "${REPO}/${NAME}:${VERSION}"
  docker push "${REPO}/${NAME}:latest"
fi

echo "Done. Run e.g.:"
echo "  docker run --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 --rm -it -p 8080:8080 ${REPO}/${NAME}:${VERSION}"
echo "  # CLI:  docker run --rm --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 -v /audio:/audio:ro --entrypoint stt-transcribe ${REPO}/${NAME}:${VERSION} /audio/x.wav"
