#!/usr/bin/env bash
# Build wyoming-faster-whisper for Jetson Thor + Orin on JP 7.2.1 (one image, both boards).
# Usage: ./build.sh [--push]
set -euo pipefail
cd "$(dirname "$0")"
VERSION="${VERSION:-0.1.0}"
REPO="${DOCKER_REPO:-futureproofhomes}"

TAGS=(
  -t "${REPO}/wyoming-faster-whisper:${VERSION}"
  -t "${REPO}/wyoming-faster-whisper-thor:${VERSION}"
  -t "${REPO}/wyoming-faster-whisper-orin:${VERSION}"
  -t "${REPO}/wyoming-faster-whisper:latest"
)

echo "Building ${REPO}/wyoming-faster-whisper:${VERSION}"
docker build "${TAGS[@]}" .

if [ "${1:-}" = "--push" ]; then
  for tag in "${REPO}/wyoming-faster-whisper:${VERSION}" "${REPO}/wyoming-faster-whisper-thor:${VERSION}" \
             "${REPO}/wyoming-faster-whisper-orin:${VERSION}" "${REPO}/wyoming-faster-whisper:latest"; do
    docker push "$tag"
  done
fi

echo "Done. Run e.g.:"
echo "  docker run --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 --rm -p 10300:10300 -v /data/whisper:/data ${REPO}/wyoming-faster-whisper-thor:${VERSION}"
