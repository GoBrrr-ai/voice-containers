#!/usr/bin/env bash
# Build paradee-tts for Jetson Thor + Orin on JP 7.2.1 (one image, both boards).
# Usage: ./build.sh [--push]
set -euo pipefail
cd "$(dirname "$0")"
VERSION="${VERSION:-0.1.0}"
REPO="${DOCKER_REPO:-narandill}"

TAGS=(
  -t "${REPO}/paradee-tts:${VERSION}"
  -t "${REPO}/paradee-tts-thor:${VERSION}"
  -t "${REPO}/paradee-tts-orin:${VERSION}"
  -t "${REPO}/paradee-tts:latest"
)

echo "Building ${REPO}/paradee-tts:${VERSION}"
docker build "${TAGS[@]}" .

if [ "${1:-}" = "--push" ]; then
  for tag in "${REPO}/paradee-tts:${VERSION}" "${REPO}/paradee-tts-thor:${VERSION}" \
             "${REPO}/paradee-tts-orin:${VERSION}" "${REPO}/paradee-tts:latest"; do
    docker push "$tag"
  done
fi

echo "Done. Run e.g.:"
echo "  docker run --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 --rm -v /data:/data ${REPO}/paradee-tts-thor:${VERSION} paradee-say \"Hello\" -o /data/p.wav"
