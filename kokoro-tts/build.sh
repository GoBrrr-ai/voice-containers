#!/usr/bin/env bash
# Build kokoro-tts for Jetson Thor + Orin on JP 7.2.1 (one image, both boards).
# Usage: ./build.sh [--push]
set -euo pipefail
cd "$(dirname "$0")"
VERSION="${VERSION:-0.1.0}"
REPO="${DOCKER_REPO:-narandill}"

TAGS=(
  -t "${REPO}/kokoro-tts:${VERSION}"
  -t "${REPO}/kokoro-tts:latest"
)

echo "Building ${REPO}/kokoro-tts:${VERSION}"
docker build "${TAGS[@]}" .

if [ "${1:-}" = "--push" ]; then
  for tag in "${REPO}/kokoro-tts:${VERSION}" "${REPO}/kokoro-tts:${VERSION}" \
             "${REPO}/kokoro-tts:${VERSION}" "${REPO}/kokoro-tts:latest"; do
    docker push "$tag"
  done
fi

echo "Done. Run e.g.:"
echo "  docker run --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 --rm -v /data:/data ${REPO}/kokoro-tts:${VERSION} kokoro-say \"Hello\" -o /data/k.wav"
