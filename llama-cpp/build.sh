#!/usr/bin/env bash
# Build llama.cpp image for Jetson Thor + Orin on JP 7.2.1 (one image, both boards).
# Usage:
#   ./build.sh                     # build + tag
#   ./build.sh --push              # build + tag + docker push
#   VERSION=0.8.0 ./build.sh
set -euo pipefail

cd "$(dirname "$0")"

VERSION="${LLAMA_CPP_VERSION:-0.8.0}"
REF="${LLAMA_CPP_REF:-v${VERSION}}"
PYVER="${LLAMA_CPP_PYTHON_VERSION:-0.3.36}"
CUDA_ARCHS="${CUDA_ARCHS:-87-real;110a-real;110-virtual}"
REPO="${DOCKER_REPO:-narandill}"

TAGS=(
  -t "${REPO}/llama-cpp:${VERSION}"
  -t "${REPO}/llama-cpp-thor:${VERSION}"
  -t "${REPO}/llama-cpp-orin:${VERSION}"
  -t "${REPO}/llama-cpp:latest"
)
[ "${1:-}" = "--push" ] && TAGS+=("${EXTRA_PUSH_TAGS:-}")

echo "Building ${REPO}/llama-cpp:${VERSION}"
echo "  ref=${REF} python=${PYVER} archs=${CUDA_ARCHS}"

docker build "${TAGS[@]}" \
  --build-arg LLAMA_CPP_REF="${REF}" \
  --build-arg LLAMA_CPP_PYTHON_VERSION="${PYVER}" \
  --build-arg CUDA_ARCHS="${CUDA_ARCHS}" \
  .

if [ "${1:-}" = "--push" ]; then
  for tag in "${REPO}/llama-cpp:${VERSION}" "${REPO}/llama-cpp-thor:${VERSION}" \
             "${REPO}/llama-cpp-orin:${VERSION}" "${REPO}/llama-cpp:latest"; do
    docker push "$tag"
  done
fi

echo "Done. Run e.g.:"
echo "  docker run --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 --rm -it \\"
echo "    -p 8080:8080 -v /data/models:/data ${REPO}/llama-cpp-thor:${VERSION} \\"
echo "    llama-server -m /data/model.gguf -ngl 999 --host 0.0.0.0"
