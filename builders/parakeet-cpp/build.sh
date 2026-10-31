#!/usr/bin/env bash
# Build the shared parakeet.cpp dist builder (sm_87+sm_110a cubins + PTX).
set -euo pipefail
cd "$(dirname "$0")"
TAG="${TAG:-narandill/edge-builder:parakeet-cpp}"
REF="${PARAKEET_CPP_REF:-2eac2e77d9c7ef8899f7a86a46e693ac5ac1a5cd}"
docker build --build-arg PARAKEET_CPP_REF="$REF" -t "$TAG" .
echo "Done: $TAG"
