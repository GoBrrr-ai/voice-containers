#!/usr/bin/env bash
# Build the shared wheels builder (cu132). Byte-identical instruction stream to
# the original build -> replays from BuildKit cache in seconds when available.
set -euo pipefail
cd "$(dirname "$0")"
TAG="${TAG:-narandill/edge-wheels:cu132}"
docker build --target builder -t "$TAG" .
echo "Done: $TAG"
echo "Wheels:"
docker run --rm --runtime=runc --entrypoint ls "$TAG" /wheels
