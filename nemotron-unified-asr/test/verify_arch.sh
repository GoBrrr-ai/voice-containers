#!/usr/bin/env bash
# Verify SASS/PTX coverage of the GGML CUDA artifacts in a built image.
# Copies artifacts out via docker cp, runs cuobjdump from the devel image.
# Usage: ./verify_arch.sh <image>
set -euo pipefail

IMAGE="${1:?usage: verify_arch.sh <image>}"
DEVEL="nvidia/cuda:13.2.1-devel-ubuntu24.04"
WORK="$(mktemp -d /tmp/verify-arch-XXXXXX)"
CID="verify-arch-$$"
trap 'docker rm -f "$CID" >/dev/null 2>&1 || true; rm -rf "$WORK"' EXIT

docker create --name "$CID" "$IMAGE" >/dev/null
mkdir -p "$WORK"
docker cp "$CID":/opt/parakeet-dist/lib/. "$WORK/lib" 2>/dev/null || true
docker cp "$CID":/opt/parakeet-dist/bin/parakeet-cli "$WORK/parakeet-cli" 2>/dev/null || true

: > "$WORK/files.txt"
for f in "$WORK"/lib/libggml-cuda.so*; do
  [ -f "$f" ] && echo "REQUIRE /verify/${f#"$WORK"/}" >> "$WORK/files.txt"
done
for f in "$WORK"/lib/libggml.so* "$WORK"/lib/libggml-base.so* "$WORK"/lib/libggml-cpu.so* "$WORK"/parakeet-cli; do
  [ -f "$f" ] && echo "INFO /verify/${f#"$WORK"/}" >> "$WORK/files.txt"
done
if [ ! -s "$WORK/files.txt" ]; then echo "no CUDA artifacts found in $IMAGE"; exit 1; fi

docker run --rm --runtime=runc -i -v "$WORK":/verify:ro --entrypoint bash "$DEVEL" < "$(dirname "$0")/_verify_arch_inner.sh"
