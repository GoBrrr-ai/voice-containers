#!/usr/bin/env bash
# Build shared builders + every engine image. Fails fast; layer-cached after first run.
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECTS="${PROJECTS:-$(tr '\n' ' ' < scripts/projects.txt)}"
echo "== builders (cached)"
(cd builders/parakeet-cpp && ./build.sh)
(cd builders/wheels-cu132 && ./build.sh)
for p in $PROJECTS; do
  echo "== build $p"
  (cd "$p" && ./build.sh)
done
echo "== ALL BUILT"
