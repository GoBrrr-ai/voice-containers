#!/usr/bin/env bash
# Verify arch (cuobjdump) + run the GPU smoke suite for every project.
# Waits for free VRAM when co-tenants run; NEVER kills other processes.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="${DOCKER_REPO:-narandill}"
SMOKE="${SMOKE:-1}"   # SMOKE=0 → arch verification only
PROJECTS="${PROJECTS:-$(tr '\n' ' ' < scripts/projects.txt)}"
fail=0
for p in $PROJECTS; do
  # Single source of truth: the VERSION default in each build.sh (the inner
  # most-specific `:-X` — llama-cpp nests it inside LLAMA_CPP_VERSION).
  v="$(grep -oE 'VERSION:-[0-9.]+' "$p/build.sh" 2>/dev/null | head -1 | sed 's/.*VERSION:-//')"
  v="${v:-0.1.0}"
  # image names are FLAT (narandill/<name>, per AGENTS.md) -- $p is the repo path
  name="$(basename "$p")"; img="$REPO/$name:$v"
  echo "== verify $p ($img)"
  if [ -x "$p/test/verify_arch.sh" ]; then "$p/test/verify_arch.sh" "$img" || fail=1
  else echo "  (no arch verification — CPU-only or none needed)"; fi
  if [ "$SMOKE" = "1" ]; then
    for s in "$p"/test/smoke_*.sh; do
      [ -e "$s" ] || { echo "  (no smoke scripts)"; break; }
      echo "== smoke $p ($(basename "$s"))"
      (cd "$p" && ./test/"$(basename "$s")" "$img") || fail=1
    done
  fi
done
if [ "$fail" = "0" ]; then echo "SWEEP PASS"; else echo "SWEEP FAIL"; exit 1; fi
