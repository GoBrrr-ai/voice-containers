#!/usr/bin/env bash
# Inner: cuobjdump files listed in /verify/files.txt (REQUIRE <path> = enforce,
# INFO <path> = report only). Requires SASS sm_87 + sm_110 and PTX sm_110.
set -euo pipefail
FAIL=0
CHECKED=0
while read -r CLASS f; do
  [ -f "$f" ] || { echo "skip (missing): $f"; [ "$CLASS" = REQUIRE ] && FAIL=1; continue; }
  CHECKED=$((CHECKED+1))
  echo "--- $(basename "$f") [$CLASS] ---"
  ELF=$(cuobjdump --list-elf "$f" 2>/dev/null | grep -oE 'sm_[0-9]+[a-z]?' | sort -u | tr '\n' ' ' || true)
  PTX=$(cuobjdump --list-ptx "$f" 2>/dev/null | grep -oE '(sm|compute)_[0-9]+[a-z]?' | sort -u | tr '\n' ' ' || true)
  echo "  SASS: ${ELF:-none}"
  echo "  PTX:  ${PTX:-none}"
  if [ "$CLASS" = REQUIRE ]; then
    for arch in sm_87 sm_110; do
      case "$ELF" in *$arch*) : ;; *) echo "  !! MISSING $arch SASS"; FAIL=1 ;; esac
    done
    case "$PTX" in *sm_110*) : ;; *) echo "  !! MISSING sm_110 (compute_110) PTX"; FAIL=1 ;; esac
  fi
done < /verify/files.txt
if [ "$CHECKED" = 0 ]; then echo "RESULT: FAIL - no artifacts were checked"; exit 1; fi
if [ "$FAIL" = 0 ]; then echo "RESULT: PASS - sm_87 + sm_110 SASS and sm_110 PTX present"; else echo "RESULT: FAIL"; exit 1; fi
