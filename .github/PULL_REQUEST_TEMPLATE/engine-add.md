## New engine: `<name>` (`packages/<task>/<name>`)

## Checklist

- [ ] package dir complete: `Dockerfile` + `build.sh` + `README.md` +
      `test/verify_arch.sh` + `test/smoke_*.sh`
- [ ] registered in **all four** lists: `Makefile PROJECTS`,
      `scripts/sweep-build.sh`, `scripts/sweep-smoke.sh`,
      `.github/workflows/*.yml`
- [ ] `bench/names.py`: family/color/emoji + `PACKAGES` entry (golden rule) +
      `display()` branch if the label needs one
- [ ] `make verify-all` (cuobjdump arch) green; smoke run on a quiet GPU
      (co-tenancy gate, never killing anything)
- [ ] baseline row via `bench run` — `history.jsonl` never hand-edited —
      then `make bench-plot`
- [ ] new upstream code/deps → [`NOTICE`](../NOTICE) entry, license kept

## Numbers

VRAM, image size, headline latency vs the incumbent.

## Local-only

No `docker push`, no `git push`.
