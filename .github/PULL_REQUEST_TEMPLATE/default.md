## What & why

## How it was verified

- [ ] `make lint` (pre-commit) green
- [ ] `make contract` — 17/17 (if the voice-api surface moved)
- [ ] `make test` — unit suite (pure helpers)
- [ ] images rebuilt locally if `packages/` changed; `make verify-all` green

## Generated surfaces

- [ ] `README.md` / `assets/benchmarks/` re-rendered via `make bench-plot`
      (never hand-edited)

## Notes

Local-only by doctrine: no `docker push`, no `git push`.
