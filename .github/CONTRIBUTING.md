# Contributing to voice-containers

Production container images for NVIDIA Jetson edge GPUs — Orin (`sm_87`) and
Thor (`sm_110`) — a streaming voice stack: speech recognition, speech
synthesis, and a local LLM, plus the benchmark harness that measures it all.
Every number in `README.md` is rendered from raw data, so **a contribution can
be as small as one benchmark run from your box**.

## The hard rules

1. **The GPU is shared — never kill co-tenants.** Other workloads (and the
   people owning them) live on the same device. Never `docker stop`, restart
   or rebuild anything you did not start, and never free VRAM by force. Every
   GPU measurement runs behind the co-tenancy gate in `bench/runner.py`
   (busy GPU → wait, then defer) — use it, don't route around it.
2. **Every number ships with its raw artifact.** Results go to
   `results/<run_id>/`; one line per engine config lands in
   [`benchmarks/history.jsonl`](benchmarks/history.jsonl); baseline-class runs
   publish `benchmarks/raw/<run_id>/` (no audio). Charts are rendered only by
   `bench/charts.py` under the `bench/theme.py` design system, and the README
   only from [`README.md.j2`](README.md.j2) — CI fails if `README.md` does not
   reproduce from the committed data. No prose-only results.
3. **`README.md` is generated.** Never hand-edit it. Edit the template or the
   data, then `make bench-plot`.
4. **Local-first: nothing is pushed.** Images are local tags
   `narandill/<name>:<version>`; publishing is the maintainer's tag-gated
   [`publish.yml`](.github/workflows/publish.yml). No `docker push`, no
   `git push` from drive-by contributions.
5. **Bench code runs in the bench venv.** The system `python3` lacks the
   stack — always `bench/.venv/bin/python …` (or `make` targets, which do it
   for you).

## The layout in one paragraph

`packages/<task>/<engine>/` — one standalone service per directory
(`Dockerfile` + `build.sh` + `README` + `test/{verify_arch,smoke_*}`).
`bench/` — the harness: datasets, engines, probes, metrics, matrix, charts,
readme renderer. `benchmarks/` — append-only `history.jsonl` + committed raw
runs. `manifests/` — frozen benchmark manifests (seed 42), the reproducibility
anchor. `task/` — gitignored local workspace (specs, reviews, ledgers);
scratch there never gets committed.

## Adding an engine

New package = `packages/<task>/<name>` **plus registration in all four
lists**: `Makefile PROJECTS`, `scripts/sweep-build.sh`,
`scripts/sweep-smoke.sh`, and the workflows — then a `bench/names.py` entry
(family/color/emoji + `PACKAGES`, the golden rule: every package registers),
`make verify-all`, a smoke on a quiet GPU, a baseline `bench run`, and the
README re-render. The
[`engine-add` PR template](.github/PULL_REQUEST_TEMPLATE/engine-add.md) is the
checklist. New third-party code or models → a
[`NOTICE`](NOTICE) entry, license kept with it.

## Running / contributing benchmarks

```bash
make builders        # shared artifact builders (cached)
make build-all       # every engine image, local tags only
make verify-all      # cuobjdump arch gate — no GPU touched
make bench-prepare   # frozen manifests (seed 42) — the cohort contract
bench/.venv/bin/python -m bench.cli run bench/configs/<cohort>.yaml
make bench-plot      # charts + matrix + README from history.jsonl
```

Cohorts `(task, device, power profile, manifest)` are never ranked across a
mix; incomplete runs never replace complete ones. Attach
`results/<run_id>/` (or the raw export) — the
[`benchmark-result` issue](.github/ISSUE_TEMPLATE/benchmark_result.md) or the
[`bench-results` PR](.github/PULL_REQUEST_TEMPLATE/bench-results.md) template
carries the provenance CI can't check.

## Development setup

```bash
python3 -m venv bench/.venv && bench/.venv/bin/pip install -r bench/requirements.txt
bench/.venv/bin/pip install pytest        # or: pip install -e '.[dev]' extras
make lint        # pre-commit: whitespace/EOF, yaml/json, shellcheck, hadolint, ruff, bandit
make contract    # voice-api v1 contract test — run before campaigns
make test        # unit suite: pure helpers only, no GPU, no docker
```

## Commit style

Short, lowercase, grouped by concern (see `git log`).

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE). New third-party code needs a
[`NOTICE`](NOTICE) entry and its license kept with it; the container images
redistribute their upstreams' licenses inside the images themselves.
