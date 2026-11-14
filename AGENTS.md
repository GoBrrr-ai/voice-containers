# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository.

## What this repo is

`voice-containers` — production container images for NVIDIA Jetson edge GPUs
(sm_87 Orin + sm_110 Thor): 16 self-contained image projects (10 speech
recognition, 5 speech synthesis, 1 llama.cpp) under `packages/`, two shared
`builders/`, an HTTP contract (`docs/voice-api.md`) and a spec-driven
benchmark harness (`bench/`) that renders the README, charts and result
matrices from summary data. See `ARCHITECTURE.md` for the dependency graph.

## Golden rules

1. **README.md is GENERATED.** It is rendered by `bench plot --readme` from
   `README.md.j2` (Jinja2, `StrictUndefined`) + `benchmarks/history.jsonl` +
   `assets/benchmarks/manifest.json`. Never hand-edit `README.md` — edit the
   template or the data, then regenerate (`make bench-plot`).
2. **Publishing is tag-driven, nothing ad-hoc.** Images leave the machine only
   via `vX.Y.Z` tags → `publish.yml` (build → arch gate → push to Docker Hub
   `narandill/*`) or manually via `scripts/docker_release.py`. No `docker
   push` outside those paths. Git pushes to `origin/main` only when the user
   asks.
3. **Never measure on a busy GPU.** The runner gates GPU runs on co-tenant
   memory (≥100 GB held → wait ≤30 min → defer). Never kill co-tenants.
4. **Use the bench venv.** The system `python3` lacks numpy/jinja2 — always
   run bench/pytest/plot code with `bench/.venv/bin/python` (or rely on
   `.envrc` putting it on PATH).
5. **Banned strings:** the pre-rename project name `edge-containers` must not
   appear in any tracked file (`grep -ri edge-containers` over tracked files
   returns 0). Same for `PRD` — say "spec". The spec file
   `task/benchmark-prd.md` keeps its lowercase name.
6. **`task/` is gitignored** and lives inside this repo. An older checkout
   elsewhere on disk keeps a compat symlink into it — if you find such a
   symlink, don't delete it.
7. **Engine ids are internal.** `parakeet_cpp_0…3`, `kokoro_0`, `piper_3` …
   are config indices. Anything user-visible (tables, charts, cards) uses the
   original project + model names via `bench/names.py` — add new projects
   there (image → family/emoji/streaming note + display label).
8. **Git carries summary data, not raw results.** Only `benchmarks/history.jsonl`,
   `manifests/`, charts and metadata-only `benchmarks/raw/<run_id>/`
   (`bench/history.py::publish_raw`: audit trail + precomputed box
   statistics — the distribution-chart input that makes `bench plot`
   reproducible from git alone)
   are committed; raw logs, full utterance records, audio and `results/` stay
   local. Charts render as text-embedded SVG, byte-stable
   across runs (`bench/charts.py`; no raster step). History rewrites are
   ceremony (force-push, signed) — avoid.
9. **GPU runtime images:** strip `/usr/local/cuda/compat*` in the final stage; see
   `docs/postmortem-cuda-compat-hook.md` (ctk 1.19.1 createContainer hook panic).

## Layout

| Path | Purpose |
|---|---|
| `packages/stt/` · `packages/tts/` · `packages/llm/` | the 16 image projects — authoritative list in `scripts/projects.txt` |
| `builders/` | shared artifact builders (`parakeet-cpp` dist, `wheels-cu132`) — not shipped |
| `bench/` | harness: datasets, engines, probes, metrics, matrix, charts, readme |
| `benchmarks/` | `history.jsonl` (append-only) + metadata-only `raw/<run_id>/` |
| `assets/` | generated benchmark charts + `logo.png`/`logo.gif` (README header) |
| `manifests/` | frozen benchmark manifests (seed 42) — the reproducibility anchor |
| `docs/` | `voice-api.md` + OpenAPI spec — the HTTP contract for non-wyoming images |
| `deploy/` | docker-compose composing the built images (no builds) |
| `tools/` | `voice_api_contract_test.py` — stdlib contract check (`make contract`) |
| `scripts/` | `projects.txt` (roster), sweep runners, `docker_release.py` |
| `task/`, `reports/`, `selfcheck.json` | workspace + generated outputs (gitignored) |

Each project dir is standalone: `Dockerfile + build.sh + README +
test/{verify_arch,smoke_*}`. Dockerfiles start with `ARG BASE_IMAGE /
FROM ${BASE_IMAGE}` and use `uv pip install`.

## The project roster — one file

`scripts/projects.txt` (one `packages/...` path per line) is the single
source of truth. The Makefile, `scripts/sweep-*.sh` and both build workflows
read it; **adding a project = one line there** (+ `bench/names.py` if it
benches). Image names stay flat: `narandill/<name>`.

## Commands

```bash
make builders          # shared artifact builders (cached)
make build-all         # docker build every image (roster from scripts/projects.txt)
make verify-all        # cuobjdump arch verification (no GPU)
make smoke-all         # GPU smoke suite (co-tenancy aware)
make lint              # pre-commit over the repo
make test              # unit tests + coverage gate (bench venv, no GPU/docker)
make contract          # voice-api v1 wire contract (stdlib, CI-safe)

make bench-prepare     # frozen manifests (LibriSpeech 500 utts / 444 sentences, seed 42)
make bench-selfcheck   # device, CUDA arch, probes, engines report
bench/.venv/bin/python -m bench.cli run bench/configs/thor_stt_baseline.yaml
bench/.venv/bin/python -m bench.cli run bench/configs/thor_tts_baseline.yaml
make bench-plot        # charts + matrix + README.md from history.jsonl (no inference)
make bench-prune       # delete superseded benchmarks/raw dirs (--yes applies)
```

## Benchmark data flow

```
bench run  →  results/<run_id>/         (+ tegrastats, docker stats, utterances.jsonl)
           →  benchmarks/history.jsonl  one line per engine config (append-only)
           →  benchmarks/raw/<run_id>/  baseline runs: audit trail + box stats (no raw logs/audio)
bench plot --readme
           →  assets/benchmarks/<task>/*.svg + manifest.json
           →  reports/matrix/{matrix_<task>.csv,.json,index_<task>.html}
           →  README.md                 rendered from README.md.j2
```

- Completeness = `n_ok` equals the manifest line count; incomplete runs are
  superseded by later full ones (`matrix.select_rows`).
- Cohorts `(task, device, power_profile, manifest)` are never ranked together.
- TTFA for single-chunk engines (paradee, piper1-tts) is derived
  (= synthesis completion time) in `bench/history.py::_derive_metrics`;
  streaming engines (kokoro, phonon-31m) report the measured first chunk.
- CPU engines have no `GPU W` of their own — tegrastats module draw during
  their window is idle noise; tables show `— (CPU)`, the power chart draws a
  hollow point annotated "GPU idle".
- Footer commit hashes come from each run's `env.json`, not current HEAD.
- The chain probe (`bench chain_probe`) writes `results/chain_*/` only — its
  summary lives in task notes, never in git.

## Conventions

- Commit style: short lowercase messages; all commits carry the single
  identity `Mieszko Syty <mieszko@ms1design.pl>` (GPG-signed). If a
  pre-commit hook rewrites files, `git add -A` and retry.
- Charts: dark theme (`bench/theme.py`), identity colors per engine family
  (Okabe-Ito), marker shape = backend (circle CPU, square CUDA), titles say
  "Speech recognition/synthesis" — never bare "STT"/"TTS" in visible text.
- Metric columns are p50 **and** p90; direction-aware coloring (best green /
  worst red) uses GitHub-renderable LaTeX (`$\color{...}{\text{...}}$`).
- Gates live in `matrix.csv` only, not in README tables.

## Validating changes

```bash
make bench-plot                        # regenerates README.md + charts
grep -ci edge-containers README.md     # 0 — banned-string check (rule 5)
bash -n scripts/sweep-build.sh         # after touching scripts
pre-commit run --all-files
```

## Task workspace & wiki

`task/` (gitignored) is the workspace — **start at `task/README.md`** (30-second
map) whose index is `task/wiki/README.md`: benchmarking, engines, platform,
memory/VRAM, Orin Nano deployment, experiment log, decision log, archive map.
Read it before re-researching anything: the deferred experiment queue (E1–E10)
lives in `task/orin-nano-8gb-plan.md`, the TRT report in
`task/research-trt-vram.md`, and topic research (each with a `summary.md`)
under `task/research/`. Quick search:
`grep -ril "<term>" task/ --include="*.md" | head`.
`task/src/` contains upstream `.git` checkouts — never commit or COPY them
into images.
