# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository.

## What this repo is

`edge-containers` — production container images for NVIDIA Jetson edge GPUs
(sm_87 Orin + sm_110 Thor): a streaming voice stack (speech recognition +
synthesis) organized as `stt/`, `tts/` and `llm/` project dirs, shared
`builders/`, and a spec-driven benchmark harness (`bench/`) that renders the
README, charts and result matrices from raw data.

## Golden rules

1. **README.md is GENERATED.** It is rendered by `bench plot --readme` from
   `README.md.j2` (Jinja2, `StrictUndefined`) + `benchmarks/history.jsonl` +
   `assets/benchmarks/manifest.json`. Never hand-edit `README.md` — edit the
   template or the data, then regenerate (`make bench-plot`).
2. **Never push anywhere.** Nothing goes to Docker Hub or GitHub. All images
   are local tags `narandill/<name>:<version>` (+`latest`), one version each.
3. **Never measure on a busy GPU.** The runner gates GPU runs on co-tenant
   memory (≥100 GB held → wait ≤30 min → defer). Never kill co-tenants.
4. **Use the bench venv.** The user's system `python3` lacks numpy/jinja2 —
   always run bench code with `bench/.venv/bin/python`.
5. **No banned strings**: the pre-rename names (purged; git
   history excepted) and `PRD` (say "spec"). The spec file
   `task/benchmark-prd.md` itself keeps its lowercase name.
6. **`task/` is gitignored** and lives inside this repo. The old
   `jetson-containers` clone keeps a compat symlink to it — don't delete.
7. **Engine ids are internal.** `parakeet_cpp_0…3`, `kokoro_0`, `piper_3` …
   are config indices. Anything user-visible (tables, charts, cards) uses the
   original project + model names via `bench/names.py` — add new projects
   there (image → family/emoji/streaming note + display label).

## Layout

| Path | Purpose |
|---|---|
| `packages/stt/` | speech recognition: wyoming-faster-whisper, nemotron-unified-asr, parakeet-redux-stt, phonon2-stt |
| `packages/tts/` | speech synthesis: wyoming-piper1-tts, kokoro-tts, phonon31m-tts, paradee-tts |
| `packages/llm/` | llama-cpp (outside the voice stack) |
| `builders/` | shared artifact builders (wheels-cu132, parakeet-cpp) — not shipped |
| `bench/` | harness: datasets, engines, probes, metrics, matrix, charts, readme |
| `benchmarks/` | `history.jsonl` (append-only) + `raw/<run_id>/` committed artifacts |
| `assets/benchmarks/` | generated charts (2× dark PNGs) + `manifest.json` |
| `manifests/` | frozen benchmark manifests (seed 42) — the reproducibility anchor |
| `task/` | spec, reviews, research, voice-data (gitignored) |
| `reports/`, `selfcheck.json` | generated outputs — gitignored |

Each project dir is standalone: `Dockerfile + build.sh + README +
test/{verify_arch,smoke_*}`. Dockerfiles start with `ARG BASE_IMAGE /
FROM ${BASE_IMAGE}` and use `uv pip install`.

## Commands

```bash
make builders          # shared artifact builders (cached)
make build-all         # docker build every engine image (loops PROJECTS in Makefile)
make verify-all        # cuobjdump arch verification (no GPU)
make smoke-all         # GPU smoke suite (co-tenancy aware)
make lint              # pre-commit over the repo

make bench-prepare     # frozen manifests (LibriSpeech 500 utts / 444 sentences, seed 42)
make bench-selfcheck   # device, CUDA arch, probes, engines report
bench/.venv/bin/python -m bench.cli run bench/configs/thor_stt_baseline.yaml
bench/.venv/bin/python -m bench.cli run bench/configs/thor_tts_baseline.yaml
make bench-plot        # charts + matrix + README.md from history.jsonl (no inference)
```

`PROJECTS` in the Makefile / `scripts/sweep-*.sh` / `.github/workflows/*.yml`
lists full paths (`packages/stt/...`, `packages/tts/...`, `packages/llm/...`) —
update all four when adding a project. Docker image names stay flat
(`narandill/<name>`).

## Benchmark data flow

```
bench run  →  results/<run_id>/         (+ tegrastats, docker stats, utterances.jsonl)
           →  benchmarks/history.jsonl  one line per engine config (append-only)
           →  benchmarks/raw/<run_id>/  baseline-class runs publish raws (no audio)
bench plot --readme
           →  assets/benchmarks/<task>/*.png + manifest.json
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

## Conventions

- Commit style: short lowercase messages. If a pre-commit hook rewrites files,
  `git add -A` and retry.
- Charts: dark theme (`bench/theme.py`), identity colors per engine family
  (Okabe-Ito), marker shape = backend (circle CPU, square CUDA), titles say
  "Speech recognition/synthesis" — never bare "STT"/"TTS" in visible text.
- Metric columns are p50 **and** p90; direction-aware coloring (best green /
  worst red) uses GitHub-renderable LaTeX (`$\color{...}{\text{...}}$`).
- Gates live in `matrix.csv` only, not in README tables.

## Validating changes

```bash
make bench-plot                                # regenerates README.md + charts
grep -ci "PRD\|narandill" README.md     # must be 0
bash -n scripts/sweep-build.sh                 # after touching scripts
pre-commit run --all-files
```
