# edge-containers

<p align="center">
  <img src="https://img.shields.io/badge/platform-Jetson%20Orin-eb6834" alt="Platform: Jetson AGX Orin (sm_87)">
  <img src="https://img.shields.io/badge/platform-Jetson%20Thor-8A2BE2" alt="Platform: Jetson AGX Thor (sm_110)">
  <img src="https://img.shields.io/badge/CUDA-13.2-76b900" alt="CUDA 13.2 SBSA">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT">
</p>

<p align="center">
  <b>Production container images for NVIDIA Jetson edge GPUs</b> — a full streaming
  voice stack (speech recognition + synthesis) and a spec-driven benchmark harness.
  One docker image per package serves <b>both boards</b> (native cubins for <code>sm_87</code> +
  <code>sm_110a</code> plus forward-compat PTX), built on CUDA 13.2 SBSA / JetPack 7.2.1
  userspace with the GPU driver injected from the host.
</p>

<p align="center">
  <a href="#packages">Packages</a> •
  <a href="#-benchmarks">Benchmarks</a> •
  <a href="#quick-start">Quick start</a> •
  <a href="#benchmarking">Benchmarking</a> •
  <a href="#ci">CI</a>
</p>

## Packages

### Speech recognition (`packages/stt/`)

| Package | Image (`narandill/<name>`) | Family | What it is | Streaming |
|---|---|---|---|---|
| [`canary-flash-stt`](packages/stt/canary-flash-stt/) | `canary-flash-stt:0.1.0` | 🐤 canary | NVIDIA Canary-180M-Flash int8 (en/de/es/fr + X→EN) via sherpa-onnx | VAD-chunked (opt-in) |
| [`wyoming-faster-whisper`](packages/stt/wyoming-faster-whisper/) | `wyoming-faster-whisper:0.1.0` | ⚡ faster-whisper | faster-whisper 1.2.1 on our CTranslate2 4.8.4 (sm_87+sm_110) | wyoming streaming |
| [`moonshine-streaming-stt`](packages/stt/moonshine-streaming-stt/) | `moonshine-streaming-stt:0.1.0` | 🌙 moonshine | Moonshine Streaming Medium 245M via moonshine-voice (official ORT runtime) | **native incremental** (live partials) |
| [`nemotron-streaming-stt`](packages/stt/nemotron-streaming-stt/) | `nemotron-streaming-stt:0.1.0` | 🔊 nemotron-streaming | NVIDIA Nemotron Speech Streaming EN 0.6B int8 (160 ms lookahead) via sherpa-onnx | **true native** (cache-aware, 160 ms chunks) |
| [`parakeet-tdt-stt`](packages/stt/parakeet-tdt-stt/) | `parakeet-tdt-stt:0.1.0` | 🦜 parakeet-tdt | NVIDIA Parakeet TDT 0.6B v3 int8 (25 languages) via sherpa-onnx | VAD-chunked (opt-in) |
| [`nemotron-unified-asr`](packages/stt/nemotron-unified-asr/) | `nemotron-unified-asr:0.1.0` | 🦜 parakeet.cpp | nvidia nemotron-unified-asr 0.6B (q8_0) on parakeet.cpp | **true buffered** (0.16–2.08 s) |
| [`parakeet-redux-stt`](packages/stt/parakeet-redux-stt/) | `parakeet-redux-stt:0.1.0` | 🦜 parakeet.cpp | moondream parakeet-redux 178M (q8_0) GPU + packed CPU | VAD-chunked (opt-in) |
| [`phonon2-stt`](packages/stt/phonon2-stt/) | `phonon2-stt:0.1.0` | 🦜 parakeet.cpp | fermion phonon-2.5 (q8_0) GPU + packed CPU | VAD-chunked (opt-in) |

### Speech synthesis (`packages/tts/`)

| Package | Image (`narandill/<name>`) | Family | What it is | Streaming |
|---|---|---|---|---|
| [`kokoro-tts`](packages/tts/kokoro-tts/) | `kokoro-tts:0.1.0` | 🗣️ kokoro | Kokoro-82M v1.0 int8 ONNX on onnxruntime-gpu (sm_87+sm_110a) | `create_stream()` |
| [`paradee-tts`](packages/tts/paradee-tts/) | `paradee-tts:0.2.0` | ✨ paradee | Paradee-8M v1.2 int8 ONNX — a Kokoro distillation (4 voices) | sentence-chunked |
| [`phonon31m-tts`](packages/tts/phonon31m-tts/) | `phonon31m-tts:0.1.0` | 🌊 phonon-31m | gradium/phonon-31m-en via `ptts` — **CPU-only by design** (307 MB) | **true streaming** (TTFA ~76 ms p95) |
| [`wyoming-piper1-tts`](packages/tts/wyoming-piper1-tts/) | `wyoming-piper1-tts:0.1.0` | 🎺 piper1-tts | piper-tts 1.8.0 on our onnxruntime-gpu 1.31.1 | wyoming streaming |
| [`pocket-tts`](packages/tts/pocket-tts/) | `pocket-tts:0.1.0` | 🎒 pocket-tts | Kyutai Pocket TTS 100M (official pkg) — streaming + voice cloning, 7 languages | **true streaming** (~200 ms TTFA) |

## Supported Hardware

One image per package serves **both boards** — native cubins for sm_87 and
sm_110a baked side by side, plus forward-compat PTX — so no per-board tags.

<table width="100%">
  <thead>
    <tr><th width="25%"></th><th align="center" width="37.5%"><strong>Jetson AGX Thor</strong></th><th align="center" width="37.5%"><strong>Jetson AGX Orin</strong></th></tr>
  </thead>
  <tbody>
    <tr><td align="center"><strong>status</strong></td><td align="center">✅ supported</td><td align="center">✅ supported</td></tr>
    <tr><td align="center"><strong>compute capability</strong></td><td align="center">sm_110a + sm_110 PTX</td><td align="center">sm_87 + sm_87 PTX</td></tr>
    <tr><td align="center"><strong>host software</strong></td><td align="center">JetPack 7.2.1+ (CUDA 13.2 SBSA userspace, driver injected from host)</td><td align="center">JetPack 7.2.1+ (CUDA 13.2 SBSA userspace, driver injected from host)</td></tr>
    <tr><td align="center"><strong>older hosts</strong></td><td align="center" colspan="2">driver 580.00 vs baked 13.2 userspace: add <code>-e NVIDIA_DISABLE_REQUIRE=1</code> (minor-version compatibility)</td></tr>
    <tr><td align="center"><strong>benchmarked</strong></td><td align="center">✅ measured — see <a href="#-benchmarks">Benchmarks</a></td><td align="center">— (same cubins, no numbers yet)</td></tr>
  </tbody>
</table>


## 📊 Benchmarks

> Rendered from the raw benchmark artifacts (`benchmarks/raw/`, `benchmarks/history.jsonl`)
by `bench plot --readme` — regenerate with `make bench-plot`.
> Cohort: **jetson-agx-thor-128gb · default power · commit 44cdb56 · 2026-11-01**.
Charts: ★ marks the best engine for that chart's metric. Memory: **cgroup MiB**
counts the container's active file cache too (the model was just read from
disk); **anon MiB** is true process memory.


### Speech recognition — LibriSpeech test-clean

| engine | model | backend | WER | CER | SER | halluc. | RTF p50 | RTF p90 | lat p50 (s) | lat p90 (s) | cgroup MiB | anon MiB | GPU W |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 🥇 nemotron-unified-asr | nemotron-unified 0.6B q8_0 | cuda | $\color{#3fb950}{\text{0.0177}}$ | $\color{#3fb950}{\text{0.0047}}$ | $\color{#3fb950}{\text{0.2240}}$ | $\color{#3fb950}{\text{0.0000}}$ | $\color{#f85149}{\text{0.1092}}$ | $\color{#f85149}{\text{0.2380}}$ | $\color{#f85149}{\text{0.6920}}$ | 0.7954 | 2,179 | 1,029 | 7 |
| 🥈 phonon2-stt | phonon-2.5 q8_0 | cuda | 0.0199 | 0.0061 | 0.2620 | $\color{#3fb950}{\text{0.0000}}$ | 0.1051 | 0.2321 | 0.6686 | 0.7717 | 2,170 | 1,030 | 7 |
| 🥉 parakeet-redux-stt | parakeet-redux 178M q8_0 | cuda | 0.0249 | 0.0083 | 0.3200 | $\color{#3fb950}{\text{0.0000}}$ | 0.1055 | 0.2323 | 0.6677 | 0.7716 | 2,181 | 1,031 | 7 |
| parakeet-redux-stt | redux packed (CPU) | cpu | 0.0252 | 0.0083 | 0.3200 | $\color{#3fb950}{\text{0.0000}}$ | 0.0668 | 0.1001 | 0.4241 | $\color{#f85149}{\text{0.8946}}$ | 1,101 | 730 | — (CPU) |
| wyoming-faster-whisper | faster-whisper tiny int8 | cuda | $\color{#f85149}{\text{0.0721}}$ | $\color{#f85149}{\text{0.0279}}$ | $\color{#f85149}{\text{0.5940}}$ | $\color{#f85149}{\text{0.0020}}$ | $\color{#3fb950}{\text{0.0118}}$ | $\color{#3fb950}{\text{0.0179}}$ | $\color{#3fb950}{\text{0.0748}}$ | $\color{#3fb950}{\text{0.1375}}$ | 616 | 561 | 14 |

> *500 utterances · 5 configurations · 🥇 **nemotron-unified 0.6B q8_0** at **1.77% WER**
(🥇🥈🥉 = composite top 3); fastest RTF p50
**0.012**. Green = best in column, red = worst. GPU W is the
tegrastats module draw over each engine's window; the packed-CPU engine has none
(hollow point, idle draw) in the power chart.*

<p align="center">
<img src="assets/benchmarks/stt/stt-wer-vs-rtf.png" alt="Speech recognition — accuracy vs speed with Pareto front" width="100%">
</p>


<p align="center">
<img src="assets/benchmarks/stt/stt-latency-box.png" alt="Speech recognition — per-utterance RTF distributions" width="100%">
</p>

<p align="center">
<img src="assets/benchmarks/stt/stt-power.png" alt="Speech recognition — GPU power vs accuracy" width="100%">
</p>

<p align="center">
<img src="assets/benchmarks/stt/stt-memory.png" alt="Speech recognition — peak container memory" width="100%">
</p>

<p align="center">
<img src="assets/benchmarks/stt/stt-ranking.png" alt="Speech recognition — corpus WER" width="100%">
</p>

<p align="center">
<img src="assets/benchmarks/stt/stt-heatmap.png" alt="Speech recognition — normalized metric heatmap" width="100%">
</p>


### Speech synthesis — 444 sentences (LibriSpeech text)

| engine | model | backend | RTF p50 | RTF p90 | synth p50 (s) | synth p90 (s) | TTFA p50 (s) | TTFA p90 (s) | cgroup MiB | anon MiB | GPU W |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 🥇 kokoro-tts | kokoro 82M int8 | cuda | 0.0482 | 0.0868 | 0.3098 | 0.8135 | 0.3098 | 0.8096 | 1,583 | 1,457 | 14 |
| 🥈 paradee-tts | paradee 8M int8 (CPU) | cpu | $\color{#f85149}{\text{0.1497}}$ | $\color{#f85149}{\text{0.1516}}$ | $\color{#f85149}{\text{1.1171}}$ | $\color{#f85149}{\text{3.2504}}$ | $\color{#f85149}{\text{1.1171}}$ | $\color{#f85149}{\text{3.2504}}$ | 629 | 563 | — (CPU) |
| 🥉 paradee-tts | paradee 8M int8 (GPU) | cuda | $\color{#3fb950}{\text{0.0188}}$ | $\color{#3fb950}{\text{0.0462}}$ | $\color{#3fb950}{\text{0.1412}}$ | $\color{#3fb950}{\text{0.1905}}$ | 0.1412 | 0.1905 | 1,513 | 1,418 | 16 |
| wyoming-piper1-tts | piper1-tts libritts-r medium | cuda | 0.0406 | 0.1210 | 0.1884 | 0.2413 | 0.1884 | 0.2413 | 1,158 | 1,099 | 26 |
| phonon31m-tts | phonon-31m (CPU) | cpu | 0.1119 | 0.1216 | 0.6763 | 1.9760 | $\color{#3fb950}{\text{0.0663}}$ | $\color{#3fb950}{\text{0.0978}}$ | 512 | 498 | — (CPU) |

> *444 sentences · 5 configurations · 🥇🥈🥉 = composite top 3 · fastest RTF p50
**0.019**; best TTFA p50 **66 ms**
(300 ms gate). TTFA = time to first audio: streaming engines report the first
chunk; single-chunk engines (paradee, piper1-tts) deliver at synthesis completion, so
their TTFA equals the synth p50/p90.*

<p align="center">
<img src="assets/benchmarks/tts/tts-ranking.png" alt="Speech synthesis — real-time factor ranking" width="100%">
</p>

<p align="center">
<img src="assets/benchmarks/tts/tts-ttfa.png" alt="Speech synthesis — time to first audio with p90 ticks" width="100%">
</p>


<p align="center">
<img src="assets/benchmarks/tts/tts-synth-box.png" alt="Speech synthesis — synthesis latency distributions" width="100%">
</p>

<p align="center">
<img src="assets/benchmarks/tts/tts-memory.png" alt="Speech synthesis — peak container memory" width="100%">
</p>

<p align="center">
<img src="assets/benchmarks/tts/tts-heatmap.png" alt="Speech synthesis — normalized metric heatmap" width="100%">
</p>


> Full per-utterance data, probes and gate details: `benchmarks/raw/`, `docs/REPORT-benchmark.md`.


## Quick start

```bash
# 1) shared artifact builders (cached; seconds on replay)
make builders

# 2) build everything
make build-all          # or: cd packages/tts/kokoro-tts && ./build.sh

# 3) verify cubins + smoke on the GPU (waits for free VRAM, never kills co-tenants)
make verify-all
make smoke-all

# 4) run something
docker run --runtime nvidia --rm -p 10300:10300 -v /data/wfw:/data narandill/wyoming-faster-whisper:0.1.0
docker run --runtime nvidia --rm --entrypoint stt-transcribe narandill/nemotron-unified-asr:0.1.0 /audio/x.wav
```

> On pre-7.2.1 hosts add `-e NVIDIA_DISABLE_REQUIRE=1` to every `docker run` (driver
580.00 vs the baked 13.2 userspace — minor-version compatibility). After the host
upgrade, drop the flag.

## Layout

```
packages/             the images — packages/<stt|tts|llm>/<name>/
  stt/                  speech recognition: wyoming-faster-whisper, nemotron-unified-asr, parakeet-redux-stt, phonon2-stt, moonshine-streaming-stt, canary-flash-stt, parakeet-tdt-stt, nemotron-streaming-stt
  tts/                  speech synthesis: wyoming-piper1-tts, kokoro-tts, phonon31m-tts, paradee-tts, pocket-tts
  llm/                  LLM runtime (outside the voice stack)
builders/             shared artifact builders (wheels-cu132, parakeet-cpp) — not shipped
bench/                spec-driven benchmark harness (datasets, engines, probes, metrics)
benchmarks/           history.jsonl + raw run artifacts (committed) — feeds the charts
assets/benchmarks/    generated charts + manifest (committed)
manifests/            frozen benchmark manifests (seed 42) — the reproducibility anchor
scripts/              sweep-build.sh / sweep-smoke.sh
docs/                 build & benchmark reports
```

Each package dir is standalone: `Dockerfile + build.sh + README +
test/{verify_arch,smoke_*}`.

**Decoupling contract:** an engine references shared artifacts only through a single
overridable ARG (`WHEELS_IMAGE`, `PARAKEET_DIST_IMAGE`) — see `ARCHITECTURE.md`.

## Benchmarking

```bash
make bench-prepare      # frozen manifests: 500 utterances + 444 sentences (LibriSpeech, seed 42)
make bench-selfcheck    # device, CUDA arch, probes, engines
bench/.venv/bin/python -m bench.cli run bench/configs/thor_stt_baseline.yaml
bench/.venv/bin/python -m bench.cli run bench/configs/thor_tts_baseline.yaml

# matrix + charts + README (reads benchmarks/history.jsonl — never re-runs inference)
make bench-plot
```

The runner **gates every GPU run on co-tenancy**: the window counts as clean
only after `gpu_confirm_samples` (default 6) consecutive samples under the
threshold — a lucky gap between co-tenant bursts doesn't count — and it defers
rather than measure contaminated. Co-tenants are never killed. Before each
engine (and once the window opens) the runner also **drops the host page
cache** (`sysctl -w vm.drop_caches=3` via a targeted NOPASSWD grant — install
it with `bench grant-permissions`; without the grant it silently skips), so a
warm file cache can't inflate the memory numbers. Every run appends to
`benchmarks/history.jsonl`; baseline-class runs also publish their raw
artifacts to `benchmarks/raw/<run_id>/`.

## CI

| Workflow | Trigger | Runner | What it does |
|---|---|---|---|
| `lint.yml` | push/PR | `ubuntu-latest` | pre-commit (shellcheck, hadolint, ruff, yaml/json checks) |
| `build-smoke.yml` | `workflow_dispatch` | `[self-hosted, jetson]` | build matrix + arch verification (+ optional GPU smoke) |
| `publish.yml` | tag `v*` | `[self-hosted, jetson]` | build → arch gate → push all `narandill/*` tags (needs `DOCKERHUB_USER`/`DOCKERHUB_TOKEN` secrets) |

## Licenses

Build recipes are MIT. Each image carries its engine's license — check every project's
README (all engines here are permissive or OpenMDW; nothing NC).
