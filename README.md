# edge-containers

<p align="center">
  <img src="https://img.shields.io/badge/platform-Jetson%20Orin-8A2BE2" alt="Platform: Jetson AGX Orin (sm_87)">
  <img src="https://img.shields.io/badge/platform-Jetson%20Thor-8A2BE2" alt="Platform: Jetson AGX Thor (sm_110)">
  <img src="https://img.shields.io/badge/CUDA-13.2-76b900" alt="CUDA 13.2 SBSA">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT">
</p>

<p align="center">
  <b>Production container images for NVIDIA Jetson edge GPUs</b> — a full streaming
  voice stack (speech recognition + synthesis) and a spec-driven benchmark harness.
  One fat image per package serves <b>both boards</b> (native cubins for sm_87 +
  sm_110a plus forward-compat PTX), built on CUDA 13.2 SBSA / JetPack 7.2.1
  userspace with the GPU driver injected from the host.
</p>

> Status: **local only** — nothing is on Docker Hub or GitHub yet. The publish
> workflow activates when this repo is pushed and tagged.

<p align="center">
  <a href="#engine-families">Families</a> •
  <a href="#-benchmarks">Benchmarks</a> •
  <a href="#packages">Packages</a> •
  <a href="#quick-start">Quick start</a> •
  <a href="#benchmarking">Benchmarking</a> •
  <a href="#ci">CI</a>
</p>


## Engine families

> Measured on **jetson-agx-thor-128gb** (default power) — every number below comes from
the committed raw artifacts; full tables and charts in [Benchmarks](#-benchmarks).


<table width="100%">
  <tr>
    <td width="50%" valign="top" align="center">
      <br>
      <h3>🦜&nbsp;parakeet.cpp</h3>
      <p>one parakeet.cpp runtime, several q8_0 GGUF models</p>
      <p><a href="packages/stt/nemotron-unified-asr/">nemotron-unified 0.6B q8_0</a><br><a href="packages/stt/phonon2-stt/">phonon-2.5 q8_0</a><br><a href="packages/stt/parakeet-redux-stt/">parakeet-redux 178M q8_0</a><br><a href="packages/stt/parakeet-redux-stt/">redux packed (CPU)</a><br></p>
      <br>
    </td>
    <td width="50%" valign="top" align="center">
      <br>
      <h3>⚡&nbsp;faster-whisper</h3>
      <p>faster-whisper 1.2.1 on our CTranslate2 4.8.4 build</p>
      <p><a href="packages/stt/wyoming-faster-whisper/">faster-whisper tiny int8</a><br></p>
      <br>
    </td>
  </tr>
</table>


<table width="100%">
  <tr>
    <td width="25%" valign="top" align="center">
      <br>
      <h3>🗣️&nbsp;kokoro</h3>
      <p>Kokoro-82M v1.0, int8 ONNX on onnxruntime-gpu</p>
      <p><a href="packages/tts/kokoro-tts/">kokoro 82M int8</a><br></p>
      <br>
    </td>
    <td width="25%" valign="top" align="center">
      <br>
      <h3>✨&nbsp;paradee</h3>
      <p>Paradee-8M v1.0 int8 ONNX — a Kokoro distillation</p>
      <p><a href="packages/tts/paradee-tts/">paradee 8M int8 (CPU)</a><br><a href="packages/tts/paradee-tts/">paradee 8M int8 (GPU)</a><br></p>
      <br>
    </td>
    <td width="25%" valign="top" align="center">
      <br>
      <h3>🎺&nbsp;piper1-tts</h3>
      <p>piper-tts 1.8.0 on our onnxruntime-gpu build</p>
      <p><a href="packages/tts/wyoming-piper1-tts/">piper1-tts libritts-r medium</a><br></p>
      <br>
    </td>
    <td width="25%" valign="top" align="center">
      <br>
      <h3>🌊&nbsp;phonon-31m</h3>
      <p>gradium/phonon-31m-en via the ptts runtime — CPU-only by design</p>
      <p><a href="packages/tts/phonon31m-tts/">phonon-31m (CPU)</a><br></p>
      <br>
    </td>
  </tr>
</table>


## 📊 Benchmarks

> Rendered from the raw benchmark artifacts (`benchmarks/raw/`, `benchmarks/history.jsonl`)
by `bench plot --readme` — regenerate with `make bench-plot`. 
> Cohort: **jetson-agx-thor-128gb · default power · commit 4dd4272 · 2026-10-31**.
Charts: ★ marks the best engine for that chart's metric. Memory: **cgroup MiB**
counts the container's active file cache too (the model was just read from
disk); **anon MiB** is true process memory.


### Speech recognition — LibriSpeech test-clean

| engine | model | backend | WER | CER | SER | halluc. | RTF p50 | RTF p90 | lat p50 (s) | lat p90 (s) | cgroup MiB | anon MiB | GPU W |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 🥇 nemotron-unified-asr | nemotron-labs-unified-asr-0.6b-q8_0 | cuda | $\color{#3fb950}{\text{0.0177}}$ | $\color{#3fb950}{\text{0.0047}}$ | $\color{#3fb950}{\text{0.2240}}$ | $\color{#3fb950}{\text{0.0000}}$ | $\color{#f85149}{\text{0.1113}}$ | $\color{#f85149}{\text{0.2486}}$ | $\color{#f85149}{\text{0.6912}}$ | 0.8496 | 2,119 | — | 11 |
| 🥈 phonon2-stt | phonon-2.5-english-q8_0 | cuda | 0.0199 | 0.0061 | 0.2620 | $\color{#3fb950}{\text{0.0000}}$ | 0.0900 | 0.2010 | 0.5754 | 0.6871 | 2,070 | — | 8 |
| 🥉 parakeet-redux-stt | redux-q8_0 | cuda | 0.0249 | 0.0083 | 0.3200 | $\color{#3fb950}{\text{0.0000}}$ | 0.0984 | 0.2206 | 0.5806 | 0.7790 | 2,102 | — | 18 |
| parakeet-redux-stt | parakeet-redux packed (redux-packed.gguf) | cpu | 0.0252 | 0.0083 | 0.3200 | $\color{#3fb950}{\text{0.0000}}$ | 0.0707 | 0.1041 | 0.4580 | $\color{#f85149}{\text{0.9607}}$ | 1,006 | — | — (CPU) |
| wyoming-faster-whisper | faster-whisper-tiny-int8 | cuda | $\color{#f85149}{\text{0.0722}}$ | $\color{#f85149}{\text{0.0280}}$ | $\color{#f85149}{\text{0.5940}}$ | $\color{#f85149}{\text{0.0020}}$ | $\color{#3fb950}{\text{0.0130}}$ | $\color{#3fb950}{\text{0.0185}}$ | $\color{#3fb950}{\text{0.0857}}$ | $\color{#3fb950}{\text{0.1507}}$ | 613 | — | 14 |

*500 utterances · 5 configurations · 🥇 **nemotron-unified 0.6B q8_0** at **1.77% WER**
(🥇🥈🥉 = composite top 3); fastest RTF p50
**0.013**. Green = best in column, red = worst. GPU W is the
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
| 🥇 kokoro-tts | Kokoro-82M v1.0 int8 (model_uint8.onnx) | cuda | 0.0850 | $\color{#f85149}{\text{0.1923}}$ | 0.5781 | 1.2016 | 0.5773 | 1.1899 | 1,526 | — | 34 |
| 🥈 paradee-tts | Paradee-8M-v1.0 int8 (baked) | cpu | $\color{#f85149}{\text{0.1503}}$ | 0.1522 | $\color{#f85149}{\text{1.1309}}$ | $\color{#f85149}{\text{3.2617}}$ | $\color{#f85149}{\text{1.1309}}$ | $\color{#f85149}{\text{3.2617}}$ | 604 | — | — (CPU) |
| 🥉 paradee-tts | Paradee-8M-v1.0 int8 (baked) | cuda | $\color{#3fb950}{\text{0.0188}}$ | $\color{#3fb950}{\text{0.0462}}$ | $\color{#3fb950}{\text{0.1414}}$ | 0.1912 | 0.1414 | 0.1912 | 1,508 | — | 16 |
| wyoming-piper1-tts | en_US-libritts_r-medium | cuda | 0.0334 | 0.0998 | 0.1587 | $\color{#3fb950}{\text{0.1738}}$ | 0.1587 | 0.1738 | 1,149 | — | 12 |
| phonon31m-tts | phonon-31m-en | cpu | 0.1237 | 0.1553 | 0.7679 | 2.3239 | $\color{#3fb950}{\text{0.0757}}$ | $\color{#3fb950}{\text{0.1124}}$ | 516 | — | — (CPU) |

*444 sentences · 5 configurations · 🥇🥈🥉 = composite top 3 · fastest RTF p50
**0.019**; best TTFA p50 **76 ms**
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


Full per-utterance data, probes and gate details: `benchmarks/raw/`, `docs/REPORT-benchmark.md`.


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

## Packages

### Speech recognition (`packages/stt/`)

| Package | Image (`narandill/<name>`) | What it is | Streaming |
|---|---|---|---|
| `packages/stt/wyoming-faster-whisper` | `wyoming-faster-whisper:0.1.0` | faster-whisper 1.2.1 on our CTranslate2 4.8.4 (sm_87+sm_110) | wyoming streaming |
| `packages/stt/nemotron-unified-asr` | `nemotron-unified-asr:0.1.0` | nvidia/nemotron-labs-unified-asr-0.6b q8_0 GGUF on parakeet.cpp | **true buffered** (0.16–2.08 s) |
| `packages/stt/parakeet-redux-stt` | `parakeet-redux-stt:0.1.0` | moondream/parakeet-redux (ternary 178M) q8_0 GPU + packed CPU | VAD-chunked (opt-in) |
| `packages/stt/phonon2-stt` | `phonon2-stt:0.1.0` | Fermion Phonon-2.5-english q8_0 GPU + packed CPU | VAD-chunked (opt-in) |

### Speech synthesis (`packages/tts/`)

| Package | Image (`narandill/<name>`) | What it is | Streaming |
|---|---|---|---|
| `packages/tts/wyoming-piper1-tts` | `wyoming-piper1-tts:0.1.0` | piper-tts 1.8.0 on our onnxruntime-gpu 1.31.1 | wyoming streaming |
| `packages/tts/kokoro-tts` | `kokoro-tts:0.1.0` | Kokoro-82M v1.0 int8 ONNX on onnxruntime-gpu (sm_87+sm_110a) | `create_stream()` |
| `packages/tts/phonon31m-tts` | `phonon31m-tts:0.1.0` | gradium/phonon-31m-en via `ptts` — **CPU-only by design** (307 MB) | **true streaming** (TTFA ~76 ms p95) |
| `packages/tts/paradee-tts` | `paradee-tts:0.1.0` | Paradee-8M-v1.0 int8 ONNX (Kokoro distillation) | sentence-chunked |

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

On pre-7.2.1 hosts add `-e NVIDIA_DISABLE_REQUIRE=1` to every `docker run` (driver
580.00 vs the baked 13.2 userspace — minor-version compatibility). After the host
upgrade, drop the flag.

## Layout

```
packages/             the images — packages/<stt|tts|llm>/<name>/
  stt/                  speech recognition: wyoming-faster-whisper, nemotron-unified-asr, parakeet-redux-stt, phonon2-stt
  tts/                  speech synthesis: wyoming-piper1-tts, kokoro-tts, phonon31m-tts, paradee-tts
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
