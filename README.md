# edge-containers

<p align="center">
  <img src="https://img.shields.io/badge/platforms-Jetson%20AGX%20Thor%20%C2%B7%20AGX%20Orin%20%28sm_110%20%C2%B7%20sm_87%29-76b900" alt="Platforms: Jetson AGX Thor, AGX Orin">
  <img src="https://img.shields.io/badge/CUDA-13.2%20SBSA%20%C2%B7%20JetPack%207.2.1-76b900" alt="CUDA 13.2 SBSA on JetPack 7.2.1">
  <img src="https://img.shields.io/docker/v/narandill/llama-cpp?label=llama-cpp&sort=semver&color=1baf7a" alt="llama-cpp on Docker Hub">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT">
</p>

<p align="center">
  <b>Production container images for NVIDIA Jetson edge GPUs</b> — an LLM runtime, a full
  streaming voice stack (STT + TTS), and a PRD-driven benchmark harness. One fat image per
  package serves <b>both boards</b> (native cubins for sm_87 + sm_110a plus forward-compat
  PTX), built on CUDA 13.2 SBSA / JetPack 7.2.1 userspace with the GPU driver injected
  from the host.
</p>

> Status: **local only** — nothing is on Docker Hub or GitHub yet. The publish workflow
> activates when this repo is pushed and tagged.


## 📊 Benchmarks

Rendered from the raw benchmark artifacts (`benchmarks/raw/`, `benchmarks/history.jsonl`)
by `bench plot --readme` — regenerate with `make bench-plot`. Cohort:
**jetson-agx-thor-128gb · default power · commit 4dd4272 · 2026-10-31**.

### Speech recognition (STT) — LibriSpeech test-clean

| engine | model | backend | rank | WER | CER | SER | halluc. | RTF p50 | lat p50 (s) | mem MiB | W | gates |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| parakeet_cpp_0 | ron-labs-unified-asr-0.6b-q8_0 | cuda | 1 | $\color{#3fb950}{\text{0.0177}}$ | $\color{#3fb950}{\text{0.0047}}$ | $\color{#3fb950}{\text{0.2240}}$ | $\color{#3fb950}{\text{0.0000}}$ | $\color{#f85149}{\text{0.1113}}$ | $\color{#f85149}{\text{0.6912}}$ | 2,119 | 11 | ✓ |
| parakeet_cpp_2 | phonon-2.5-english-q8_0 | cuda | 2 | 0.0199 | 0.0061 | 0.2620 | $\color{#3fb950}{\text{0.0000}}$ | 0.0900 | 0.5754 | 2,070 | 8 | ✓ |
| parakeet_cpp_1 | redux-q8_0 | cuda | 3 | 0.0249 | 0.0083 | 0.3200 | $\color{#3fb950}{\text{0.0000}}$ | 0.0984 | 0.5806 | 2,102 | 18 | ✓ |
| parakeet_cpp_3 | redux-packed | cpu | 4 | 0.0252 | 0.0083 | 0.3200 | $\color{#3fb950}{\text{0.0000}}$ | 0.0707 | 0.4580 | 1,006 | — | ✓ |
| faster_whisper_4 | faster-whisper-tiny-int8 | cuda | 5 | $\color{#f85149}{\text{0.0722}}$ | $\color{#f85149}{\text{0.0280}}$ | $\color{#f85149}{\text{0.5940}}$ | $\color{#f85149}{\text{0.0020}}$ | $\color{#3fb950}{\text{0.0130}}$ | $\color{#3fb950}{\text{0.0857}}$ | 613 | 14 | ✓ |

*500 utterances · 5 configurations · winner:
**parakeet_cpp_0** (ron-labs-unified-asr-0.6b-q8_0) at **1.77% WER**;
fastest RTF p50 **0.013**. Green = best in column, red = worst.*




### Speech synthesis (TTS) — 444 sentences (LibriSpeech text)

| engine | model | backend | rank | RTF p50 | synth p50 (s) | TTFA p50 (s) | mem MiB | W | gates |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| kokoro_0 | model_uint8.onnx | cuda | 1 | 0.0850 | 0.5781 | $\color{#f85149}{\text{0.5773}}$ | 1,526 | 34 | ✓ |
| paradee_2_cpu | baked | cpu | 2 | $\color{#f85149}{\text{0.1503}}$ | $\color{#f85149}{\text{1.1309}}$ | — | 604 | 5 | ✓ |
| paradee_2_cuda | baked | cuda | 3 | $\color{#3fb950}{\text{0.0188}}$ | $\color{#3fb950}{\text{0.1414}}$ | — | 1,508 | 16 | ✓ |
| piper_3 | en_US-libritts_r-medium | cuda | 4 | 0.0334 | 0.1587 | — | 1,149 | 12 | ✓ |
| ptts_1 | phonon-31m-en | cpu | 5 | 0.1237 | 0.7679 | $\color{#3fb950}{\text{0.0757}}$ | 516 | 23 | ✓ |

*444 sentences · 5 configurations · fastest RTF p50
**0.019**; best TTFA p50 **76 ms**
(gate 300 ms).*


<p align="center">
<img src="assets/benchmarks/stt/stt-wer-vs-rtf.png" alt="STT accuracy vs speed — WER against RTF, Pareto front" width="100%">
</p>

<p align="center">
<img src="assets/benchmarks/stt/stt-heatmap.png" alt="STT normalized metric heatmap" width="100%">
</p>


<p align="center">
<img src="assets/benchmarks/tts/tts-ranking.png" alt="TTS RTF ranking" width="100%">
</p>

<p align="center">
<img src="assets/benchmarks/tts/tts-ttfa.png" alt="TTS time-to-first-audio" width="100%">
</p>


Full per-utterance data, probes and gate details: `reports/matrix/`, `docs/REPORT-benchmark.md`.


## Projects

| Project | Image (`narandill/<name>`) | What it is | Streaming |
|---|---|---|---|
| `llama-cpp` | `llama-cpp:0.8.0` | llama.cpp v0.8.0 + llama-cpp-python 0.3.36, CUDA graphs, Orin MMVQ patch | — (LLM) |
| `wyoming-faster-whisper` | `wyoming-faster-whisper:0.1.0` | Wyoming STT: faster-whisper 1.2.1 on our CTranslate2 4.8.4 (sm_87+sm_110) | wyoming streaming |
| `wyoming-piper1-tts` | `wyoming-piper1-tts:0.1.0` | Wyoming TTS: piper-tts 1.8.0 on our onnxruntime-gpu 1.31.1 | wyoming streaming |
| `nemotron-unified-asr` | `nemotron-unified-asr:0.1.0` | nvidia/nemotron-labs-unified-asr-0.6b q8_0 GGUF on parakeet.cpp | **true buffered** (0.16–2.08 s) |
| `parakeet-redux-stt` | `parakeet-redux-stt:0.1.0` | moondream/parakeet-redux (ternary 178M) q8_0 GPU + packed CPU | VAD-chunked (opt-in) |
| `phonon2-stt` | `phonon2-stt:0.1.0` | Fermion Phonon-2.5-english q8_0 GPU + packed CPU | VAD-chunked (opt-in) |
| `kokoro-tts` | `kokoro-tts:0.1.0` | Kokoro-82M v1.0 int8 ONNX on onnxruntime-gpu (sm_87+sm_110a) | `create_stream()` |
| `phonon31m-tts` | `phonon31m-tts:0.1.0` | gradium/phonon-31m-en via `ptts` — **CPU-only by design** (307 MB) | **true streaming** (TTFA ~76 ms p95) |
| `paradee-tts` | `paradee-tts:0.1.0` | Paradee-8M-v1.0 int8 ONNX (Kokoro distillation) | sentence-chunked |

## Quick start

```bash
# 1) shared artifact builders (cached; seconds on replay)
make builders

# 2) build everything
make build-all          # or: cd <project> && ./build.sh

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
<project>/            each is standalone: Dockerfile + build.sh + README + test/{verify_arch,smoke_*}
builders/             shared artifact builders (wheels-cu132, parakeet-cpp) — not shipped
bench/                PRD-driven benchmark harness (datasets, engines, probes, metrics)
benchmarks/           history.jsonl + raw run artifacts (committed) — feeds the charts
assets/benchmarks/    generated charts + manifest (committed)
reports/matrix/       generated matrix.csv/md + HTML per task
scripts/              sweep-build.sh / sweep-smoke.sh
docs/                 build & benchmark reports
```

**Decoupling contract:** an engine references shared artifacts only through a single
overridable ARG (`WHEELS_IMAGE`, `PARAKEET_DIST_IMAGE`) — see `ARCHITECTURE.md`.

## Benchmarks

```bash
make bench-prepare      # frozen manifests: 500 STT + 444 TTS sentences (LibriSpeech, seed 42)
make bench-selfcheck    # device, CUDA arch, probes, engines
bench/.venv/bin/python -m bench.cli run bench/configs/thor_stt_baseline.yaml
bench/.venv/bin/python -m bench.cli run bench/configs/thor_tts_baseline.yaml

# matrix + charts + README (reads benchmarks/history.jsonl — never re-runs inference)
make bench-plot
```

The runner **gates every GPU run on co-tenancy**: if other compute apps hold the GPU,
it waits for a clean window and defers rather than measure contaminated. Every run
appends to `benchmarks/history.jsonl`; baseline-class runs also publish their raw
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
