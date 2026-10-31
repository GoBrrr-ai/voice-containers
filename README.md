# edge-containers

Production container images for **NVIDIA Jetson AGX Thor (sm_110)** and **AGX Orin (sm_87)** —
an LLM runtime, a full streaming voice stack (STT + TTS), and a PRD-driven benchmark harness.
One fat image per package serves **both boards** (native cubins for sm_87 + sm_110a plus
forward-compat PTX), built on CUDA 13.2 SBSA / JetPack 7.2.1 userspace with the GPU driver
injected from the host.

> Status: **local only** — nothing is on Docker Hub or GitHub yet. The publish workflow
> activates when this repo is pushed and tagged.

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
| `phonon31m-tts` | `phonon31m-tts:0.1.0` | gradium/phonon-31m-en via `ptts` — **CPU-only by design** (307 MB) | **true streaming** (TTFA ~28 ms p95) |
| `paradee-tts` | `paradee-tts:0.1.0` | Paradee-8M-v1.0 int8 ONNX (Kokoro distillation) | sentence-chunked |

`-thor` / `-orin` tags point at the same image as the version tag (both cubin sets are baked).

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
docker run --runtime nvidia --rm -p 10300:10300 -v /data/wfw:/data narandill/wyoming-faster-whisper-thor:0.1.0
docker run --runtime nvidia --rm --entrypoint stt-transcribe narandill/nemotron-unified-asr-thor:0.1.0 /audio/x.wav
```

On pre-7.2.1 hosts add `-e NVIDIA_DISABLE_REQUIRE=1` to every `docker run` (driver
580.00 vs the baked 13.2 userspace — minor-version compatibility). After the host
upgrade, drop the flag.

## Layout

```
<project>/            each is standalone: Dockerfile + build.sh + README + test/{verify_arch,smoke_*}
builders/             shared artifact builders (wheels-cu132, parakeet-cpp) — not shipped
bench/                PRD-driven benchmark harness (datasets, engines, probes, metrics)
scripts/              sweep-build.sh / sweep-smoke.sh
docs/                 build reports (wave 1: llama.cpp + wyoming; wave 2: the 6 voice engines)
```

**Decoupling contract:** an engine references shared artifacts only through a single
overridable ARG (`WHEELS_IMAGE`, `PARAKEET_DIST_IMAGE`) — see `ARCHITECTURE.md`.

## Benchmarks

```bash
make bench-prepare      # LibriSpeech test-clean (500 utts) + LibriTTS-R text (500 sent), seed 42
make bench-selfcheck    # device, CUDA arch, probes, engines
python -m bench.cli run --config bench/configs/thor_stt_baseline.yaml
python -m bench.cli report results/<run_id>
```

See `bench/README.md` and the PRD for the full metric list (WER/CER, RTFx, TTFA,
finalization latency, memory, energy via tegrastats, gate checks, Pareto front).

## CI

| Workflow | Trigger | Runner | What it does |
|---|---|---|---|
| `lint.yml` | push/PR | `ubuntu-latest` | pre-commit (shellcheck, hadolint, ruff, yaml/json checks) |
| `build-smoke.yml` | `workflow_dispatch` | `[self-hosted, jetson]` | build matrix + arch verification (+ optional GPU smoke) |
| `publish.yml` | tag `v*` | `[self-hosted, jetson]` | build + push all `narandill/*` tags (needs `DOCKERHUB_USER`/`DOCKERHUB_TOKEN` secrets) |

## Licenses

Build recipes are MIT. Each image carries its engine's license — check every project's
README (all engines here are permissive or OpenMDW; nothing NC).
