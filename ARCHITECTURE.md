# Build architecture (JP 7.2.1, sm_87 + sm_110)

**Status:** 2026-11-07 · 16 engine images (`scripts/projects.txt`) from 2
shared builders. This repo is the single source of truth.

## Rules

1. **No external builder framework.** No `#---` YAML Dockerfile headers and
   no builder CLI anywhere — every project is a plain `docker build` via its
   own `build.sh`.
2. **Shared dependencies are allowed** but each must be a **self-contained
   builder project** under `builders/`, producing tagged artifact images.
3. **Decoupling contract:** an engine image may reference shared artifacts
   only through a **single overridable `ARG` per artifact** — so decoupling is
   a one-line change (override the ARG, vendor the file, or inline a builder
   stage). No project may hard-code another project's build stage.

## Dependency graph

```
builders/parakeet-cpp  →  narandill/edge-builder:parakeet-cpp
│   parakeet.cpp GGML dist (parakeet-cli/-server + libggml*), pinned ref 2eac2e77,
│   sm_87+sm_110a SASS + compute_110 PTX, MelKernel GPU-stream fix
│   consumed via ARG PARAKEET_DIST_IMAGE — the GGUF quartet:
│   ├── nemotron-unified-asr   (bakes nemotron-labs-unified-asr-0.6b-q8_0.gguf)
│   ├── parakeet-redux-stt     (bakes redux-q8_0 + redux-packed)
│   ├── parakeet-tdt-cpp-stt   (bakes parakeet-tdt-0.6b-v3-q4_k)
│   └── phonon2-stt            (bakes phonon-2.5-english q8_0 + packed)
│
builders/wheels-cu132  →  narandill/edge-wheels:cu132
│   onnxruntime-gpu 1.31.1 (sm_87+sm_90a+sm_110a SASS, sm_110a PTX)
│   + ctranslate2 4.8.4 + wyoming forks — 75-min source build, do NOT rebuild casually
│   consumed via ARG WHEELS_IMAGE / ORT_WHEEL_IMAGE — nine images:
│   ├── wyoming-faster-whisper   (whisper via wyoming, HA-native protocol)
│   ├── wyoming-piper1-tts       (piper via wyoming; no-tempfile fork patch)
│   ├── parakeet-tdt-stt         (CTranslate2 int8)
│   ├── parakeet-tdt-ort-stt     (onnxruntime int4)
│   ├── moonshine-streaming-stt  (CPU bundle + CUDA ORT variant, auto-selected)
│   ├── nemotron-streaming-stt   (sherpa-onnx streaming, 160 ms profile)
│   ├── canary-flash-stt         (int8, en/de/es/fr + x↔en translation)
│   ├── kokoro-tts               (kokoro-onnx; wheel force-installed LAST)
│   └── paradee-tts              (paradee pkg; wheel force-installed LAST)
│
standalone (no shared deps):
    ├── llama-cpp        own devel→runtime stages, cuda 13.2, GGML_CUDA flags pinned
    ├── phonon31m-tts    plain ubuntu:24.04 + ptts aarch64 wheel (CPU by design)
    └── pocket-tts       plain ubuntu:24.04 (CPU/CUDA auto)
```

Every engine image carries one version tag + `latest`; both boards are served
by the **same image** — sm_87 and sm_110(a) cubins baked side by side.

The HTTP surface of every non-wyoming voice image is the shared contract in
[`docs/voice-api.md`](docs/voice-api.md) (OpenAI-compatible; wyoming pair is
HA-native and out of scope).

## Decoupling recipes (standalone-repo migration)

| Today | At migration |
|---|---|
| `ARG PARAKEET_DIST_IMAGE=narandill/edge-builder:parakeet-cpp` | DONE — `builders/parakeet-cpp/` (pinned ref, carries the MelKernel GPU-stream fix). To fully vendor: `COPY --from=builder /opt/parakeet-dist …` inline builder stage (identical stages across the 4 GGUF Dockerfiles share BuildKit cache). |
| `ARG WHEELS_IMAGE=narandill/edge-wheels:cu132` | DONE — `builders/wheels-cu132/` replicates the original instruction stream byte-identically (BuildKit cache hit, no recompile). To fully vendor: copy the wheels out of the built image and `COPY` them. |
| `nvidia/cuda:13.2.1-{devel,runtime}-ubuntu24.04` bases | Keep (upstream images) |

## Verification gates (every CUDA image)

- `test/verify_arch.sh <image>` — cuobjdump: REQUIRE sm_87 + sm_110(/a) SASS
  and sm_110-family PTX in the shared libs.
- `test/smoke_*.sh <image>` — real-GPU smoke on Thor (waits for a free GPU,
  never kills occupants; `-e NVIDIA_DISABLE_REQUIRE=1` only until the JP 7.2.1
  host upgrade).

## Build order

1. `builders/parakeet-cpp` (once; ~7 min) — cached afterwards
2. the GGUF quartet (each ~2–4 min: model download + runtime assembly)
3. the nine wheels consumers (fast; builder image cached)
4. `builders/wheels-cu132` — only when the wheel set changes (~75 min source build)
5. the standalone trio any time (`llama-cpp` compiles ~15–20 min; the two
   ubuntu-based TTS images are minutes)

## CI

| workflow | trigger | runner | notes |
|---|---|---|---|
| `lint.yml` / `tests.yml` | push/PR | hosted | pre-commit; pytest (no GPU) |
| `build-smoke.yml` | manual dispatch | self-hosted jetson | build + optional GPU smoke, any project subset |
| `publish.yml` | tag `v*` | self-hosted jetson | build → arch gate → push `narandill/*:<ver>`; Hub metadata via `scripts/docker_release.py` |
