# Build architecture (JP 7.2.1 fleet, sm_87 + sm_110)

**Status:** 2026-10-31 · This repo is the single source of truth (migrated from the jetson-containers working branch).

## Rules

1. **No jetson-containers builder system.** No `#---` YAML Dockerfile headers,
   nothing under `packages/`, and no `jetson-containers build` CLI anywhere —
   every project is a plain `docker build` via its own `build.sh`.
2. **Shared dependencies are allowed** but each must be a **self-contained
   builder project** under `builders/`, producing tagged artifact images.
3. **Decoupling contract:** an engine image may reference shared artifacts
   only through a **single overridable `ARG` per artifact** — so decoupling is
   a one-line change (override the ARG, vendor the file, or inline a builder
   stage). No project may hard-code another project's build stage.

## Dependency graph

```
(builders/)
├── parakeet-cpp/           → narandill/edge-builder:parakeet-cpp
│     (parakeet.cpp GGML dist: parakeet-cli/-server + libggml*,
│      sm_87+sm_110a SASS + compute_110 PTX; pinned ref 2eac2e77)
│        ↑ ARG PARAKEET_DIST_IMAGE
│        ├── nemotron-unified-asr  (bakes nemotron-labs-unified-asr-0.6b-q8_0.gguf)
│        ├── parakeet-redux-stt    (bakes redux-q8_0 + redux-packed)
│        └── phonon2-stt           (bakes phonon-2.5-english q8_0 + packed)
│
├── wheels builder (ex combined wyoming) → narandill/edge-wheels:cu132
│     (onnxruntime-gpu 1.31.1 wheel: sm_87+sm_90a+sm_110a SASS, sm_110a PTX
│      + ctranslate2 4.8.4 — 75-min source build, do NOT rebuild casually)
│        ↑ ARG ORT_WHEEL_IMAGE
│        ├── kokoro-tts   (kokoro-onnx 0.6.1 --no-deps, wheel force-installed LAST)
│        └── paradee-tts  (paradee pkg, wheel force-installed LAST)
│
└── phonon31m-tts   (NO shared deps: plain ubuntu:24.04 + ptts aarch64 wheel)
```

Every engine image carries one version tag + `latest` (both boards are served
by the same image — sm_87 and sm_110 cubins are baked side by side).

## Decoupling recipes (standalone-repo migration)

| Today | At migration |
|---|---|
| `ARG PARAKEET_DIST_IMAGE=narandill/edge-builder:parakeet-cpp` | DONE — `builders/parakeet-cpp/` (pinned ref, carries the MelKernel GPU-stream fix). To fully vendor: `COPY --from=builder /opt/parakeet-dist …` inline builder stage (identical stages across the 3 STT Dockerfiles share BuildKit cache). |
| `ARG WHEELS_IMAGE=narandill/edge-wheels:cu132` | DONE — `builders/wheels-cu132/` replicates the original instruction stream byte-identically (BuildKit cache hit, no recompile). To fully vendor: copy the wheels out of the built image and `COPY` them. |
| `nvidia/cuda:13.2.1-{devel,runtime}-ubuntu24.04` bases | Keep (upstream images, not JC) |

## Verification gates (every CUDA image)

- `test/verify_arch.sh <image>` — cuobjdump: REQUIRE sm_87 + sm_110(/a) SASS
  and sm_110-family PTX in the shared libs.
- `test/smoke_*.sh <image>` — real-GPU smoke on Thor (waits for a free GPU,
  never kills occupants; `-e NVIDIA_DISABLE_REQUIRE=1` only until the JP 7.2.1
  host upgrade).

## Build order

1. `builders/parakeet-cpp` (once; ~7 min) — done
2. STT trio (each ~2–4 min: models download + runtime assembly; builder cached)
3. TTS trio (kokoro/paradee reuse the existing cu132 wheel image — no rebuild)
4. `builders/wheels-cu132` (only if the wheel set changes; ~75 min source build)

Consumers of `narandill/edge-wheels:cu132` also include the two wyoming
projects (`wyoming-faster-whisper`, `wyoming-piper1-tts`) and `llama-cpp`
stands alone (own builder stage).
