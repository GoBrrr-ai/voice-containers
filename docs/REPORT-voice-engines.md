# Wave 2 Report — Streaming STT + TTS Engine Images (Jetson Thor sm_110 + Orin sm_87)

**Date:** 2026-10-31 · **Branch:** `jp721-sm87-sm110` · Companion to wave 1 (`REPORT-jp721.md`): bases, toolchain and the ORT/CT2 builds are shared.

---

## Deliverables — 6 engines, one image per package, both boards

| Image | Tags | Size | Engine / runtime | Streaming |
|---|---|---|---|---|
| `narandill/nemotron-unified-asr` | `0.1.0`,`latest` + `-thor`/`-orin` | **3.99 GB** | nvidia/nemotron-labs-unified-asr-0.6b q8_0 GGUF (895 MB) on parakeet.cpp | **true buffered streaming** 0.16–2.08 s presets (token-identical to NeMo per upstream parity) |
| `narandill/parakeet-redux-stt` | same | **4.21 GB** | moondream/parakeet-redux q8_0 (GPU) + packed (CPU dotprod) + silero-vad | VAD-chunked (offline TDT) |
| `narandill/phonon2-stt` | same | **4.24 GB** | Fermion Phonon-2.5-english q8_0 + packed + silero-vad | VAD-chunked (offline TDT) |
| `narandill/kokoro-tts` | same | **4.87 GB** | Kokoro-82M v1.0 **int8 ONNX** (169 MB) on our ORT-GPU 1.31.1 | `create_stream()` phoneme-chunked async gen (TTFA 0.7–1.1 s measured) |
| `narandill/phonon31m-tts` | same | **307 MB** | gradium/phonon-31m-en q8 (61.5 MB) via `ptts` 0.6.0 — **CPU-only, no CUDA** | **true token streaming**, TTFA 172 ms measured (LB #1: 28 ms p95) |
| `narandill/paradee-tts` | same | **5.03 GB** | Paradee-8M-v1.0 int8 ONNX (8.6 MB) on our ORT-GPU | sentence-chunked wrapper (graph is whole-utterance) |

All: `test/verify_arch.sh` + `test/smoke_*.sh` per project; entrypoint wrapper (OpenAI-compatible `POST /v1/audio/transcriptions` for STT; HTTP chunked-stream `/say` for phonon31m) + CLI helpers (`stt-transcribe`, `kokoro-say`, `ptts-say`, `paradee-say`, `paradee-stream`).

## Architecture — no jetson-containers builder system

- Plain `docker build` per project (`build.sh`); no `#---` package headers, no `packages/`, no JC CLI (verified by grep).
- **Shared artifacts, each referenced via a single overridable ARG** (`PARAKEET_DIST_IMAGE`, `ORT_WHEEL_IMAGE`) — the future standalone-repo migration is a one-line change per engine (see `projects/ARCHITECTURE.md` dependency graph).
- `narandill/edge-builder:parakeet-cpp` (intermediate): parakeet.cpp @ pinned `2eac2e77` + vendored ggml, gcc-11 host compiler, `87-real;110a-real;110-virtual`, carries our upstream-bound MelKernel fix (below).

## Verification matrix — all on Thor GPU

| Check | nemotron | redux | phonon2 | kokoro | phonon31m | paradee |
|---|---|---|---|---|---|---|
| cuobjdump sm_87+sm_110 SASS + PTX | ✅¹ | ✅¹ | ✅¹ | ✅² | n/a CPU | ✅² |
| Batch GPU inference | ✅ | ✅ | ✅ | ✅³ | ✅³ | ✅³ |
| Streaming (GPU) | ✅ (fixed, see §bug) | ✅ (VAD) | ✅ (VAD) | ✅ create_stream | ✅ tts.stream | ✅ sentence-chunk |
| CPU fallback | ✅ | ✅ | ✅ | ✅ | — (CPU-only) | ✅ |
| Server round-trip | ✅ | ✅ | ✅ | n/a (library) | ✅ | n/a (library) |

¹ `libggml-cuda.so`: sm_87+sm_110a SASS, compute_110 PTX · ² `libonnxruntime_providers_cuda.so`: sm_87+sm_90a+sm_110a SASS, sm_110a PTX · ³ CUDA EP confirmed live (`providers` print).

Transcripts round-trip: TTS wav → STT → `"Hello from Jits and Thor."` (model hearing of "Jetson"; pipeline exact).

## The MelKernel bug — found, root-caused, fixed, upstream-bound

**Symptom:** nemotron `--stream` on GPU died after backend init — appeared as a hang under the 99 GB vLLM co-tenant, as a **deterministic SIGSEGV** (rc 139) on a free GPU.

**Root cause** (gdb + objdump on the exact faulting instruction, `MelKernel::MelKernel+388` = `ldr x3, [x25, #16]` = `w->ne[0]`):
`BufferedStreamingSession` constructs `MelKernel` **before** weights are realized; the ctor captures `ggml_tensor* w = ml.tensor(...)`, then `weight_to_host_f32(...)` triggers the one-time `ensure_weights_realized()` ("realize_weights **mutates tensor->buffer**" per ggml's own comment) — the captured `w` dangles → SEGV. Offline paths realize earlier → never crash. CPU streaming → no device buffers to realize → fine.

**Fix** (`projects/parakeet-cpp-builder/files/mel-kernel-dangling-tensor.patch`): `const int wlen = (int)wbuf.size();` — the window tensor is 1-D, so the fetched element count *is* the length; no dangling deref. CUDA graphs were exonerated (graphs stay ON, ~1% perf kept; `GGML_CUDA_DISABLE_GRAPHS`/`GGML_CUDA_PDL=0` tested and **not** needed).

**Upstream:** no issue exists as of 2026-10-31 (their unified-streaming parity tests are CPU-only; the streaming path landed in #109 one day before our pin). Draft report text + evidence in `task/research/parakeet-gpu-stream-hang/summary.md`.

## Debugging lessons baked into the smoke harness

- GPU guard now **coexists with co-tenants** (waits only under ~100 GB VRAM pressure; never kills — your vLLM experiments survived untouched throughout).
- Server probe: HTTP GET (a bodyless POST wedges the single-slot server; a raw-TCP probe false-positives through docker's userland proxy; and `curl -w` already emits `000` on failure — appending another broke the comparison under `set -e`).
- `parakeet-cli --stream-latency` takes **seconds** (0.56, not 560); `--json` + `--stream` rejected by design.

## Known issues / notes

1. **nemotron `--stream-latency` env is in seconds** (`PARAKEET_STREAM_LATENCY=0.56` default; presets 0.16/0.24/0.32/0.56/1.12/2.08).
2. kokoro `uint8` graph logs a harmless ORT `ScatterND` warning; TTFA variance 0.7–1.1 s (first-run graph capture).
3. paradee prints an HF "unauthenticated" warning at runtime (etag check; model + dict are baked in `/opt/hf-cache` — offline-safe).
4. paradee CUDA EP verified on dynamic input shapes; CPU EP is the guaranteed fallback (18× RT on one thread).
5. Image sizes include the shared 1.8 GB CUDA runtime base + cuDNN (ORT CUDA EP requires it); phonon31m proves the floor: **307 MB**.
6. First STT-server request lazy-loads the model (~seconds); subsequent requests are fast.

## Not done / next

- Docker Hub push (awaiting approval, as always).
- File the MelKernel issue upstream (patch + evidence ready).
- Orin hardware validation (arch coverage proven via cuobjdump; no Orin on this bench).
- Wave-3 candidate: wire the winners into the actual duplex assistant (nemotron GGUF streaming + phonon31m are the ultra-light stack; ~4 GB total with a 4B LLM).
