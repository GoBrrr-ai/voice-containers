# GPU / CPU backend matrix — Jetson sm_87 + sm_110 (CUDA 13.2, JP 7.2.1)

Deployment hardware has **shared unified memory**: GPU allocations draw from
the same pool as the CPU, so the bench's `cgroup MiB` column is the budget
metric regardless of backend. Policy: **GPU-first** — every package ships
native sm_87 + sm_110a cubins (no JIT) except the one CPU-by-design engine.
Mission 2026-11-02: "all packages on GPU, native for both archs; re-implement
whatever blocks." Status per package below; perf numbers live in the README
matrix (the bench decides GPU-vs-CPU per utterance, honestly).

## Where the native cubins live (verified with cuobjdump)

| Artifact | SASS | PTX |
|---|---|---|
| our ORT 1.31.1 `providers_cuda.so` (builders/wheels-cu132) | 316× sm_87 + 316× sm_110a + 292× sm_90a | none (deliberate: cubins cover the exact fleet) |
| CTranslate2 4.8.4 (`libctranslate2.so.4.8.4`) | 34× sm_87 + 34× sm_110 | none |
| parakeet.cpp `libggml-cuda.so` (builders/parakeet-cpp) | sm_87 + sm_110a | none |
| NVIDIA torch 2.15.0+cu132 aarch64 wheel | sm_110 native (Thor); **sm_80 cubins run natively on Orin sm_87** via within-major 8.0→8.7 binary compat — no JIT anywhere | none in wheel |

PTX note: with the cubin set covering exactly {Orin sm_87, Thor sm_110a},
PTX is dead weight — and worse for TTFA (driver JIT on first load, cache
invalidated by driver updates). The only PTX shipped is ORT's
`110a-virtual`, kept as free forward-compat insurance for future
Thor-family silicon (plain compute_110 is rejected by ptxas for the FP4/MoE
kernels).

## Speech recognition (12 configs)

| Image | GPU | CPU | Notes |
|---|---|---|---|
| `nemotron-unified-asr` 0.6B q8_0 (parakeet.cpp) | ✅ native | (capable — GGUF) | |
| `parakeet-redux-stt` q8_0 + packed (parakeet.cpp) | ✅ native | ✅ | |
| `phonon2-stt` q8_0 + packed (parakeet.cpp) | ✅ native | (capable) | |
| `wyoming-faster-whisper` tiny-int8 (CT2) | ✅ native | (capable) | |
| `canary-flash-stt` 0.2.0 (sherpa GPU wheel) | ✅ int8 on CUDA EP | ✅ | one image, `provider=cpu|cuda`; int8-QDQ dequant overhead can make GPU slower on tiny clips |
| `parakeet-tdt-stt` 0.2.0 (sherpa GPU wheel) | ✅ int8 on CUDA EP | ✅ | maintainer-verified int8+CUDA (k2-fsa #2282); fp32 likely faster on GPU — follow-up |
| `nemotron-streaming-stt` 0.2.0 (sherpa GPU wheel) | ✅ int8 on CUDA EP | ✅ | streaming transducer on CUDA: verified upstream via streaming-zipformer #3620 + GPU-CI nemo-fastconformer smoke |
| `moonshine-streaming-stt` 0.2.0 | ✅ fp32 5-graph on CUDA EP (mixed: frontend on CPU) | ✅ two paths | official .ort lib is CPU-only by design (fused com.microsoft ops); GPU = Masterx fp32 export on our ORT (driver in rootfs). int8 graphs unusable on GPU: `DynamicQuantizeLinear` has no CUDA kernel |

**Why this was blocked before**: sherpa-onnx has NO CUDA kernels of its own —
GPU means linking against a CUDA-enabled ORT
(`task/research/sherpa-onnx-cuda-aarch64/`). Upstream ships no aarch64
CUDA-13 artifact (the official `+cuda` aarch64 wheel bundles ORT
1.11.0/CUDA 10.2). Fix: `builders/wheels-cu132` now builds
`sherpa_onnx-2.0.1+cuda` against our ORT via the maintainer-endorsed
`SHERPA_ONNXRUNTIME_INCLUDE_DIR/LIB_DIR` escape hatch (k2-fsa #2046).

## Speech synthesis (6 configs)

| Image | GPU | CPU | Notes |
|---|---|---|---|
| `kokoro-tts` int8 ONNX | ✅ native | (capable) | |
| `paradee-tts` v1.2 int8 ONNX | ✅ native | ✅ | |
| `wyoming-piper1-tts` ONNX | ✅ native | (capable) | |
| `pocket-tts` 0.2.0 (torch cu132) | ✅ sm_110 native (Thor) / sm_80-cubin compat (Orin) | ✅ | measured RTF **0.13 GPU** vs 0.39 CPU on Thor; ubuntu base needs `NVIDIA_VISIBLE_DEVICES=all` or the nvidia runtime injects no libcuda |
| `phonon31m-tts` (ptts) | ❌ **by design** | ✅ (26× RT) | the ptts Rust engine has no GPU backend — the mission's single sanctioned CPU-only exception |

## Runtime pitfalls (all hit & fixed 2026-11-02)

- **ubuntu-base GPU images need `NVIDIA_VISIBLE_DEVICES=all` +
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility`** — without them the nvidia
  runtime injects no libcuda ("Found no NVIDIA driver"). nvidia/cuda bases
  bake these in.
- **The nvidia/cuda base's `requirements.nvidia` brand list has no Thor
  entry** — any docker run on this host needs `-e NVIDIA_DISABLE_REQUIRE=1`
  unless the image sets it (repo-wide smoke convention:
  `RUNTIME=(--runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1)`).
- **Never CUDA-EP-test inside the wheels BUILDER image**: its stub
  `libcuda.so.1` (builder-only link trick) shadows the injected driver →
  "CUDA driver is a stub library".
- **sherpa + moonshine frontend on CUDA**: the first frontend call reduces
  an empty carried-sample buffer (`sample_len=0`); CUDA EP rejects it
  (`PrepareForReduce: can't reduce on dim 0`), CPU EP is fine → moonshine
  runs a MIXED-EP pipeline (frontend CPU, 4 heavy graphs CUDA).
- **fp16 ONNX is the broken quant on CUDA EP** (empty outputs, k2-fsa
  #2282) — int8 (QDQ/QOperator) and fp32 work; fp32 often fastest on GPU
  since int8-QDQ dequantizes anyway.

## Orin Nano 8 GB caveat

The torch cu132 wheel covers Orin via sm_80-cubin binary compat (verified
gencode list, not yet on-device). On Orin, CPU-first remains the deployment
thesis (Option A) — the GPU roster above is for Thor-first deployment and
any future Orin GPU probing.
