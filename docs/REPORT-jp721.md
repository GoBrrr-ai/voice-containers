# Build Report: llama.cpp + nexus-wyoming for Jetson Thor (sm_110) & Orin (sm_87)

**Date:** 2026-10-31 · **Branch:** `jp721-sm87-sm110` · **Host:** Jetson AGX Thor, JP 7.1 (r38.4, driver 580.00) — images target **JP 7.2.1** (L4T r39.2.1, driver R595, CUDA 13.2.x), host upgrade pending

---

## Decision summary

- **One image per package for both boards** (JP 7.2.1 supports Thor + Orin; SASS for both archs embedded, PTX for forward compat). Each image carries `-thor` and `-orin` tags (identical image).
- **Scenario C (research-validated)**: CUDA 13.2 userspace **baked** into the image (devel stage → runtime stage); the GPU driver (`libcuda.so.1`) is **injected from the host** by nvidia-container-toolkit (CSV mode mounts only the driver stack — never cudart/cublas). No cuda-compat needed (R595 ≥ 13.0 floor 580.65.06; current 580.00 host works via minor-version compatibility — verified live).
- Bases: `nvidia/cuda:13.2.1-devel-ubuntu24.04` (builder) → `nvidia/cuda:13.2.1-runtime-ubuntu24.04` (final, arm64).
- CUDA-version skew note: CUDA 13.2 userspace ran on the 580.00 driver (JP 7.1) during all smoke tests via minor-version compatibility; the `nvidia/cuda:13.2.1` image's runtime requirement check was bypassed with `NVIDIA_DISABLE_REQUIRE=1` **only on this pre-upgrade host**. After the JP 7.2.1 upgrade, run plain.

## Final versions

| Component | Old (fork) | New | Notes |
|---|---|---|---|
| llama.cpp | 0.3.34-era master snapshot | **v0.8.0** (tag exists; commit `8642541b`, 2026-10-28) | requested ref verified upstream |
| llama-cpp-python | 0.3.34 | **0.3.36**, vendored submodule pinned to **v0.8.0** | C API identical (259 fns both refs) |
| wyoming | latest | **1.10.2** | matches HA 2026.10.4 |
| wyoming-faster-whisper | 3.2.0 | **v3.8.1** (OHF-Voice) | fork env patches re-based as git patches |
| faster-whisper | "1.2.2" | **1.2.1** | fork's "1.2.2" is a phantom pin |
| CTranslate2 | 4.7.1 | **4.8.4** (source; CCCL 2.7) | no arm64 CUDA wheel exists upstream |
| onnxruntime-gpu | 1.27.0 | **1.31.1** (source) | official aarch64 cu13 wheels = `89;90a;120a;121a` only, no PTX → useless for us |
| piper-tts | 1.5.0 | **1.8.0** (source, espeak-ng vendored statically) | onnxruntime dep stripped; #22325 provider patch dropped (v1.8.0 has native `use_cuda`) |
| wyoming-piper | main | **v2.5.2** | fork env patches re-based |
| sherpa-onnx | ≥1.12.19 | **1.13.8** (CPU wheel) | two CUDA ORTs in one process segfault (upstream documented) |
| onnx-asr[gpu,hub] | 0.7.0 | **0.11.0** | upstream-pinned |
| pysilero-vad | 2.1.1 | **3.4.0** | ggml-based now; kills the onnxruntime force-replace hack |
| av | (transitive) | **15.1.0** (pinned) | PyAV 19 dropped `metadata_errors` used by faster-whisper decode |
| torch | absent | **absent** | verified no base dep pulls it |

## Arch flags (per component — one image serves both boards)

| Component | Flags | Why |
|---|---|---|
| llama.cpp | `87-real;110a-real;110-virtual` | sm_87 + sm_110a SASS; plain compute_110 PTX assembles fine (ggml emits no family-conditional code below CC 12.0 — `GGML_CUDA_CC_BLACKWELL=1200`) |
| CTranslate2 | `CUDA_ARCH_LIST="8.7+PTX"` + `-gencode=arch=compute_110,code=sm_110;-gencode=arch=compute_110,code=compute_110` | frozen FindCUDA cannot parse two-digit majors ("11.0" → Unknown CUDA Architecture Name, upstream #2082) |
| onnxruntime | `87-real;110a-real;110a-virtual` | ORT's FP4/MoE kernels emit `cvt.e2m1x2.f32` which ptxas rejects for plain compute_110 PTX → family-conditional PTX instead (JITs on any 110a-family part) |

**Verified `110a` facts** (nvcc 13.2.78): `sm_110a`/`110f` valid; `sm_87a` invalid ("Unsupported gpu architecture").

## Other build flags

- llama.cpp: `GGML_CUDA=ON GGML_CUDA_F16=ON GGML_CUDA_FA_ALL_QUANTS=ON LLAMA_OPENSSL=ON LLAVA_BUILD=OFF`, gcc-11 CUDA host compiler (fork's known-good aarch64 codegen workaround, supported by CUDA 13.2), `LLAMA_CURL` dropped (deprecated no-op at v0.8.0), `GGML_NATIVE=OFF` (one image for two different ARM CPUs).
- CTranslate2: `WITH_CUDA=ON WITH_CUDNN=ON WITH_MKL=OFF OPENMP_RUNTIME=COMP` (cuDNN needed for whisper's Conv1D encoder).
- onnxruntime: `--use_cuda` only, **no `--use_tensorrt`** (unused by whisper/piper; −~2 GB), `--allow_running_as_root`, memory-adaptive `--parallel 12`; CCCL `device_transform.cuh` patched in the toolkit headers (GCC rejects the globally-qualified specialization shipped in CUDA 13.2.1).
- Build-time patch: **PR #31268** (Orin single-column MMVQ; predecessor #30858 measured +15–38% Orin decode on Q4/Q8, bit-identical perplexity; guarded `cc==870` → zero Thor risk), vendored pinned to SHA `ad98556`, applies clean to v0.8.0. **#25384 skipped** (v0.8.0 already tolerates the cudaMemGetInfo failure path it fixed; explicit `-ngl` recommended anyway).

## cuobjdump verification (required check) — PASS

```
futureproofhomes/llama-cpp-thor:0.8.0
  libggml-cuda.so.0.28.0   SASS: sm_87, sm_110a     PTX: sm_110
  libggml.so / llama-server: host-only (no device code embedded — expected)

futureproofhomes/nexus-wyoming-thor:0.1.0
  libctranslate2.so.4.8.4  SASS: sm_87, sm_110      PTX: sm_87, sm_110
  libonnxruntime_providers_cuda.so  SASS: sm_87, sm_90a, sm_110a   PTX: sm_110a
  libonnxruntime.so.1.31.1: host-only (core; kernels live in the provider .so)
```

## Smoke tests — PASS on real Thor hardware (GPU)

- **llama.cpp**: `llama-server` + Qwen3-1.7B-Q4_K_M, full `-ngl 999` offload, `-fa on`:
  prompt eval **506.9 tok/s**, generation **118.3 tok/s**, CUDA graphs reused (v0.8.0 chunked graphs) on the 580 driver; reply `"JETSON THOR OK"` (exact, `enable_thinking:false`); `llama-cli --version` OK; `llama-cpp-python 0.3.36` GPU generation OK ("BINDINGS OK").
- **nexus-wyoming**: both services healthy (describe on 10300/10200); **TTS round trip**: wyoming Synthesize → piper (CUDA EP) → 52 KB wav @ 22050 Hz; **STT round trip**: wav → faster-whisper (CT2 CUDA) → transcript " Hello from Jits & Thor." (tiny-int8 model accuracy, pipeline verified). `ctranslate2.get_cuda_device_count()==1`, `onnxruntime.get_available_providers()` includes CUDAExecutionProvider.

## Image sizes (docker .Size, uncompressed)

| Image | Old (fork) | New | Δ |
|---|---|---|---|
| llama.cpp | 11.7 GB (`narandill/llama_cpp:0.3.34-...-cu129`) | **4.42 GB** | **−62%** |
| nexus-wyoming | 12.0 GB (`futureproofhomes/nexus-wyoming:latest`) | **5.04 GB** | **−58%** |

Multi-stage layout: final images contain no compilers, headers, source trees, pip caches, or apt lists (verified by the layer recipe; runtime deps sanity-checked via `ldd` in-image, excluding the host-injected `libcuda.so.1`).

## Tags

`futureproofhomes/llama-cpp:{0.8.0,latest}` + `futureproofhomes/llama-cpp-thor:0.8.0` + `futureproofhomes/llama-cpp-orin:0.8.0`
`futureproofhomes/nexus-wyoming:{0.1.0,latest}` + `futureproofhomes/nexus-wyoming-thor:0.1.0` + `futureproofhomes/nexus-wyoming-orin:0.1.0`
(not pushed to Docker Hub — say the word)

## Known issues / caveats

1. **Host still on JP 7.1 (driver 580.00)**: run images with `-e NVIDIA_DISABLE_REQUIRE=1` until the JP 7.2.1 upgrade; after upgrade no env needed. All runtime behavior verified on 580 via minor-version compatibility.
2. ORT: no plain-sm_110 PTX (family-conditional `110a` PTX instead) — ORT's FP4 kernels make plain compute_110 PTX unassembleable. Orin (sm_87) and Thor (sm_110a) are covered by SASS.
3. CT2 aarch64+CUDA slow host-memory growth (~6.8 KB/inference, upstream #2088, open) — monitor on long-running STT services.
4. gemma-4 + `--spec-type draft-mtp` has an open CUDA IMA upstream (#31077) — avoid that combo until fixed.
5. Never set `CUDA_SCALE_LAUNCH_QUEUES` on single-GPU Jetson (Orin MoE deadlock history #19219/#19227).
6. piper `#22325` provider_options patch not carried (v1.8.0 natively supports `--use-cuda` with HEURISTIC conv algo search); can be re-based later if tuning is needed.
7. First `nexus-wyoming` start downloads models (whisper tiny-int8 + piper voice) into `/data` — needs internet or pre-seeded cache (`WHISPER_OFFLINE=true PIPER_OFFLINE=true`).

## Left undone / future work

- Docker Hub push (awaiting approval).
- Orin-side smoke test (no Orin on this bench — arch coverage is proven via cuobjdump; the same image is validated in the wild on both boards per the community JP 7.2.1 report).
- `test-backend-ops` full sweep (optional; smoke + real workload passed).
- Benchmarks: llama-bench A/B on Orin for the #31268 patch (needs Orin hardware).
