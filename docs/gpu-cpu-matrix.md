# GPU / CPU backend matrix — Jetson sm_87 + sm_110 (CUDA 13.2, JP 7.2.1)

Deployment hardware has **shared unified memory**: GPU allocations draw from
the same pool as the CPU, so the bench's `cgroup MiB` column is the budget
metric regardless of backend. Policy: **GPU-first wherever a JP 7.2.1 wheel
exists**; CPU-only engines are listed with the exact blocker that keeps
them off the GPU.

Status as of 2026-11-02 (bench run `20261102_122832` STT / `20261102_143219`
TTS). ✅ = measured in the committed matrix; (capable) = works but not benched.

## Speech recognition (9 configs)

| Image | GPU | CPU | GPU blocker (if any) |
|---|---|---|---|
| `nemotron-unified-asr` 0.6B q8_0 (parakeet.cpp) | ✅ | (capable — GGUF runs on CPU) | — |
| `parakeet-redux-stt` q8_0 + packed (parakeet.cpp) | ✅ | ✅ (packed ternary, ultra-light) | — |
| `phonon2-stt` q8_0 + packed (parakeet.cpp) | ✅ | (capable — packed gguf is baked) | — |
| `wyoming-faster-whisper` tiny-int8 (CT2) | ✅ | (capable — CT2 CPU) | — |
| `moonshine-streaming-stt` 245M int8 | ❌ | ✅ | **Wheel gap**: official `moonshine-voice` bundles a CPU-only ORT. GPU route: the community five-graph ONNX export (`Masterx/...`) on our ORT-GPU wheel — untested; frontend graph is fp32-only. |
| `canary-flash-stt` 180M int8 (sherpa) | ❌ | ✅ | **Two blockers**: (1) sherpa-onnx has no aarch64 CUDA-13 build; (2) the ONNX exports ship no featurizer graph — the NeMo 128-bin mel contract must be hand-implemented before our ORT-GPU wheel can serve the soniqo/istupakov graphs. |
| `parakeet-tdt-stt` 0.6B v3 int8 (sherpa) | ❌ | ✅ | sherpa-onnx GPU build missing (see above). **Cheapest GPU win in the repo**: a GGUF exists (`mudler/parakeet-cpp-gguf`: f16 1.44 GB, q8_0 941 MB, **q6_k 813 MB — fits the 0.8 GB budget**) → a parakeet.cpp variant like redux/phonon2 is a config-only change away. |
| `nemotron-streaming-stt` 0.6B int8 (sherpa) | ❌ | ✅ | **Hardest**: no GGUF (parakeet.cpp does not implement this cache-aware RNNT), the FP32 export is 2.2 GB, and sherpa-onnx has no aarch64 CUDA-13 build. |

## Speech synthesis (6 configs)

| Image | GPU | CPU | GPU blocker (if any) |
|---|---|---|---|
| `kokoro-tts` int8 ONNX | ✅ | (capable) | — |
| `paradee-tts` v1.2 int8 ONNX | ✅ (RTF p50 0.011 — fastest synth) | ✅ | — |
| `wyoming-piper1-tts` ONNX | ✅ | (capable) | — |
| `phonon31m-tts` (ptts) | ❌ **by design** | ✅ (26× RT measured) | Not a wheel gap: the `ptts` Rust engine has no GPU backend at all; it is CPU-optimized (TTFA champion). |
| `pocket-tts` 100M (torch) | ❌ (v1) | ✅ (RTF p50 0.39) | **Wheel exists — test missing**: NVIDIA's `torch 2.15.0+cu132` aarch64 wheel gencodes sm_80/90/100/110/120 → **Thor sm_110 ✓, Orin sm_87 ✗** (`cudaErrorNoKernelImageForDevice`). Whether the package moves tensors to CUDA is unverified (it advertises CPU-only). oneDNN must stay **off** either way (v3.13.3 aborts on Neoverse V3AE; on 2.14.1 it is 2.7× slower than the reference+NVPL path). |

## Upgrade paths, ranked by effort ÷ value

1. **parakeet-tdt GGUF GPU variant** — config-only; reuses the parakeet.cpp
   adapter/ids; q6_k fits the Orin budget (rides E3's quant testing).
2. **pocket-tts GPU on Thor** — the wheel is already in the image; one
   device-support probe decides it. Orin stays CPU (no sm_87 gencode).
3. **canary GPU via soniqo/istupakov graphs + our ORT-GPU wheel** — needs
   the mel featurizer implemented (contract fully spelled in the export's
   `config.json`); then Canary FP16 ≈ 0.4 GB is the report's best GPU ASR.
4. **moonshine GPU via the five-graph export** — medium effort, untested
   upstream on CUDA EP.
5. **nemotron-streaming GPU** — blocked from every angle; keep CPU.

## Orin Nano 8 GB caveat

NVIDIA's cu132 aarch64 wheels ship **no sm_87 gencode** — anything riding
them (pocket-tts GPU, future torch-based engines) is **Thor-only** until a
JP 7.2.1 sm_87 wheel appears. The report's rule 1 (`cudaErrorNoKernelImageForDevice`)
applies verbatim. On Orin, the CPU roster is not a fallback — it is the
deployment thesis (Option A), and `phonon31m-tts` + `parakeet-*-packed`
were chosen for exactly that.
