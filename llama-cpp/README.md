# llama.cpp for Jetson Thor (sm_110) + Orin (sm_87) — JP 7.2.1

One image serves both boards. Built for JetPack 7.2.1 (L4T r39.2.1, driver R595)
with CUDA 13.2 userspace baked in; the driver (`libcuda.so.1`) is injected from
the host by nvidia-container-toolkit. Never bake or mount host CUDA.

## What's inside
- llama.cpp **v0.8.0** C++ binaries: `llama-server`, `llama-cli`, `llama-bench`, ...
- **llama-cpp-python 0.3.36** wheel, vendored llama.cpp forced to `v0.8.0`
- Base: `nvidia/cuda:13.2.1-devel-ubuntu24.04` (builder) → `13.2.1-runtime-ubuntu24.04` (final)
- Arch flags: `87-real;110a-real;110-virtual` (sm_87 SASS, sm_110a SASS, compute_110 PTX)
- Build-time patch: PR #31268 (Orin single-column MMVQ, see `patches/README.md`)
- No cuDNN/TensorRT needed. Host compiler gcc-11 (fork's known-good pin).

## Build
```bash
./build.sh            # tags futureproofhomes/llama-cpp:{ver,latest} + -thor/-orin aliases
./build.sh --push
```

## Verify (SASS/PTX presence)
```bash
./test/verify_arch.sh futureproofhomes/llama-cpp-thor:0.8.0
```

## Smoke test on hardware
```bash
./test/smoke_llama.sh futureproofhomes/llama-cpp-thor:0.8.0
# starts llama-server (full -ngl 999 offload, Qwen3-1.7B-Q4_K_M) and sends one completion
```

## Run
```bash
docker run --runtime nvidia --rm -p 8080:8080 -v /data/models:/data \
  futureproofhomes/llama-cpp-thor:0.8.0 \
  llama-server -m /data/model.gguf -ngl 999 --host 0.0.0.0
```
Notes for v0.8.0: `-ngl -1` means auto — pass an explicit `-ngl 999` on UMA
boards (Thor/Orin) so offload is deterministic. `-fa on` for FlashAttention
(default auto). Do NOT set `CUDA_SCALE_LAUNCH_QUEUES` on single-GPU Jetson
(Orin MoE deadlock history, #19219/#19227).

On a pre-7.2.1 host (driver < 595) add `-e NVIDIA_DISABLE_REQUIRE=1` until the
host is upgraded; CUDA 13.2 userspace runs on the 580-series driver via minor
version compatibility.
