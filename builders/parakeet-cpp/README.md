# parakeet-cpp-builder (shared artifact builder — NOT an engine image)

Builds [mudler/parakeet.cpp](https://github.com/mudler/parakeet.cpp) (GGML
family) with CUDA for **sm_87 + sm_110a native cubins + compute_110 PTX** and
exports `/opt/parakeet-dist` (`parakeet-cli`, `parakeet-server`, ggml libs).
This image is an intermediate artifact — **do not ship or run it directly**.

One binary serves three engine images (only the baked GGUF differs):
`nemotron-unified-asr`, `parakeet-redux-stt`, `phonon2-stt`.

```sh
docker build -t futureproofhomes/edge-builder:parakeet-cpp .
```

Toolchain mirrors the proven llama.cpp build: gcc-11 CUDA host compiler,
`GGML_NATIVE=OFF`, libcuda stub symlink (builder-only). Pinned ref:
`PARAKEET_CPP_REF=2eac2e77d9c7ef8899f7a86a46e693ac5ac1a5cd` (2026-10-30 master,
required for Phonon-2.5 + Redux GGUFs; v0.5.0 tag is too old). Carries our
upstream-bound fix for the GPU `--stream` SIGSEGV (MelKernel dangling tensor
after weight realization — see files/mel-kernel-dangling-tensor.patch).

**Decoupling:** engine images reference this image only via
`ARG PARAKEET_DIST_IMAGE` — at standalone-repo migration time, either retag /
rebuild this builder there and override the ARG, or vendor `/opt/parakeet-dist`
into each engine. Nothing else in the engine Dockerfiles changes.
