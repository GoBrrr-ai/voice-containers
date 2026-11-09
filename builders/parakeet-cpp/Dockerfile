# syntax=docker/dockerfile:1
# =============================================================================
# Shared builder: mudler/parakeet.cpp (GGML family) with CUDA for sm_87 + sm_110.
# Produces /opt/parakeet-dist (parakeet-cli, parakeet-server, libs) that the
# STT engine images COPY in. GGML toolchain identical to our llama.cpp build
# (gcc-11 CUDA host compiler, semver-less arch list, GGML_NATIVE=OFF for
# portability across A78AE (Orin) and Thor CPUs).
#
# This image is an intermediate artifact — do not ship.
# =============================================================================

ARG BUILDER_BASE=nvidia/cuda:13.2.1-devel-ubuntu24.04
ARG PARAKEET_CPP_REF=master
# Cap compile parallelism on shared hosts (RAM guard; 0 = nproc). A -j14 CUDA
# build peaks ~14-28 GB — the 2026-11-03 OOM rule applies while vllm serves.
ARG PARALLEL_JOBS=0

FROM ${BUILDER_BASE} AS builder

ARG PARAKEET_CPP_REF

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates cmake ninja-build build-essential \
        gcc-11 g++-11 pkg-config curl patch \
    && rm -rf /var/lib/apt/lists/*

# libcuda stub: SONAME-correct symlink (builder-only; same trick as llama.cpp)
RUN if [ -f /usr/local/cuda/lib64/stubs/libcuda.so ]; then \
        ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1; \
    fi

RUN git clone --recursive https://github.com/mudler/parakeet.cpp /src/parakeet.cpp \
    && git -C /src/parakeet.cpp checkout "${PARAKEET_CPP_REF}"

# Fix: MelKernel ctor derefs the `w` tensor AFTER weight_to_host_f32() has (on
# first use) realized the weights — which mutates tensor buffers and can leave
# the captured pointer dangling -> SIGSEGV in the GPU streaming path
# (MelKernel::MelKernel+388, deterministic; CPU streaming / GPU offline fine).
# Upstream-unreported as of 2026-10-31 — file with a link to this patch.
COPY files/mel-kernel-dangling-tensor.patch /tmp/patches/
RUN patch -p1 -d /src/parakeet.cpp < /tmp/patches/mel-kernel-dangling-tensor.patch
# voice-api v1 discovery: GET /v1/voices + /v1/models report the ONE model the
# server was started with (basename of the resolved model file) — each GGUF
# image then answers with exactly its baked model.
COPY files/server-model-routes.patch /tmp/patches/
RUN patch -p1 -d /src/parakeet.cpp < /tmp/patches/server-model-routes.patch

RUN cmake -B /src/parakeet.cpp/build -S /src/parakeet.cpp \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="87-real;110a-real;110-virtual" \
        -DCMAKE_CUDA_HOST_COMPILER="$(command -v gcc-11 || command -v gcc)" \
        -DPARAKEET_GGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
    && cmake --build /src/parakeet.cpp/build --parallel "$( [ "${PARALLEL_JOBS}" -gt 0 ] && echo "${PARALLEL_JOBS}" || nproc )" \
# Upstream ships no cmake install rules and no global RUNTIME_OUTPUT_DIRECTORY:
# executables sit next to their CMakeLists (build/examples/cli/...). Find them.
    && mkdir -p /opt/parakeet-dist/bin /opt/parakeet-dist/lib \
    && find /src/parakeet.cpp/build -type f -executable \
        \( -name 'parakeet-*' -o -name 'diarize' -o -name 'voice-detect' \
           -o -name 'sound-window-eval' -o -name 'ced*' \) \
        -not -path '*/CMakeFiles/*' -exec cp -a {} /opt/parakeet-dist/bin/ \; \
    && find /src/parakeet.cpp/build -name '*.so*' -not -path '*/CMakeFiles/*' \
        -exec cp -a {} /opt/parakeet-dist/lib/ \; \
    && ls -la /opt/parakeet-dist/bin/ /opt/parakeet-dist/lib/

# Strip the CUDA forward-compat pair the cu13.2 base ships at
# /usr/local/cuda/compat{,_orin}: nvidia-container-toolkit 1.19.1 injects a
# createContainer hook (`enable-cuda-compat`) into every container create and
# that hook panics parsing these libs' ELF notes (slice [:73] vs cap 71) —
# docker run dies with exit 125 before the entrypoint, no logs. The 595.58.03
# libs are never newer than the fleet's 595.78 driver, so the hook could only
# no-op; libcuda.so.1 is host-injected at run time regardless. The dist
# subtree consumers COPY --from is /opt/parakeet-dist — untouched by this.
# Post-mortem + repro: docs/postmortem-cuda-compat-hook.md
RUN rm -rf /usr/local/cuda/compat /usr/local/cuda/compat_orin
