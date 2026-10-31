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

RUN cmake -B /src/parakeet.cpp/build -S /src/parakeet.cpp \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="87-real;110a-real;110-virtual" \
        -DCMAKE_CUDA_HOST_COMPILER="$(command -v gcc-11 || command -v gcc)" \
        -DPARAKEET_GGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
    && cmake --build /src/parakeet.cpp/build --parallel "$(nproc)" \
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
