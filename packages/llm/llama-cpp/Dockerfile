# syntax=docker/dockerfile:1
# =============================================================================
# llama.cpp v0.8.0 + llama-cpp-python 0.3.36 for Jetson AGX Thor (sm_110) and
# AGX Orin (sm_87) on JetPack 7.2.1 (L4T r39.2.1, driver R595, CUDA 13.2).
#
# ONE image serves both boards: SASS cubins for sm_87 + sm_110a, plus PTX
# compute_110 for forward compatibility. CUDA 13.2 userspace is baked in
# (cudart/cublas/cublasLt come from the runtime base); the GPU driver
# (libcuda.so.1) is injected from the host by nvidia-container-toolkit.
#
# Targets:
#   narandill/llama-cpp:<ver> + :latest (one image, both boards)
# =============================================================================

ARG BUILDER_BASE=nvidia/cuda:13.2.1-devel-ubuntu24.04
ARG RUNTIME_BASE=nvidia/cuda:13.2.1-runtime-ubuntu24.04

ARG LLAMA_CPP_REF=v0.8.0
ARG LLAMA_CPP_PYTHON_VERSION=0.3.36
# sm_87 (AGX Orin, no 'a' variant exists), sm_110a (AGX Thor, Blackwell family),
# compute_110 PTX so future 11.x parts can JIT.
ARG CUDA_ARCHS=87-real;110a-real;110-virtual

# -----------------------------------------------------------------------------
# Builder: full CUDA devel toolchain. Nothing from this stage reaches the final
# image except /opt/llama-dist and the python wheel.
# -----------------------------------------------------------------------------
FROM ${BUILDER_BASE} AS builder

ARG LLAMA_CPP_REF
ARG LLAMA_CPP_PYTHON_VERSION
ARG CUDA_ARCHS
ARG GGML_FLAGS="-DGGML_CUDA=ON -DGGML_CUDA_F16=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DLLAMA_OPENSSL=ON"
# Fork-verified pin: GCC 13 on Ubuntu 24.04 has an aarch64+CUDA codegen bug in
# fused kernels; GCC 11 is within the CUDA 13.2 supported-host-compiler range.
ARG CUDA_HOST_COMPILER=gcc-11

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates cmake ninja-build build-essential \
        gcc-11 g++-11 libssl-dev pkg-config \
        python3 python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --break-system-packages uv

COPY patches/ /tmp/patches/

# libggml-cuda links the CUDA driver API (VMM support). Build containers have no
# driver - only the stub, whose filename (libcuda.so) does not match its SONAME
# (libcuda.so.1) so executables fail to resolve it transitively. Place a
# SONAME-correct symlink to the stub on the linker path. Builder-only: the
# final image never sees it, the real libcuda.so.1 is host-injected at runtime.
RUN if [ -f /usr/local/cuda/lib64/stubs/libcuda.so ]; then \
        ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1; \
    fi

# --- C++ binaries (llama-server, llama-cli, llama-bench, ...) ----------------
RUN git clone --depth 1 --branch "${LLAMA_CPP_REF}" \
        https://github.com/ggml-org/llama.cpp /src/llama.cpp

# Profitable post-v0.8.0 patches (vendored, pinned by SHA; see patches/README)
RUN cd /src/llama.cpp && git apply --verbose /tmp/patches/*.patch

RUN cmake -B /src/llama.cpp/build -S /src/llama.cpp \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DCMAKE_CUDA_HOST_COMPILER="$(command -v ${CUDA_HOST_COMPILER} || command -v gcc)" \
        ${GGML_FLAGS} \
        -DGGML_NATIVE=OFF \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_CURL=OFF \
    && cmake --build /src/llama.cpp/build --config Release --parallel "$(nproc)" \
    && cmake --install /src/llama.cpp/build --prefix /opt/llama-dist

# --- llama-cpp-python wheel, vendored llama.cpp forced to v0.8.0 -------------
# The 0.3.36 python code is ABI-compatible with the v0.8.0 C API (259 identical
# functions; verified by API diff), so pointing its vendored submodule at the
# tag is safe. llama-cpp-python builds its own libllama inside the wheel.
RUN git clone --depth 1 --branch "v${LLAMA_CPP_PYTHON_VERSION}" \
        https://github.com/abetlen/llama-cpp-python.git /src/lcpp

RUN cd /src/lcpp \
    && git submodule update --init --recursive \
    && cd vendor/llama.cpp \
    && git fetch --depth 1 origin "refs/tags/${LLAMA_CPP_REF}:refs/tags/${LLAMA_CPP_REF}" \
    && git checkout --detach "${LLAMA_CPP_REF}" \
    && git apply --verbose /tmp/patches/*.patch

RUN FORCE_CMAKE=1 \
    CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS} \
        -DCMAKE_CUDA_HOST_COMPILER=$(command -v ${CUDA_HOST_COMPILER} || command -v gcc) \
        -DGGML_NATIVE=OFF ${GGML_FLAGS} -DLLAVA_BUILD=OFF -DLLAMA_CURL=OFF" \
    uv build --wheel --out-dir /wheels /src/lcpp

# -----------------------------------------------------------------------------
# Runtime: only binaries, shared libs and the python wheel. No compilers, no
# headers, no source trees, no caches.
# -----------------------------------------------------------------------------
FROM ${RUNTIME_BASE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv libgomp1 libssl3t64 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/llama-dist/bin/ /usr/local/bin/
COPY --from=builder /opt/llama-dist/lib/  /usr/local/lib/
COPY --from=builder /wheels/              /tmp/wheels/

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir /tmp/wheels/llama_cpp_python-*.whl \
    && rm -rf /tmp/wheels

ENV PATH="/opt/venv/bin:${PATH}"

RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/llama-cpp.conf \
    && echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda-lib64.conf \
    && ldconfig

# Sanity (driver-free, build containers have no GPU): no missing shared deps
# except libcuda.so.1, which nvidia-container-toolkit injects from the host at
# run time. GPU-touching checks live in test/smoke_llama.sh instead.
RUN missing="$(ldd /usr/local/bin/llama-server /usr/local/bin/llama-cli \
                   /usr/local/lib/libggml-cuda.so 2>/dev/null \
                   | grep 'not found' | grep -v 'libcuda.so.1' || true)"; \
    if [ -n "$missing" ]; then echo "$missing"; exit 1; fi; \
    python3 -c "import ctypes" && echo "runtime deps OK"

EXPOSE 8080
WORKDIR /data
