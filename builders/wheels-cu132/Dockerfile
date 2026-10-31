# syntax=docker/dockerfile:1
# =============================================================================
# Shared wheels builder (cu132): onnxruntime-gpu + ctranslate2 + piper-tts +
# wyoming-faster-whisper + wyoming-piper + pysilero-vad for sm_87 + sm_110.
# Instruction stream is byte-identical to the original build -> BuildKit
# replays it from cache in seconds (no 75-minute recompile). Build with:
#   docker build --target builder -t futureproofhomes/edge-wheels:cu132 .
# =============================================================================
# =============================================================================
# nexus-wyoming: Home-Assistant voice stack (STT + TTS) in ONE container for
# Jetson AGX Thor (sm_110) + AGX Orin (sm_87) on JetPack 7.2.1 (CUDA 13.2).
#
# ONE image serves both boards (SASS sm_87 + sm_110, PTX compute_110). CUDA
# 13.2 userspace + cuDNN 9 are baked in; the GPU driver (libcuda.so.1) is
# injected from the host by nvidia-container-toolkit at run time.
#
# Runs two wyoming services under supervisord:
#   - wyoming-faster-whisper (STT)  port 10300
#   - wyoming-piper (TTS)           port 10200
#
# Targets:
#   narandill/nexus-wyoming-thor:<ver>  and  narandill/nexus-wyoming-orin:<ver>
#   (identical image, two tags) + narandill/nexus-wyoming:<ver>
# =============================================================================

ARG BUILDER_BASE=nvidia/cuda:13.2.1-devel-ubuntu24.04
ARG RUNTIME_BASE=nvidia/cuda:13.2.1-runtime-ubuntu24.04

ARG CTRANSLATE2_REF=v4.8.4
ARG ONNXRUNTIME_REF=v1.31.1
ARG PIPER_REF=v1.8.0
ARG WYOMING_PIPER_REF=v2.5.2
ARG WYOMING_WHISPER_REF=v3.8.1
ARG PYTHON_VER=3.12
# sm_87 (AGX Orin) + sm_110a (AGX Thor). ORT's FP4/MoE kernels emit
# family-conditional instructions (cvt.e2m1x2.f32) that ptxas rejects for
# plain compute_110 PTX, so the forward-compat PTX is compute_110a (JITs on
# any 110a-family part; upstream's own aarch64 wheels ship no PTX at all).
ARG CUDA_ARCHS_ORT=87-real;110a-real;110a-virtual

# -----------------------------------------------------------------------------
# Builder: full CUDA toolchain. Produces wheels + shared libs only.

FROM ${BUILDER_BASE} AS builder

ARG CTRANSLATE2_REF
ARG ONNXRUNTIME_REF
ARG PIPER_REF
ARG WYOMING_PIPER_REF
ARG WYOMING_WHISPER_REF
ARG CUDA_ARCHS_ORT

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates cmake ninja-build build-essential gcc-11 g++-11 \
        libssl-dev pkg-config python3 python3-dev python3-pip \
        libcudnn9-dev-cuda-13 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --break-system-packages uv numpy packaging wheel

COPY patches/ /tmp/patches/

# --- libcuda stub: SONAME-correct symlink so executables link (builder-only) -
RUN if [ -f /usr/local/cuda/lib64/stubs/libcuda.so ]; then \
        ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1; \
    fi

# --- CTranslate2 (GPU, from source: upstream publishes no arm64 CUDA wheel) --
# Frozen FindCUDA cannot parse two-digit majors ("11.0" -> Unknown CUDA
# Architecture Name, upstream issue #2082): keep the dotted list single-digit
# and inject Thor's sm_110 via explicit -gencode flags.
RUN git clone --depth 1 --branch "${CTRANSLATE2_REF}" --recursive \
        https://github.com/OpenNMT/CTranslate2.git /src/CTranslate2

RUN cmake -B /src/CTranslate2/build -S /src/CTranslate2 \
        -DCMAKE_BUILD_TYPE=Release \
        -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_MKL=OFF -DOPENMP_RUNTIME=COMP \
        -DCUDA_ARCH_LIST="8.7+PTX" \
        -DCUDA_NVCC_FLAGS="-gencode=arch=compute_110,code=sm_110;-gencode=arch=compute_110,code=compute_110" \
        -DCMAKE_INSTALL_PREFIX=/opt/ct2-dist \
    && cmake --build /src/CTranslate2/build --parallel "$(nproc)" \
    && cmake --install /src/CTranslate2/build \
    && cp -r /opt/ct2-dist/* /usr/local/ && ldconfig

RUN cd /src/CTranslate2/python \
    && python3 -m pip install --break-system-packages -r install_requirements.txt \
    && python3 setup.py bdist_wheel -d /wheels

# --- onnxruntime-gpu (from source: official aarch64 cu13 wheels contain only
# 89;90a;120a;121a SASS with no PTX - no sm_87, no sm_110, nothing to JIT) ----
RUN git clone --depth 1 --branch "${ONNXRUNTIME_REF}" \
        https://github.com/microsoft/onnxruntime /src/onnxruntime \
    && cd /src/onnxruntime && git submodule update --init --recursive

# CCCL device_transform.cuh breaks under GCC with CUDA 13.2 (qualified-name
# specialization at global scope); patch it in the toolkit headers.
COPY files/patch_cccl.py /tmp/patch_cccl.py
RUN python3 /tmp/patch_cccl.py

# No TensorRT: nothing in this stack uses the TRT EP (saves ~1.5-2 GB).
# Memory-adaptive parallelism: nvcc/cicc uses several GB per job.
RUN cd /src/onnxruntime \
    && MAX_JOBS=$(( ($(free -g | awk '/^Mem:/{print $2}') - 8) / 6 )) \
    && if [ "$MAX_JOBS" -lt 2 ]; then MAX_JOBS=2; fi \
    && if [ "$MAX_JOBS" -gt 12 ]; then MAX_JOBS=12; fi \
    && echo "ORT build MAX_JOBS=$MAX_JOBS" \
    && ./build.sh --config Release --update --parallel "$MAX_JOBS" --build \
        --build_wheel --build_shared_lib --skip_tests --skip_submodule_sync \
        --allow_running_as_root \
        --cmake_extra_defines CMAKE_CXX_FLAGS="-Wno-unused-variable" \
        --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS_ORT}" \
        --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
        --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu \
        --use_cuda \
    && cp build/Linux/Release/dist/onnxruntime*.whl /wheels/

# --- piper-tts (piper1-gpl; strips the onnxruntime dep, GPU ort wins later) --
RUN git clone --depth 1 --branch "${PIPER_REF}" \
        https://github.com/OHF-voice/piper1-gpl /src/piper1-gpl \
    && sed -i 's/"onnxruntime>=1,<2",//' /src/piper1-gpl/setup.py \
    && uv build --wheel --out-dir /wheels /src/piper1-gpl

# --- wyoming-faster-whisper + wyoming-piper (fork's LAN-IP / name patches) ---
RUN git clone --depth 1 --branch "${WYOMING_WHISPER_REF}" \
        https://github.com/OHF-Voice/wyoming-faster-whisper /src/wfw \
    && cd /src/wfw && git apply --verbose /tmp/patches/0001-*.patch \
    && uv build --wheel --out-dir /wheels .

RUN git clone --depth 1 --branch "${WYOMING_PIPER_REF}" \
        https://github.com/OHF-Voice/wyoming-piper /src/wyp \
    && cd /src/wyp && git apply --verbose /tmp/patches/0002-*.patch \
    && uv build --wheel --out-dir /wheels .

# --- pre-build pysilero-vad wheel (compiled ggml ext; avoid compiling in the
# compiler-free runtime stage in case upstream ships no aarch64 wheel) --------
RUN python3 -m pip wheel --no-cache-dir --no-deps --wheel-dir /wheels "pysilero-vad==3.4.0"

# -----------------------------------------------------------------------------
# Runtime: wheels + cuDNN + supervisor. No compilers, headers, sources, caches.
# -----------------------------------------------------------------------------
