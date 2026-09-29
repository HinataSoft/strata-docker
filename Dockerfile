# syntax=docker/dockerfile:1.7
#
# Strata in Docker (NVIDIA, Linux).
#
# The Strata sources are NOT part of this repository.  They arrive as a named build context called "strata",
# which is either a local checkout or a git URL - see docker-compose.yml.  Nothing here is copied into the
# Strata tree and nothing there knows about this one.
#
#   docker compose build
#   docker compose --profile init run --rm strata-init     once: downloads the model, builds the pack and MTP
#   docker compose up -d strata                            every time after that: serves on :8080
#
# Two stages:
#   builder  compiles the engine and the vision encoder against the CUDA toolkit.  No GPU is needed - the
#            target architecture is a build arg - so this runs anywhere, with or without the NVIDIA runtime.
#   runtime  carries only the two binaries, the Python side and NVIDIA's CUDA libraries from pip.
#
# Why it compiles at all: as of v0.1.21 the upstream release has no Linux engine (it publishes
# strata-windows-x64.zip only), so setup.py would try to apt-get the CUDA toolkit at first run.  The builder
# stage does that work and writes engine/BUILD.json with source="local", which makes setup.py adopt these
# binaries and skip the download, the compile and the build-tool install entirely - whether or not a Linux
# asset ever appears upstream.

ARG CUDA_VERSION=13.0.1
ARG UBUNTU_VERSION=24.04

# ---------------------------------------------------------------------------------------------- builder
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

# 120 = RTX 50 (the 5060 Ti).  "86;120" also covers the RTX 3060; each extra arch costs build time and size.
ARG CUDA_ARCHITECTURES=120
# ON  = ggml gets an AVX2 baseline, so the image runs on any AVX2 CPU and can be built anywhere.
# OFF = ggml is built with -march=native.  Better tuning for one exact CPU, but the image then runs ONLY on
#       it - anywhere else the engine dies with SIGILL.  Safe when you build on the target machine.
#       Strata's own expert kernels are per-file with a run-time check either way, so ON costs little.
ARG STRATA_PORTABLE=ON
# Whether the vision encoder is built with CUDA.  ON links it against ggml-cuda, so it needs the CUDA
# runtime libraries even when it is told to run on the CPU (STRATA_VISION=cpu).  Set OFF for a CPU-only
# encoder with no CUDA dependency at all.
ARG STRATA_VISION_CUDA=ON
# the commit setup.py pins (its LLAMA_CPP_COMMIT).  Verified against the Strata tree by the smoke test.
ARG LLAMA_CPP_COMMIT=3cf03257f219afbe7334045ff7c6a06ac68c627d

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build curl unzip ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/strata

# ggml for the build, gguf-py for the tools, mtmd for the vision encoder
RUN curl -fsSL -o /tmp/llama.zip "https://github.com/ggml-org/llama.cpp/archive/${LLAMA_CPP_COMMIT}.zip" \
    && unzip -q /tmp/llama.zip -d /tmp/llama \
    && mkdir -p third_party \
    && mv "/tmp/llama/llama.cpp-${LLAMA_CPP_COMMIT}" third_party/llama.cpp \
    && rm -rf /tmp/llama.zip /tmp/llama

COPY --from=strata CMakeLists.txt setup.py ./
COPY --from=strata src/ src/
COPY --from=strata include/ include/
COPY --from=strata third_party/ggml/ third_party/ggml/
COPY --from=strata tools/ tools/

RUN cmake -G Ninja -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DSTRATA_ENABLE_CUDA=ON \
        -DSTRATA_BUILD_TESTS=OFF \
        -DSTRATA_PORTABLE=${STRATA_PORTABLE} \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DSTRATA_GGML_DIR=/opt/strata/third_party/llama.cpp \
    && cmake --build build --target strata -j"$(nproc)"

RUN cmake -G Ninja -S tools/vision -B build-vision \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_DIR=/opt/strata/third_party/llama.cpp \
        -DSTRATA_VISION_CUDA=${STRATA_VISION_CUDA} \
        -DSTRATA_PORTABLE=${STRATA_PORTABLE} \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    && cmake --build build-vision --target strata-vision -j"$(nproc)"

# engine/BUILD.json exactly as build_engine() writes it, so setup.py treats these binaries as its own local
# build.  src / vision_src are setup.py's own source fingerprints: with them matching, a later setup run will
# not try to recompile (which would fail here - the runtime stage has no compiler).  cuda_dirs is left empty
# on purpose, so the generated config falls back to cuda_lib_dirs() and finds NVIDIA's pip wheels.
COPY docker/write_build_json.py docker/
RUN mkdir -p engine \
    && cp build/strata engine/strata \
    && cp build-vision/bin/strata-vision engine/strata-vision \
    && chmod 0755 engine/strata engine/strata-vision \
    && python3 docker/write_build_json.py "${CUDA_ARCHITECTURES}" \
    && cat engine/BUILD.json

# The CUDA libraries the binaries actually link against, collected from the toolkit by asking ldd rather than
# by guessing pip package names.  NVIDIA's pip wheels cover cuBLAS and the CUDA runtime, but ggml-cuda also
# pulls in libnccl, which has no wheel in that set - without it the binary dies at exec with
# "error while loading shared libraries: libnccl.so.2", before it can print anything.
# libcuda.so.1 is deliberately excluded: that one comes from the host driver via nvidia-container-toolkit.
RUN mkdir -p /opt/cudalibs \
    && { ldd engine/strata; ldd engine/strata-vision; } \
       | awk '/=> \//{print $3}' | sort -u \
       | grep -E '/lib(nccl|cublas|cublasLt|cudart|cufft|curand|cusparse|cusolver|nvrtc|nvJitLink|nvToolsExt)\.' \
       | xargs -r -I{} cp -Lv {} /opt/cudalibs/ \
    && ls -1 /opt/cudalibs

# ---------------------------------------------------------------------------------------------- runtime
# Ubuntu 24.04, matching the builder: a binary linked against glibc 2.39 will not start on Debian bookworm.
FROM ubuntu:${UBUNTU_VERSION} AS runtime

ARG LLAMA_CPP_COMMIT
ARG STRATA_REF=unknown
LABEL org.opencontainers.image.title="strata" \
      org.opencontainers.image.description="Qwen3.8-Flash-Next on one NVIDIA GPU plus system RAM" \
      org.strata.source-ref="${STRATA_REF}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv libgomp1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# nvidia-container-toolkit reads these; "utility" is what puts nvidia-smi in the container, and setup.py's
# first step refuses to run without it.  Override NVIDIA_VISIBLE_DEVICES to pin one card (a GPU-<uuid>).
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# setup.py uses sys.executable, so anything it installs lands in this venv.
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir \
        numpy jinja2 regex pyyaml tqdm requests pillow psutil \
        nvidia-cublas==13.0.2.14 nvidia-cuda-runtime==13.0.96

WORKDIR /opt/strata

COPY --from=builder /opt/strata/engine/ engine/

# The toolkit libraries the binaries link against but the pip wheels above do not ship (libnccl in
# particular).  These go through ldconfig rather than LD_LIBRARY_PATH on purpose: the config's lib_dirs
# (the pip wheels) are put on LD_LIBRARY_PATH by server.py's child_env() and therefore still win, so this
# is a fallback for what the wheels do not cover, not a second copy that shadows them.
COPY --from=builder /opt/cudalibs/ /opt/cudalibs/
RUN echo /opt/cudalibs > /etc/ld.so.conf.d/strata-cuda.conf && ldconfig
# gguf-py is what the pack and MTP tools import (STRATA_GGUF_PY); ggml/CMakeLists.txt is the marker that stops
# get_llama_cpp() from downloading the llama.cpp source again at init time.
COPY --from=builder /opt/strata/third_party/llama.cpp/gguf-py/ third_party/llama.cpp/gguf-py/
COPY --from=builder /opt/strata/third_party/llama.cpp/ggml/CMakeLists.txt third_party/llama.cpp/ggml/CMakeLists.txt

# src/ include/ third_party/ggml/ are here for source_hash(), not to be compiled: with the fingerprints in
# BUILD.json matching the tree, setup.py sees the engine as current and never tries to rebuild it.
COPY --from=strata CMakeLists.txt setup.py chat.py ./
COPY --from=strata src/ src/
COPY --from=strata include/ include/
COPY --from=strata third_party/ggml/ third_party/ggml/
COPY --from=strata tools/ tools/
COPY --from=strata serve/ serve/
COPY --from=strata data/ data/

COPY docker/ docker/

# The stamp pip_install() checks, then the smoke test: this image drives setup.py through names that are not
# a public interface, so every assumption is asserted at BUILD time rather than discovered at run time.
RUN /opt/venv/bin/python docker/write_pip_stamp.py \
    && /opt/venv/bin/python docker/verify_assumptions.py --llama-commit "${LLAMA_CPP_COMMIT}" \
    && chmod 0755 docker/entrypoint.sh \
    && mkdir -p /data /config

ENV STRATA_DATA=/data \
    STRATA_CONFIG_DIR=/config \
    STRATA_FAMILY=qwen \
    STRATA_MODEL=IQ3_S \
    STRATA_CONTEXT=262144 \
    STRATA_KV=int8 \
    STRATA_VISION=gpu \
    STRATA_HOST=0.0.0.0 \
    STRATA_PORT=8080 \
    STRATA_PROMPT_CACHE=24 \
    STRATA_PROMPT_CACHE_EVERY=8192

EXPOSE 8080
VOLUME ["/data", "/config"]

# start-period covers loading 50 GB of experts into RAM on a cold page cache
HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${STRATA_PORT}/health" || exit 1

ENTRYPOINT ["/opt/strata/docker/entrypoint.sh"]
CMD ["serve"]
