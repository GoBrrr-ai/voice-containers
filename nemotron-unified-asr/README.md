# nemotron-unified-asr (Jetson sm_87 + sm_110)

**nvidia/nemotron-labs-unified-asr-0.6b** (OpenMDW-1.1, 25 EU languages) as an
official **q8_0 GGUF** (895 MB) running on [mudler/parakeet.cpp](https://github.com/mudler/parakeet.cpp)
(GGML) with **native sm_87 + sm_110a cubins and compute_110 PTX** baked by our
shared builder (`futureproofhomes/edge-builder:parakeet-cpp`).

Why GGUF and not ONNX: the RNNT/TDT architecture has **no ONNX export path**
(optimum can't export it; vLLM/SGLang/TensorRT-LLM don't do RNNT), but the
community GGUF quant runs token-identical to NeMo — parakeet.cpp's
[parity tests](https://github.com/mudler/parakeet.cpp/blob/master/docs/parity.md)
show "every step of every preset matches NeMo's streaming loop… final tokens
and text are identical" (q8_0 offline WER 0 vs stored NeMo references).

## Streaming

Buffered "unified" streaming — token-level, with latency presets mirroring
NVIDIA's (L,C,R) table:

```sh
stt-transcribe /audio/x.wav                     # streams at the default 560 ms preset
PARAKEET_STREAM_LATENCY=0.16 stt-transcribe ...  # 0.16 s min algorithmic latency
PARAKEET_STREAM_LATENCY=2.08 stt-transcribe ... # max-accuracy preset
```

Presets: 0.16/0.24/0.32/0.56/1.12/2.08 s (`--stream-latency` takes seconds). Offline mode: `-e PARAKEET_STREAM=`.

## Run

```sh
# OpenAI-compatible server (POST /v1/audio/transcriptions, WAV only)
docker run --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 --rm -p 8080:8080 \
  narandill/nemotron-unified-asr-thor:0.1.0

# CLI batch/streaming
docker run --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 --rm \
  -v /audio:/audio:ro --entrypoint stt-transcribe \
  narandill/nemotron-unified-asr-thor:0.1.0 /audio/x.wav --json
```

After the JP 7.2.1 upgrade drop `-e NVIDIA_DISABLE_REQUIRE=1`.

## Env

| Var | Default | Meaning |
|---|---|---|
| `PARAKEET_MODEL` | `/models/nemotron-labs-unified-asr-0.6b-q8_0.gguf` | baked |
| `PARAKEET_VAD_MODEL` | `/models/silero-vad-f16.gguf` | VAD for long-audio cuts |
| `PARAKEET_STREAM` | `1` | streaming on (0 = offline) |
| `PARAKEET_STREAM_LATENCY` | `0.56` | preset in SECONDS (0.16/0.24/0.32/0.56/1.12/2.08) |
| `PARAKEET_LANG` | `en` | language-ID prompt (`auto` to detect) |
| `PARAKEET_DEVICE` | auto (first GPU) | `cpu` / `CUDA0` to force |
| `PARAKEET_PORT` / `PARAKEET_HOST` | `8080` / `0.0.0.0` | server bind |
| `PARAKEET_THREADS` / `PARAKEET_CONCURRENCY` | — | server CPU pool |

RAM: ~1.0 GB RSS (q8_0); WER (Rev-2 ranking): EN ~5.2 / best-in-class EU multi;
#1 STT for the Orin Nano Super 8 GB stack.
