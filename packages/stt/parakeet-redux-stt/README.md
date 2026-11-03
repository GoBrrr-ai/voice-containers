# parakeet-redux-stt (Jetson sm_87 + sm_110)

**moondream/parakeet-redux** (CC-BY-4.0) — moondream's 178 M-param ternary
derivative of parakeet-tdt-0.6b-v3 (25 EU languages) — as two baked GGUFs on
[mudler/parakeet.cpp](https://github.com/mudler/parakeet.cpp) with native
**sm_87 + sm_110a cubins + compute_110 PTX**:

| GGUF | Size | Target |
|---|---|---|
| `redux-q8_0.gguf` | ~898 MB | **GPU (default)** — dequantized, runs on CUDA0 |
| `redux-packed.gguf` | ~203 MB | **CPU** — packed-ternary kernels incl. aarch64 dotprod (A78AE) |

FLEURS-25 WER 10.56 — *beats* the full parakeet-v3 it was distilled from.

## Selecting a variant

```sh
# GPU q8 (default)
docker run ... narandill/parakeet-redux-stt:0.1.0

# Ultra-light CPU packed (~0.35 GB RSS, frees the GPU for the LLM)
docker run ... -e PARAKEET_DEVICE=cpu -e PARAKEET_MODEL=/models/redux-packed.gguf \
  narandill/parakeet-redux-stt:0.1.0
```

## Streaming

The model is an **offline TDT** decoder — streaming is chunked pseudo-streaming:
`stt-transcribe` injects `--vad --vad-model silero-vad-f16.gguf` which cuts
long audio at ≥0.2 s pauses into ≤30 s segments. For true token-level
streaming use the sibling `nemotron-unified-asr` image (same binary).

## Run

```sh
# OpenAI-compatible server
docker run --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 --rm -p 8080:8080 \
  narandill/parakeet-redux-stt:0.1.0

# CLI
docker run ... --entrypoint stt-transcribe narandill/parakeet-redux-stt:0.1.0 /audio/x.wav --json
```

## Env

Same contract as `nemotron-unified-asr` (PARAKEET_MODEL / PARAKEET_DEVICE /
PARAKEET_VAD_MODEL / PARAKEET_PORT …); `PARAKEET_LANG` is unused (no language
prompt in TDT models — language is implicit).

RAM: q8 on GPU ≈ 1.0 GB; packed on CPU ≈ 0.35 GB RSS (the RAM champion).

## Server (voice-api v1)

```sh
docker run -d --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 -p 8080:8080 <image>
curl -s localhost:8080/v1/voices | jq   # the baked model (parakeet-server route)
curl -s localhost:8080/v1/audio/transcriptions -F file=@/audio/x.wav -F model=<id>
```
`GET /v1/models` is an alias. `--concurrency K` / `--threads N` via
`PARAKEET_CONCURRENCY` / `PARAKEET_THREADS`. Batch-only: streaming partials
are a CLI feature (`stt-transcribe --stream-latency`), not served yet.
