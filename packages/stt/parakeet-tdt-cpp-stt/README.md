# parakeet-tdt-cpp-stt (Jetson sm_87 + sm_110)

**NVIDIA parakeet-tdt-0.6b-v3** (CC-BY-4.0, 25 languages) — the same 0.6B
TDT as our sherpa int8 row (`parakeet-tdt-stt`), here on
[mudler/parakeet.cpp](https://github.com/mudler/parakeet.cpp) with native
**sm_87 + sm_110a cubins + compute_110 PTX**.

| GGUF | Size | Target |
|---|---|---|
| `tdt-0.6b-v3-q4_k.gguf` | ~675 MB | **Orin Nano 8 GB budget** (~0.8 GB voice budget incl. VAD) + Thor |

mudler's q4_k is the smallest published GGUF for this model in the
parakeet.cpp schema (full ladder: f16 1.44 / q8_0 941 / q6_k 813 / q5_k 742 /
**q4_k 675 MB**); his quants keep convs/norms at higher precision and the
NVIDIA models are validated at WER 0 against NeMo. GGUF quants dequantize at
load/compute — the file runs identically on GPU (CUDA0) and CPU.

Even smaller artifacts exist for the *whisper.cpp* runtime only
(ggml-org q4_0 `.bin`, 356 MB) — different GGUF dialect, not loadable here.

## Usage

```sh
docker run --rm --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 \
  -v /audio:/audio:ro --entrypoint stt-transcribe \
  narandill/parakeet-tdt-cpp-stt:0.1.0 /audio/x.wav

# OpenAI-compatible server (POST /v1/audio/transcriptions)
docker run -d --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 \
  -p 8080:8080 narandill/parakeet-tdt-cpp-stt:0.1.0

# CPU-only (Orin Nano Option A): -e PARAKEET_DEVICE=cpu
```

Chunked pseudo-streaming: `-e PARAKEET_VAD_MODEL=/models/silero-vad-f16.gguf`
(silero cuts at ≥0.2 s pauses). True streaming lives in the
`nemotron-unified-asr` image (same binary, buffered streaming presets).

## License

Weights CC-BY-4.0 (nvidia/parakeet-tdt-0.6b-v3); parakeet.cpp MIT;
silero-vad MIT. GGUF conversion by mudler/parakeet-cpp-gguf.

## Server (voice-api v1)

```sh
docker run -d --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 -p 8080:8080 <image>
curl -s localhost:8080/v1/voices | jq   # the baked model (parakeet-server route)
curl -s localhost:8080/v1/audio/transcriptions -F file=@/audio/x.wav -F model=<id>
```
`GET /v1/models` is an alias. `--concurrency K` / `--threads N` via
`PARAKEET_CONCURRENCY` / `PARAKEET_THREADS`. Batch-only: streaming partials
are a CLI feature (`stt-transcribe --stream-latency`), not served yet.
