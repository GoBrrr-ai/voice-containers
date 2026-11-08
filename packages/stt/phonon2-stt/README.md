# phonon2-stt (Jetson sm_87 + sm_110)

**Fermion Research Phonon line** via parakeet.cpp (GGML, native **sm_87 +
sm_110a cubins + compute_110 PTX**). Baked models (English, Phonon-2.5
generation — the successor of [FermionResearch/Phonon-2](https://huggingface.co/FermionResearch/Phonon-2)):

| GGUF | Size | Target |
|---|---|---|
| `phonon-2.5-english-q8_0.gguf` | ~897 MB | **GPU (default)** |
| `phonon-2.5-english-packed.gguf` | ~230 MB | **CPU** (packed kernels, aarch64 dotprod) |

Phonon-2.5-english: EN 7-set WER **4.82** — the best sub-1 GB EN WER in the
Rev-2 ranking. Phonon-2.5 is a retrained 25-language/EN pair of
parakeet-tdt-0.6b-v3, so it runs as ordinary GGUF on the same binary.

## Why not the exact Phonon-2?

The exact **Phonon-2** artifact (`phonon-2.bps.tar.zst`) is a custom
`fermion-five-value-parakeet-v1` packed-tensor container — **not ONNX, not
GGUF** — and its GPU engine is pure torch (BF16, ~1.3–1.5 GB, over the STT
budget). If you need the *exact* Phonon-2, the CPU path is documented by
upstream and runs on plain Ubuntu 24.04:

```sh
pip install fermion-research==0.2.15 transformers==5.17.0 soundfile  # + CPU torch
FERMION_DEVICE=cpu python -m fermion... # see https://github.com/fermionresearch/phonon
```

CPU RSS ≈ 0.25–0.35 GB. We ship the GGUF path (lighter, no torch).

## Streaming

Offline TDT — chunked pseudo-streaming via baked `--vad --vad-model
silero-vad-f16.gguf` (silero cuts at ≥0.2 s pauses). True token-level
streaming: use `nemotron-unified-asr` (same binary).

## Run

```sh
# GPU q8 (default)
docker run --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 --rm -p 8080:8080 \
  narandill/phonon2-stt:0.2.0

# Ultra-light CPU packed
docker run ... -e PARAKEET_DEVICE=cpu -e PARAKEET_MODEL=/models/phonon-2.5-english-packed.gguf ...
```

Same env contract as the other parakeet.cpp images. RAM: q8 GPU ≈ 1.0 GB,
packed CPU ≈ 0.35 GB.

## Server (voice-api v1)

```sh
docker run -d --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 -p 8080:8080 <image>
curl -s localhost:8080/v1/voices | jq   # the baked model (parakeet-server route)
curl -s localhost:8080/v1/audio/transcriptions -F file=@/audio/x.wav -F model=<id>
```
`GET /v1/models` is an alias. `--concurrency K` / `--threads N` via
`PARAKEET_CONCURRENCY` / `PARAKEET_THREADS`. Batch-only: streaming partials
are a CLI feature (`stt-transcribe --stream-latency`), not served yet.
