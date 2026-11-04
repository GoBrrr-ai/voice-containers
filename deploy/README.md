# edge-voice — deployment

The voice stack as a `docker compose` deployment: the **voice-api pair**
(HTTP, legacy-style) by default, an optional **llama.cpp** LLM, and an optional
**HA-native wyoming pair** — the same images benchmarked in the README
(serve mode), now wired for a board.

## Board recipes

### Orin Nano 8 GB — Option A (CPU pair, recommended)

The defaults are the measured Option A co-location
([orin-nano-8gb-plan.md §3](../task/orin-nano-8gb-plan.md)): parakeet-packed
CPU STT + phonon-31m CPU TTS = **~1.23 GB pool, 0 GPU-side**, leaving
~3.5 GB for a 4B q4_K_M LLM at 8k ctx.

```sh
cp .env.example .env            # set VOICE_API_TOKEN if the box is on a LAN
docker compose --profile llm up -d
docker compose ps               # all services -> healthy (baked healthchecks)
curl -s localhost:8080/health   # {"status":"ok",...,"device":"cpu"}
```

**Thor bench box (GPU pair)** — override in `.env`:

```sh
STT_IMAGE=narandill/parakeet-tdt-cpp-stt:0.1.0
# drop the STT_DEVICE=cpu line -> the GGUF server defaults to CUDA0
```

kokoro GPU (24 kHz, pcm streaming) as TTS:
`TTS_IMAGE=narandill/kokoro-tts:0.1.0` + `TTS_DEVICE=cuda`.

### Streaming STT (SSE partials, profile `stream`)

The `stt-stream` service adds the nemotron-streaming voice-server — SSE
partials on `POST /v1/audio/transcriptions?stream=true` (`partial` events
at the 160 ms slice cadence, then one `final`), ~394 MiB VRAM on cuda
(smallest GPU footprint in the roster):

```sh
docker compose --profile stream --profile llm up -d
# STT_STREAM_PROVIDER=cpu in .env for the CPU-only budget
# STT_STREAM_PORT (default 8081) keeps the default stt service free
```

Pairs with the `tts` service over the same voice-api contract
(`docs/voice-api.md`); `VOICE_API_TOKEN` applies identically.

### Home Assistant (wyoming, HA-native)

```sh
docker compose --profile ha up -d
```

Point HA's Wyoming integration at `orin-nano:10300` (STT) and `:10200` (TTS).
The piper image ships its voice baked — no HF at boot
(`PIPER_OFFLINE=true` remains the hard airgap switch).

## Pool budget (Orin Nano 8 GB, Option A)

| component | pool cost |
|---|---:|
| parakeet packed CPU (STT) | 730 MiB |
| phonon-31m CPU (TTS) | 498 MiB |
| llama.cpp 4B q4_K_M + 8k ctx | ~3.0–3.6 GB |
| **total** | **~4.2–4.8 GB → ~1.5 GB headroom** |

8B q4 ≈ 5 GB fits with ~0.5 GB spare (tight; zram for the page cache).

## Verification (E4 acceptance)

1. `docker compose ps` — every service `healthy` (healthchecks are baked
   into the images).
2. `curl -H "Authorization: Bearer $VOICE_API_TOKEN" localhost:8080/v1/voices`
   — the model id; without the token → `401`.
3. Roundtrip: POST a wav to `:8080/v1/audio/transcriptions` → transcript;
   POST text to `:8888/v1/audio/speech` → wav (or `pcm` chunked).
4. Latency spot-check vs the bench rows (`benchmarks/history.jsonl`,
   serve-mode baseline `20261103_212058`): p50 within ~2× on the same board
   class is expected — the bench is clean-window, deployment shares the pool.

## Images

Pulled from Docker Hub (`narandill/<name>:<version>`) — see
[publishing](publishing.md) for the release flow. To run images built
locally on this host instead, `docker tag` them to the compose names or
point the `*_IMAGE` vars at the local tags.
