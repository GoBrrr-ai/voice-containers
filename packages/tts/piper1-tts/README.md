# piper1-tts (Jetson sm_87 + sm_110)

OpenAI-protocol speech synthesis: **piper-tts 1.8.0** (piper1-gpl) on our
**onnxruntime-gpu 1.31.1** build (sm_87 + sm_90a + sm_110a SASS, sm_110a PTX).
Serves the repo-wide [`voice-api v1`](../../docs/voice-api.md) contract —
`POST /v1/audio/speech` with **streamed per-sentence pcm chunks**
(`response_format: pcm`, chunked transfer) or complete wav.

```sh
docker run --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 --rm \
  -p 8091:8091 -v /data/piper:/data narandill/piper1-tts:0.2.0
```

The default voice (`en_US-libritts_r-medium`, 22.05 kHz) is **baked into the
image** — no downloads at boot, ever. `PIPER_CACHE` points at a volume to
persist alternates; `PIPER_OFFLINE`-style airgap is the default state.

## Endpoints

| endpoint | notes |
|---|---|
| `GET /health` | status + engine/voices |
| `GET /v1/voices`, `/v1/models` | baked voice, 22050 Hz, `streaming/speed: true` |
| `POST /v1/audio/speech` | `response_format` wav/pcm; `speed` 0.25–4.0 (length-scale) |

## Configuration

| variable | default | notes |
|---|---|---|
| `PIPER_PORT` | `8091` | HTTP listen port |
| `PIPER_VOICE` | `en_US-libritts_r-medium` | baked voice name (`<name>.onnx` in `PIPER_CACHE`) |
| `PIPER_USE_CUDA` | `true` | ORT CUDA execution provider |
| `PIPER_CACHE` | `/data/models/piper` | voice dir |
| `PIPER_LENGTH_SCALE` | voice default | phoneme length (<1 faster) |
| `PIPER_NOISE_SCALE` / `PIPER_NOISE_W` | voice default | generator noise |
| `PIPER_SPEAKER` | – | speaker id, multi-voice models |
| `VOICE_API_TOKEN` | – | bearer gate; unset = open |

## Verify

```sh
./test/verify_arch.sh narandill/piper1-tts:0.2.0   # cubins, no GPU
./test/smoke_tts.sh narandill/piper1-tts:0.2.0     # GPU wav + streamed pcm
```
