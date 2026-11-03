# wyoming-piper1-tts (Jetson sm_87 + sm_110)

Wyoming TTS server: **wyoming 1.10.2** + **wyoming-piper 2.5.2** +
**piper-tts 1.8.0** (piper1-gpl) on **our onnxruntime-gpu 1.31.1** build
(sm_87 + sm_90a + sm_110a SASS, sm_110a PTX; CUDA 13.2 SBSA). The TTS half of
the former combined wyoming image — same `PIPER_*` env contract (fork), so
Home Assistant config is a drop-in.

- Voice: default `en_US-libritts_r-medium`; any piper1 voice works via
  `PIPER_VOICE`. Voices download into the `/data` volume on first start
  (`PIPER_UPDATE_VOICES` refreshes them; `PIPER_OFFLINE=true` forbids it).
- CUDA EP is the default (`PIPER_USE_CUDA=true`); our ORT wheel carries the
  sm_87/sm_110 cubins, so the same image serves Orin and Thor.
- Benchmarked on AGX Thor (444 sentences, LibriSpeech test-clean text):
  **RTF p50 0.036** (p95 0.126 — ~27× realtime), **TTFA p50 0.168 s /
  p90 0.184 s**. Single-chunk engine: TTFA equals synthesis completion.

## Run

```sh
docker run --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 --rm -p 10200:10200 \
  -v /data/piper:/data narandill/wyoming-piper1-tts:0.1.0
```

Home Assistant (Wyoming integration): add host `127.0.0.1`, port `10200`
(advertised over zeroconf as `wyoming-piper-tts`).

## Env

| Var | Default | What |
|---|---|---|
| `PIPER_VOICE` | `en_US-libritts_r-medium` | piper1 voice (downloaded to `PIPER_CACHE`) |
| `PIPER_PORT` | `10200` | Wyoming TCP port |
| `PIPER_USE_CUDA` | `true` | ONNX Runtime CUDA EP |
| `PIPER_CACHE` | `/data/models/piper` | voice download/cache dir |
| `PIPER_UPDATE_VOICES` | `true` | refresh voices on start |
| `PIPER_OFFLINE` | `true`=forbid downloads | pass `--local-files-only` |
| `PIPER_LENGTH_SCALE` / `PIPER_NOISE_SCALE` / `PIPER_NOISE_W` | `1.0` / `0.667` / `0.333` | prosody knobs |
| `PIPER_SPEAKER` | — | multi-speaker voice selection |
| `PIPER_ZEROCONF_NAME` | `wyoming-piper-tts` | zeroconf advertisement |
| `HF_HOME` | `/data/hf-cache` | HF download cache |
| `PIPER_DEBUG` | — | verbose logging |

## Test

```sh
./test/verify_arch.sh narandill/wyoming-piper1-tts:0.1.0   # cubins, no GPU
./test/smoke_piper.sh narandill/wyoming-piper1-tts:0.1.0   # GPU wyoming round-trip
```
