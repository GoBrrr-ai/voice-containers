# faster-whisper-stt (Jetson sm_87 + sm_110)

OpenAI-protocol speech recognition: **faster-whisper 1.2.1** on our
**CTranslate2 4.8.4** build (sm_87 + sm_110 SASS/PTX, CUDA 13.2 SBSA).
Serves the repo-wide [`voice-api v1`](../../docs/voice-api.md) contract —
multipart or raw-WAV `POST /v1/audio/transcriptions`, plus **SSE partials**
(`stream=true`: a `partial` per decode window, one `final`).

```sh
docker run --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 --rm \
  -p 8090:8090 -v /data/wfw:/data narandill/faster-whisper-stt:0.2.0
```

The model resolves at boot from `WHISPER_MODEL` (HF repo, local path, or the
bare `tiny|base|small|medium-int8` aliases); downloads land in
`WHISPER_DATA_DIR` — set `WHISPER_OFFLINE=true` for the hard airgap switch.

## Endpoints

| endpoint | notes |
|---|---|
| `GET /health` | status + engine/device/model |
| `GET /v1/voices`, `/v1/models` | the baked model id, 99 languages, `streaming: true` |
| `POST /v1/audio/transcriptions` | `response_format` json/text/verbose_json; `stream=true` → SSE |

## Configuration

| variable | default | notes |
|---|---|---|
| `WHISPER_PORT` | `8090` | HTTP listen port |
| `WHISPER_MODEL` | `tiny-int8` | alias, HF id or local path |
| `WHISPER_BEAM_SIZE` | `1` | decode beams |
| `WHISPER_COMPUTE_TYPE` | `default` | CT2 compute type |
| `WHISPER_LANGUAGE` | `en` | ISO code, or `auto` |
| `WHISPER_DEVICE` | `cuda` | `cuda` / `cpu` |
| `WHISPER_DATA_DIR` | `/data/models/faster-whisper` | model cache |
| `WHISPER_INITIAL_PROMPT` | – | decoder biasing prompt |
| `WHISPER_VAD_FILTER` | – | truthy → skip silence via VAD |
| `WHISPER_OFFLINE` | – | `true` → `local_files_only` |
| `VOICE_API_TOKEN` | – | bearer gate; unset = open |

## Verify

```sh
./test/verify_arch.sh narandill/faster-whisper-stt:0.2.0   # cubins, no GPU
./test/smoke_stt.sh narandill/faster-whisper-stt:0.2.0     # GPU round trip + SSE
```
