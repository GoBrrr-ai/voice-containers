# voice-api v1 — the unified STT/TTS HTTP contract

OpenAI-compatible audio endpoints for every non-wyoming voice image in this
repo. One contract for the whole roster; the two wyoming images
(`wyoming-faster-whisper`, `wyoming-piper1-tts`) stay HA-native and are out of
scope. Aligned with the OpenAI Audio API (2026-11) so stock SDKs work; deltas
are explicit at the bottom.

## Endpoints

| endpoint | task | notes |
|---|---|---|
| `GET /health` | all | `{"status":"ok", ...engine metadata}` |
| `GET /v1/voices` | all | **what THIS image can serve** — never a generic list |
| `GET /v1/models` | STT | OpenAI-style alias of `/v1/voices` (`object: "model"`) |
| `POST /v1/audio/transcriptions` | STT | multipart (`file` required) or raw WAV body |
| `POST /v1/audio/speech` | TTS | JSON body |

## `GET /v1/voices`

```json
{"object": "list", "data": [
  {"id": "kokoro", "object": "voice", "task": "tts",
   "sample_rate": 24000, "languages": ["en-us"], "streaming": true,
   "speed": true, "cloning": false, "engine": "kokoro", "version": "0.2.0"}
]}
```

STT entries describe the baked model (`id` = model id to send in the `model`
field, e.g. `parakeet-tdt-0.6b-v3-q8_0`); TTS entries enumerate baked voices.
`sample_rate`: STT input requirement; TTS output rate (always declared —
never assumed).

## `POST /v1/audio/transcriptions`

multipart/form-data (OpenAI subset), or a raw WAV body
(`Content-Type: audio/wav`):

| field | req | notes |
|---|---|---|
| `file` | ✓ | WAV (16-bit PCM) preferred; decoded in memory, never on disk |
| `model` | ✓ | must equal the `/v1/voices` id — `400 invalid_request_error` otherwise |
| `response_format` | | `json` (default), `text`, `verbose_json` |
| `language` | | honored where the model supports it (canary: `src_lang`/`tgt_lang` via env) |

- `json` → `{"text": "..."}`; `text` → plain text; `verbose_json` →
  `{task, text, duration}` (no segment/timestamp fabrications).
- `stream=true` is **accepted and ignored** on engines without partial
  support (OpenAI's own precedent for `whisper-1`); streaming partials are a
  fast-follow for `nemotron-unified-asr` / moonshine.
- Audio is decoded **in memory** (`io.BytesIO` + `wave`) — no temp files.

## `POST /v1/audio/speech`

JSON body (OpenAI subset):

| field | req | notes |
|---|---|---|
| `input` | ✓ | text |
| `voice` | | must be a `/v1/voices` id (`400` otherwise); engine default if omitted |
| `model` | | engine id, validated like STT |
| `response_format` | | `wav` (default, complete file) or `pcm` (raw s16le; **streamed via chunked transfer when the engine yields chunks**) |
| `speed` | | 0.25–4.0; honored by kokoro/piper-family, ignored elsewhere |

- `pcm` sample rate = the engine's native rate, in `X-Sample-Rate` (24 000
  everywhere; piper 22 050).
- Streaming engines (kokoro, phonon-31m, pocket) write chunks as generated —
  clients read `Transfer-Encoding: chunked` progressively, like OpenAI's
  chunked-transfer streaming. `response_format: "sse"` → `400` (not
  implemented; `audio` chunking is the low-overhead path).
- Synthesis is fully in-memory — **no `*.wav` temp files anywhere** (this is
  the point of voice-api v1: the wyoming-piper fork patch removed the last
  one).

## Per-image roster

| image | serves | `response_format` | streaming | voices / model id |
|---|---|---|---|---|
| `parakeet-tdt-cpp-stt` | STT | json/text/verbose_json | — | `parakeet-tdt-0.6b-v3-q4_k` (parakeet-server, C++ route patch) |
| `parakeet-redux-stt` | STT | 〃 | — | `redux-q8_0` |
| `phonon2-stt` | STT | 〃 | — | `phonon-2.5-q8_0` |
| `nemotron-unified-asr` | STT | 〃 | — (CLI presets 0.16–2.08 s; SSE partials fast-follow) | `nemotron-labs-unified-asr-0.6b-q8_0` |
| `canary-flash-stt` | STT | 〃 | — | `canary-180m-flash-int8` (en/de/es/fr + X↔EN translation via `CANARY_SRC_LANG`/`CANARY_TGT_LANG`) |
| `parakeet-tdt-stt` | STT | 〃 | — | `parakeet-tdt-0.6b-v3-int8` (25 languages) |
| `moonshine-streaming-stt` | STT | 〃 | — | `moonshine-streaming-medium` (CPU bundle) or `moonshine-streaming-medium-ort` (CUDA; auto-selected) |
| `parakeet-tdt-ort-stt` | STT | 〃 | — | `parakeet-tdt-0.6b-v3-int4` |
| `kokoro-tts` | TTS | wav/pcm | pcm chunked | af_heart, af_bella, bf_isabella, am_puck, bm_fable (24 kHz, `speed`) |
| `paradee-tts` | TTS | wav/pcm | — (single chunk) | af_heart, bf_isabella, am_puck, bm_fable (24 kHz) |
| `phonon31m-tts` | TTS | wav/pcm | pcm chunked (true frames) | `default` (24 kHz, CPU by design); legacy `GET /say` kept |
| `pocket-tts` | TTS | wav/pcm | pcm chunked | alba, marius, javert, jean, fantine, cosette, eponine, azelma (24 kHz; cloning stays on the official `pocket-tts serve`) |

**Known deviation — RESOLVED 2026-11-04:** the four parakeet-server images
(tdt-cpp, redux, phonon2, nemotron) accepted raw WAV bodies since
commit `f938ec5` (patch hunk 2 in
`builders/parakeet-cpp/files/server-model-routes.patch`; verified CPU-mode:
raw 200 + multipart regression + negative 400, transcript-identical paths).


## Implementation notes

- Every server is **stdlib-only** (`ThreadingHTTPServer`) — the runtime
  images ship no web framework; each `rootfs/usr/local/bin/voice-server`
  embeds the exact adapter core proven in `bench/engines/client.py` (padding
  rules, EP mixtures, the kokoro `pauses` guard, the TDT greedy loop).
- Engines are loaded once at startup; one request at a time (parakeet-server
  exposes `--concurrency`); every request path is allocation-in-memory.
