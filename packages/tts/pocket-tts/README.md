# pocket-tts (Jetson sm_87 + sm_110, CPU)

**Kyutai Pocket TTS 100M** (CC-BY-4.0 weights, MIT code) via the **official
`pocket-tts` package** (torch-cpu; pure-python wheel + aarch64 torch).
Native streaming with adaptive chunks (first audio ~200 ms), zero-shot
voice cloning, 7 languages (en/fr/de/pt/it/es/nl), infinitely long inputs.

- Baked: the `english` bundle warmed into `/opt/hf-cache` at build time —
  the image runs fully offline.
- **Tiny English models** (Oct 2026): `english_18m_2026-10`,
  `english_7m_2026-10`, `english_2m_2026-10` — down to 1.9M params at 25×
  real time; pass `--language` (rebuild to re-bake, or keep the default
  100M bundle for quality).
- Voices: `alba marius javert jean fantine cosette eponine azelma` + clone
  any voice from a wav sample.
- The official CLI also ships `serve` (HTTP) — use it for a server; for an
  ORT-only minimal image, the community int8 ONNX export
  (miraclemind/pocket-tts-onnx, ~150 MB) is a footprint fallback.

## Run

```sh
# batch
docker run --rm -v /data:/data --entrypoint pocket-tts \
  narandill/pocket-tts:0.1.0 generate --text "Hello from Jetson Thor." --output-path /data/x.wav

# streaming server
docker run --rm -p 8000:8000 narandill/pocket-tts:0.1.0 serve
```

**Positioning (report rank #2):** the top CPU TTS pick — native streaming
plus voice cloning; torch runtime is the footprint price.

## Server (voice-api v1)

```sh
docker run -d --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 -p 8000:8000 narandill/pocket-tts:0.3.0
curl -s localhost:8000/v1/voices | jq        # 8 voices, 24 kHz, chunked streaming
curl -s localhost:8000/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"Hello from Jetson Thor.","voice":"alba","response_format":"pcm"}' -o /tmp/pk.pcm
```
Voice cloning stays on the official CLI server (`--entrypoint pocket-tts`,
`serve`, `POST /tts` with `voice_wav`). No speed knob upstream.
