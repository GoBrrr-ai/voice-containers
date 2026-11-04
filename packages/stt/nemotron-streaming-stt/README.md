# nemotron-streaming-stt (Jetson sm_87 + sm_110, CPU int8)

**NVIDIA Nemotron Speech Streaming EN 0.6B** (NVIDIA Open Model License) —
cache-aware streaming FastConformer RNNT, the best streaming ASR design in
the report. INT8 encoder 653 MB + decoder 7.3 + joiner 1.7 MB.

- **True native streaming** with a baked latency profile: this image ships
  the **160 ms lookahead** export — the graph does its own cache-aware
  chunking; feed audio via sherpa's stream API and decode when ready (never
  pre-chunk the audio yourself). 80/560/1120 ms profiles are separate tars
  (rebuild with one if a different latency/accuracy trade-off is wanted).
- Not related to our `nemotron-unified-asr` package (that is the
  non-streaming 0.6B unified model on parakeet.cpp GGUF; this is the
  streaming RNNT on sherpa-onnx).
- Avoid the Rekody re-export (880 MB encoder, PolyForm Shield license) —
  the k2-fsa release tar used here is canonical.

## Run

```sh
# voice-api v1 server (the default CMD — SSE partials via stream=true)
docker run -d --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 \
  -e NEMOSTREAM_PROVIDER=cuda -p 8080:8080 narandill/nemotron-streaming-stt:0.2.0
# CLI (in-process container-mode benches + humans)
docker run --rm -v /data:/data --entrypoint nemotron-stream-transcribe \
  narandill/nemotron-streaming-stt:0.2.0 /audio/x.wav
```

- **voice-api v1** (`rootfs/usr/local/bin/voice-server`, stdlib-only): `/health`,
  `/v1/voices` (`streaming: true`), `/v1/models`, and `POST
  /v1/audio/transcriptions` with **SSE partials** — `stream=true` (multipart
  field or query param) yields `event: partial {text}` on text change at the
  160 ms slice cadence, then one `event: final {text, duration}`. The
  canonical offline path (whole utterance + 0.2 s tail) serves the non-stream
  response shapes. Optional `VOICE_API_TOKEN` gate (/health exempt).
- **Measured (Thor, 500-utt pilot)**: TTFP p50 ~0.53 s (cuda) with ~26
  partials/utterance; RTF ~0.8 on BOTH providers — the tiny-chunk decode is
  launch-bound, so GPU ≈ CPU for this model; the win is the footprint
  (~394 MiB VRAM, smallest in the GPU roster) + true streaming, not
  throughput.

**Positioning (report rank #3):** best streaming design; at the memory
budget limit, CPU speed unproven until benched (A78AE is much slower than
the M-series numbers vendors quote).
