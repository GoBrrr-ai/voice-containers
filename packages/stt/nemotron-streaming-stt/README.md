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
docker run --rm -v /data:/data --entrypoint nemotron-stream-transcribe \
  narandill/nemotron-streaming-stt:0.1.0 /audio/x.wav
```

**Positioning (report rank #3):** best streaming design; at the memory
budget limit, CPU speed unproven until benched (A78AE is much slower than
the M-series numbers vendors quote).
