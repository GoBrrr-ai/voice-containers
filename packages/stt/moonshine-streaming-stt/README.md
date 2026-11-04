# moonshine-streaming-stt (Jetson sm_87 + sm_110, CPU-only)

**Moonshine Streaming Medium 245M** (MIT) — native incremental streaming
ASR via the **official `moonshine-voice` runtime** (aarch64 manylinux wheel
with the ORT C++ core inside). Open-ASR-avg WER 6.65% (float reference,
arch id 5); Small 123M (7.84%) and Tiny 34M (12.0%) bundles exist for
tighter budgets.

- Baked: the quantized `medium-streaming-en` 5-graph `.ort` bundle
  (encoder/adapter/cross_kv/decoder_kv/frontend, ~270 MB) from
  `download.moonshine.ai`, stage `quantized_26_08_21`.
- CPU-only by design (A78AE + Thor); ~270 MB files + 50–300 MB runtime.
- **True streaming**: feed PCM via `Transcriber.create_stream()` +
  `stream.add_audio()` — partials arrive on an interval (default 0.5 s),
  exposed over voice-api as **SSE partials**: `POST
  /v1/audio/transcriptions` with `stream=true` (multipart field or query
  param) → `text/event-stream` (`partial` events on text change at the
  320 ms slice cadence, then one `final` with `text` + `duration`). The
  CUDA `ort` variant serves offline-only and accept-and-ignores `stream`
  (finals land on `.on_line()`; `transcribe_without_streaming()` for files).
- Keyterm/context biasing supported on streaming architectures
  (`set_keyterms`/`set_context`).

## Run

```sh
# batch
docker run --rm -v /data:/data --entrypoint moonshine-transcribe \
  narandill/moonshine-streaming-stt:0.1.0 /audio/x.wav

# live partials (python, mounted script)
#   t = Transcriber(model_path="/models/moonshine-streaming-en", model_arch=5)
#   s = t.create_stream(); s.add_audio(chunk) ...
```

**Positioning (report rank #1):** the top streaming ASR pick for Orin —
fits comfortably, official ORT-based runtime, live partial text. GPU FP16
via the community five-graph export is untested (E-queue).

## Server (voice-api v1)

```sh
docker run -d --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 -p 8080:8080 narandill/moonshine-streaming-stt:0.2.0
curl -s localhost:8080/v1/voices | jq   # served variant: ort (CUDA) or streaming (CPU)
```
Two baked engines — `MOONSHINE_ENGINE=ort|streaming` picks one (default:
`ort` when `DEVICE=cuda`, else the CPU bundle); the model id reflects the
active variant.
