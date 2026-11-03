# parakeet-tdt-stt (Jetson sm_87 + sm_110, CPU int8)

**NVIDIA Parakeet TDT 0.6B v3** (CC-BY-4.0) — Token-and-Duration Transducer,
**25 languages**, the most accurate model in its cohort. INT8 sherpa-onnx
export: encoder 652 MB + decoder 11.8 + joiner 6.4 MB (~0.65–0.8 GB
loaded — at the report's 0.8 GB budget limit).

- **Offline per utterance** (20–30 s max per pass) — VAD-chunk longer
  audio upstream. No live partials.
- A parakeet.cpp **GGUF also exists** (f16 1.44 GB / q4_k 675 MB /
  q8_0 941 MB in mudler's repo) if a GPU-capable variant is ever wanted —
  our three parakeet.cpp packages show the pattern (quantized-variant
  testing is E3 in the deferred queue).

## Run

```sh
docker run --rm -v /data:/data --entrypoint parakeet-tdt-transcribe \
  narandill/parakeet-tdt-stt:0.1.0 /audio/x.wav
```

**Positioning (report rank #5):** accuracy + language-coverage pick when
multilingual ASR is needed and memory is tight; batch-after-VAD only.

## Server (voice-api v1)

```sh
docker run -d --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 -p 8080:8080 narandill/parakeet-tdt-stt:0.2.0
curl -s localhost:8080/v1/voices | jq        # parakeet-tdt-0.6b-v3-int8, 25 languages
curl -s localhost:8080/v1/audio/transcriptions -F file=@/audio/x.wav -F model=parakeet-tdt-0.6b-v3-int8
```
