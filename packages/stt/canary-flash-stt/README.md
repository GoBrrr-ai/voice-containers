# canary-flash-stt (Jetson sm_87 + sm_110, CPU int8)

**NVIDIA Canary-180M-Flash** (CC-BY-4.0) — 17-layer FastConformer encoder +
4-layer decoder, ~182M params, en/de/es/fr with ASR + X↔EN translation.
LibriSpeech-other 2.87 WER, SPGI 1.95. The strongest ASR in its class that
also fits a GPU budget (FP16 ~0.4 GB).

- Baked: sherpa-onnx **int8** export (encoder 132.7 MB + decoder 74.4 MB +
  tokens) on the CPU EP — verified aarch64 PyPI wheels.
- **No native streaming**: per-utterance after VAD chunking (Silero VAD
  upstream; sherpa-onnx VAD/endpointing also works). Decoder emits one
  token per step and returns log-softmax already.
- GPU path (later): the soniqo two-graph FP32/INT8 export runs on our
  onnxruntime-gpu 1.31.1 — but it ships **no featurizer graph** (the NeMo
  128-bin mel contract must be hand-implemented); see
  `task/research/stt-onnx-streaming-engines/summary.md`.

## Run

```sh
# transcription (en)
docker run --rm -v /data:/data --entrypoint canary-transcribe \
  narandill/canary-flash-stt:0.1.0 /audio/x.wav

# translation en -> de
docker run --rm -v /data:/data --entrypoint canary-transcribe \
  narandill/canary-flash-stt:0.1.0 /audio/x.wav --tgt de
```

**Positioning (report rank #4):** best GPU option once an FP16 bake lands;
as shipped (CPU int8) it is a fast accurate offline transcriber.
