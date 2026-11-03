# parakeet-tdt-ort-stt (Jetson sm_87 + sm_110)

**NVIDIA parakeet-tdt-0.6b-v3** (CC-BY-4.0) as a plain-ONNX int4 pack
([efederici/parakeet-tdt-0.6b-v3-onnx-int4](https://huggingface.co/efederici/parakeet-tdt-0.6b-v3-onnx-int4))
on **our onnxruntime-gpu 1.31.1** (CUDA EP, sm_87 + sm_110a native SASS):

| Graph | Size | EP |
|---|---|---|
| `encoder-model.int4.onnx` (MatMulNBits, block 64) | 391 MB | **CUDA** |
| `decoder_joint-model.int8.onnx` (dynamic QOperator) | 18 MB | CPU (by design — no CUDA kernels for dynamic int8) |
| `nemo128.onnx` (128-bin mel featurizer) | 138 KB | CPU |

~409 MB total — the **Orin Nano 8 GB budget** variant with the featurizer
included (the soniqo/istupakov exports lack it). Same 0.6B v3 TDT as the
sherpa int8 row (`parakeet-tdt-stt`) and the parakeet.cpp q4_k row
(`parakeet-tdt-cpp-stt`) — three runtimes, one model.

Decode: TDT greedy loop ported from NeMo `GreedyTDTInfer` (durations
0–4 frames, LSTM state advances on emitted tokens only, `max_tokens_per_step`
10), verified end-to-end against jfk.flac — see
`task/research/parakeet-int4-ort-driver/`.

```sh
docker run --rm --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 \
  -v /audio:/audio:ro --entrypoint tdt-ort-transcribe \
  narandill/parakeet-tdt-ort-stt:0.1.0 /audio/x.wav
# CPU encoder: -e TDT_ORT_PROVIDER=cpu
```

## License

Weights CC-BY-4.0 (nvidia/parakeet-tdt-0.6b-v3); export Apache-2.0
(efederici); onnxruntime MIT.

## Server (voice-api v1)

```sh
docker run -d --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 -p 8080:8080 narandill/parakeet-tdt-ort-stt:0.1.0
curl -s localhost:8080/v1/voices | jq        # parakeet-tdt-0.6b-v3-int4 (MatMulNBits)
curl -s localhost:8080/v1/audio/transcriptions --data-binary @/audio/x.wav \
  -H 'Content-Type: audio/wav' -H 'X-Model: parakeet-tdt-0.6b-v3-int4'
```
16 kHz mono input required (raw-ORT driver, no resampler).
