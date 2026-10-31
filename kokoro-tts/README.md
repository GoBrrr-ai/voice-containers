# kokoro-tts (Jetson sm_87 + sm_110)

**hexgrad/Kokoro-82M v1.0** (Apache-2.0) via [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)
0.6.1 on **our onnxruntime-gpu 1.31.1 build** (sm_87 + sm_90a + sm_110a SASS,
sm_110a PTX). CUDA EP first, CPU EP fallback.

- Baked: `model_uint8.onnx` (169 MB int8, onnx-community export) + `voices-v1.0.bin`
  (27 MB, 54 voices).
- RAM: ~0.15–0.25 GB (q8). Open TTS LB: EN WER 0.69, UTMOS ~4.5 class; 8 languages
  (en-us/en-gb/es/fr/hi/it/ja/pt-br — no Polish).
- Streaming: `create_stream()` — phoneme-chunked async generator; first audio
  after the first chunk (~0.3–0.5 s class). Same graph as batch → identical quality.

## Run

```sh
# batch to wav
docker run --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 --rm -v /data:/data \
  narandill/kokoro-tts:0.1.0 kokoro-say "Hello from Jetson Thor." -o /data/kokoro.wav

# true streaming (prints TTFA)
docker run --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 --rm -v /data:/data \
  narandill/kokoro-tts:0.1.0 kokoro-stream-demo "Streaming is native here." -o /data/stream.wav
```

## Library use (recommended for the assistant)

```python
import onnxruntime as rt
from kokoro_onnx import Kokoro

sess = rt.InferenceSession("/opt/kokoro/model_uint8.onnx",
                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
kokoro = Kokoro.from_session(sess, "/opt/kokoro/voices-v1.0.bin")

async for samples, sr in kokoro.create_stream(text, voice="af_heart", lang="en-us"):
    ...  # int16-ready float32 [-1, 1] PCM frames @ 24000 Hz
```

## Env

`KOKORO_MODEL` / `KOKORO_VOICES` (baked), `KOKORO_VOICE` (default `af_heart`),
`KOKORO_LANG` (`en-us`), `KOKORO_SPEED` (`1.0`).

**Note:** kokoro-onnx is installed `--no-deps` and our GPU ort wheel is
force-installed LAST — the `[gpu]` extra can never clobber the sm_87/sm_110
build (same pattern as piper in the former combined wyoming image).
