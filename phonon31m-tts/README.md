# phonon31m-tts (CPU-only — smallest image in the fleet)

**gradium/phonon-31m-en** (Apache-2.0; 31 M params) via the `ptts` 0.6.0
**aarch64 manylinux wheel** (gradium's Rust "xn" engine). **No CUDA at all** —
plain `ubuntu:24.04` base; the GPU stays free for the LLM.

- Baked: `model.q8.gguf` (61.5 MB) + `default-voice.safetensors` + tokenizers.
- RAM: ~0.10–0.15 GB — the RAM champion.
- **True streaming**: `tts.stream()` yields float32 PCM chunks as generated.
  Open TTS LB: TTFA p95 **28.4 ms** (p50 19.5) — #1 on CPU; RTFx 16.26; EN WER 0.87.
- Voices: `default` (Easton); more at [gradium/phonon-voices](https://huggingface.co/gradium/phonon-voices)
  (note: the 40 m siblings are CC-BY-NC — this one is Apache).

## Run

```sh
# HTTP streaming server (chunked WAV; frames written as generated)
docker run --rm -p 8880:8880 futureproofhomes/phonon31m-tts-thor:0.1.0
curl "http://localhost:8880/say?text=Hello%20from%20Thor." -o speech.wav   # streams

# CLI
docker run --rm -v /data:/data --entrypoint ptts-say \
  futureproofhomes/phonon31m-tts-thor:0.1.0 "Hello from Jetson Thor." -o /data/x.wav
```

## Library use (recommended for the assistant)

```python
from ptts import TTS

tts = TTS("/models/phonon-31m-en", lang="en", quant="q8")
with tts.stream("Speech can play while the rest is being generated.") as audio:
    for pcm in audio:           # float32 mono chunks
        player.write(pcm, audio.sample_rate)
```

CLI equivalent: `ptts --model /models/phonon-31m-en --lang en --quant q8 --voice default "text" -o x.wav`.

## Env

`PTTS_MODEL=/models/phonon-31m-en`, `PTTS_LANG=en`, `PTTS_QUANT=q8`,
`TTS_PORT=8880`. The stdlib-only `tts-stream-server` (no extra deps) streams
chunked WAV on `/say?text=...&voice=...` and answers `/health`.

**Wyoming**: upstream ships a Home-Assistant Wyoming server
([phonon-home-assistant](https://github.com/gradium-ai/phonon-home-assistant),
port 10200) built from their Rust source — wire it to the same baked
`/models` dir if you want HA-native TTS.
