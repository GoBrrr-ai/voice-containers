# paradee-tts (Jetson sm_87 + sm_110)

**sahilmahendrakar/Paradee-8M-v1.2** (Apache-2.0; 7.88 M params) — an int8
ONNX **distillation of Kokoro-82M** (paper 2610.06817): af_heart UTMOS 4.40
(teacher 4.52), WER 5.8% vs Kokoro's 5.7% — an 8.5 MB voice that runs
anywhere.

- Runtime: the `paradee` package (misaki G2P — Kokoro's own phonemizer — no
  torch) on **our onnxruntime-gpu 1.31.1** (sm_87 + sm_110a). The int8 graph
  also runs CPU-only at 62× realtime on a single thread (upstream card) —
  the safe fallback.
- Baked: `paradee_int8.onnx` (8.5 MB) warmed into `/opt/hf-cache` at build
  time (no runtime download).
- **No native streaming** — the graph synthesizes whole utterances (TTFA
  ~1 s). `paradee-stream` implements **sentence-chunked streaming**: chunks at
  sentence boundaries and emits WAV frames per chunk.
- Voices (v1.2): `af_heart`, `am_puck` (US) and `bf_isabella`, `bm_fable`
  (UK — misaki British G2P, loaded lazily per accent); 24 kHz output.

## Run

```sh
# batch (default voice af_heart)
docker run --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 --rm -v /data:/data \
  narandill/paradee-tts:0.2.0 paradee-say "Hello from Jetson Thor." -o /data/p.wav

# British voice
docker run ... narandill/paradee-tts:0.2.0 paradee-say "The bath is full." --voice bf_isabella -o /data/p.wav

# sentence-chunked streaming (prints first-chunk TTFA)
docker run ... narandill/paradee-tts:0.2.0 paradee-stream "First sentence. Second one." -o /data/ps.wav
```

## Library use

```python
from paradee import Paradee, SAMPLE_RATE
tts = Paradee(version="1.2", quantized=True)
audio = tts("Paradee is a small voice that runs anywhere.", voice="af_heart")  # float32 @ 24 kHz
```

## Env

`PARADEE_MODEL_VERSION=1.2` (baked cache), `HF_HOME=/opt/hf-cache`.

**Positioning (Rev-2 ranking, honorable mention):** use Kokoro-82M for
streaming quality or phonon-31m for ultra-low TTFA; Paradee is the
minimum-footprint voice (reply caching, tiny pods). Not a kokoro-onnx
drop-in: its graph takes `input_ids int64 [1,T≤512]` + `speed` + `voice`
and returns `waveform [1,samples] @ 24 kHz` — the `paradee` package wraps
that.
