# Benchmark harness — edge-containers

Implements `task/benchmark-prd.md` §1-9 (core). **§10 matrix/dashboard and
§7.3 off-device quality scoring (UTMOS, round-trip WER) are fast-follow** — the
JSONL schema already carries their fields.

## Design (PRD §1, non-negotiable)

- Datasets and engines are **plugins behind fixed interfaces** — no engine- or
  dataset-specific branches in runner/metrics.
- One fixed normalizer (Whisper English/Basic) applied to references and
  hypotheses; name+version recorded everywhere.
- Subsets frozen into **committed manifests** (seed 42; STT stratified by
  speaker; TTS stratified by length bucket + 20 longest sentences).
- `time.perf_counter_ns` everywhere; device sync before stopping timers.
- Per-utterance JSONL appended immediately (crash-safe).
- Each (engine × backend × dataset) job runs in a **fresh container** (the
  engine image itself, with `bench/engines/client.py` mounted) — process
  isolation per PRD §2; parent probes sample `tegrastats` + `docker stats`.

## Engines (adapters in `bench/engines/client.py`)

| Adapter | Image | Backends | Notes |
|---|---|---|---|
| `faster_whisper` | `futureproofhomes/wyoming-faster-whisper-thor` | cuda, cpu | CT2 GPU; model via HF cache volume |
| `parakeet_cpp` | the 3 STT images (unified/redux/phonon) | cuda, cpu | CLI subprocess; `--stream` for streaming runs |
| `kokoro` | `futureproofhomes/kokoro-tts-thor` | cuda, cpu | `create` + `create_stream` |
| `ptts` | `futureproofhomes/phonon31m-tts-thor` | cpu (by design) | `tts.stream` |
| `paradee` | `futureproofhomes/paradee-tts-thor` | cuda, cpu | int8 graph |
| `piper` | `futureproofhomes/wyoming-piper1-tts-thor` | cuda, cpu | `PiperVoice.synthesize_wav` |

## Usage

```bash
python -m venv bench/.venv && bench/.venv/bin/pip install -r bench/requirements.txt
source bench/.venv/bin/activate

python -m bench.cli prepare --dataset librispeech --split test-clean --lang en --n 500 --seed 42 --task stt
python -m bench.cli prepare --dataset libritts_r --split test.clean --lang en --n 500 --seed 42 --task tts
python -m bench.cli selfcheck

python -m bench.cli run --config bench/configs/thor_stt_baseline.yaml
python -m bench.cli report results/<run_id>
python -m bench.cli compare results/<run_a> results/<run_b> --out compare.md
```

## Config (PRD 9.2)

See `configs/thor_stt_baseline.yaml` and `configs/thor_tts_baseline.yaml` —
switching dataset/task/engines is config-only. Power profiles (`locked` via
`sudo jetson_clocks` vs `default` DVFS) are recorded in `env.json`; never
compare across profiles (PRD 7A.2).

## Acceptance (PRD 9.4 — v1 coverage)

1. ✅ dataset swap = config change; identical schemas
2. ✅ CPU-only engine (ptts) + CUDA engines from one config; unsupported pairs → `skipped`
3. ✅ GPU timings synchronized (client-side `_sync_cuda`)
4. ✅ LibriSpeech sanity gate — pilot: tiny-int8 WER 7.2% (expected 5–8% band)
5. ✅ distribution stats (p50/90/95/99, std, IQR, MAD, low-confidence flag)
6. ✅ crash-safe JSONL append + per-engine container logs captured to the run dir
7. ⏳ determinism check (run pilot twice, compare WER) — after baselines
8. ⏳ Orin-side run (no Orin on this bench)

### v1 measurement caveats (documented deviations)

- **parakeet adapters measure CLI cold-start per utterance** (`parakeet-cli` subprocess
  per row) — this matches deployment usage (`stt-transcribe` per audio segment) but is
  pessimistic vs. a resident server. RTF includes process spawn + model load.
- **TTS text v1 = LibriSpeech test-clean transcripts** (444 sentences); LibriTTS-R
  text is a fast-follow (avoids the 30 GB dataset download — TTS needs text only).
- §10 comparison matrix/dashboard and §7.3 off-device quality scoring (UTMOS /
  round-trip WER) are fast-follow; the JSONL schema already carries their fields.
