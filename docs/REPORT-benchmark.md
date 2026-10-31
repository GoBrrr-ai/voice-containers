# Benchmark Report — wave 3: harness commissioning + Thor baselines

Nov 1, 2026 · Thor AGX 128 GB (sm_110), JetPack 7.2.1 userspace, driver 580.00,
CUDA 13.2 SBSA containers · default DVFS (no `jetson_clocks`) · all runs on the
frozen manifests (`manifests/*_seed42.jsonl`, LibriSpeech test-clean).

Companion: `docs/REPORT-voice-engines.md` (builds + smokes), `task/benchmark-prd.md` (spec).

## 1. What was validated

The PRD §1–9 harness (`bench/`) end to end on-device:

- **Pipeline**: config → per-engine fresh container (`--entrypoint python3 client.py
  job_<eid>.json`) → per-utterance JSONL (crash-safe append) → host-side normalization
  (Whisper normalizer) → corpus WER/CER + RTF/latency distributions + probes.
- **Artifacts per run**: `utterances.jsonl`, `summary.json` (per-engine + corpus),
  `summary.csv`, `job_<eid>.json`, `container_<eid>.json`, `logs_<eid>.txt`
  (captured container output), `docker_stats_<eid>.csv`, `tegrastats.log`
  (100 ms samples: RAM/CPU/GPU/power), `env.json`.
- **Isolation**: every engine job runs in its own fresh container of the engine
  image itself — the client is mounted, nothing is installed at bench time.
- **Sanity gate (PRD 9.4)**: faster-whisper tiny-int8 → **WER 7.2%** on test-clean,
  inside the expected 5–8% band. PASS.

## 2. STT baseline — 500 utterances, LibriSpeech test-clean (seed 42)

`bench/configs/thor_stt_baseline.yaml` · run `20261031_160902_*` (5 × 500 = 2500/2500 ok)

| Engine (all q8_0) | Backend | WER | CER | SER | RTF p50 | RTF p95 | Latency p50 | Latency p95 | Halluc. |
|---|---|---|---|---|---|---|---|---|---|
| **nemotron-unified-asr 0.6b** | GPU (CUDA0) | **1.77%** | 0.47% | 22.4% | 0.086 | 0.213 | 0.554 s¹ | 0.694 s | 0.0 |
| **Phonon-2.5-english** | GPU | 1.99% | 0.61% | 26.2% | 0.080 | 0.197 | 0.511 s¹ | 0.638 s | 0.0 |
| **parakeet-redux (ternary)** | GPU | 2.49% | 0.83% | 32.0% | 0.080 | 0.193 | 0.516 s¹ | 0.633 s | 0.0 |
| **parakeet-redux packed** | CPU | 2.52% | 0.83% | 32.0% | 0.062 | 0.094 | 0.393 s¹ | 1.025 s | 0.0 |
| faster-whisper tiny-int8 | GPU (CT2) | 7.22% | 2.80% | 59.4% | 0.012 | 0.020 | 0.077 s | 0.160 s | 0.2% |

¹ parakeet latency = **CLI cold-start per utterance** (process spawn + ggml load +
infer) — this matches deployment usage (`stt-transcribe` per audio segment) but is
pessimistic vs. a resident server; whisper loads once per job.

Headlines:

- **Accuracy tier**: Nemotron 1.77% < Phonon-2 1.99% < redux 2.49% ≈ redux-packed-CPU
  2.52% ≪ whisper-tiny 7.22%. The three parakeet-family models are within ~0.75% of
  each other; the packed *CPU* ternary costs only +0.03% WER vs. GPU q8_0 — a very
  attractive zero-GPU tier.
- **Speed tier**: whisper tiny is latency king when resident (77 ms p50). The parakeet
  CLIs pay ~0.4–0.5 s model load per utterance; RTF *excluding* load (from the
  CPU-packed p95 ≈ inference only) still lands ≪ real time.
- Zero hallucinations flagged on the parakeet family; whisper 0.2% (PRD 6.1 detector).

## 3. TTS baseline — 444 sentences, LibriSpeech test-clean text (seed 42)

See §3 note: v1 uses LibriSpeech transcripts as prompts (LibriTTS-R text fast-follow).

<!-- TTS_BASELINE_TABLE -->

## 4. Determinism (PRD 9.1 #7)

Two full 500-utt baseline runs, same manifest/seed/config:

| Engine | WER run 1 | WER run 2 | Δ |
|---|---|---|---|
| parakeet_cpp_0 (nemotron GPU) | 1.77% | 1.77% | **exact** |
| parakeet_cpp_1 (redux GPU) | 2.49% | 2.49% | **exact** |
| parakeet_cpp_2 (phonon2 GPU) | 1.99% | 1.99% | **exact** |
| parakeet_cpp_3 (redux CPU) | (broken run) | 2.52% | — |
| faster_whisper_4 (tiny-int8) | 7.20% | 7.22% | 1 word / 10 571 |

Parakeet-family decoding is **exactly deterministic** across processes and runs.
Whisper/CT2 shows ±1-word GPU float jitter (kernel reduction order) — engine-level,
not harness-level. The harness itself is deterministic.

## 5. Commissioning bugs found and fixed (all in the harness, none in images)

1. `TegraStats.stop` shadowed by `self.stop = False` → `TypeError: 'bool' object is
   not callable` after every run (results never aggregated).
2. Relative paths handed to `docker -v` → containers died instantly; runner now
   resolves everything against the repo root.
3. Engine adapter contracts verified against the actual image packages: kokoro's
   `create_stream` is async (driven incrementally now, TTFA preserved); ptts wants
   `quant=q8_0` and honors `PARAKEET_DEVICE=cpu` env for packed-ternary models;
   parakeet subprocess return codes are checked (empty output can't fake `ok`).
4. Client WAV IO moved to stdlib `wave` — no `soundfile` dependency in any image.
5. Per-engine ids (`parakeet_cpp_0..3`) — the same adapter appears once per model
   variant; WER is computed per engine, never pooled.
6. STT images gained `python3` (bench-client contract); model alias `tiny-int8`
   resolved to `rhasspy/faster-whisper-tiny-int8` (the wyoming fork's registry).

## 6. Infrastructure

- **38 `futureproofhomes/*` tags** built locally by `scripts/sweep-build.sh`
  (2 shared builders + 9 engine projects, `-thor`/`-orin`/version/latest).
- Shared builders replay from BuildKit cache; the parakeet builder now pins
  upstream SHA `2eac2e77…` (reproducible; carries the MelKernel GPU-stream fix).
- Bench venv: `bench/.venv` (py3.14) — `datasets[audio]` + torchcodec for the
  frozen manifests; 16 kHz mono PCM16 cache ≈ 600 MB.

## 7. Fast-follow (documented, not blocking)

- §10 comparison matrix + HTML dashboard; §7.3 off-device quality (UTMOS, round-trip
  WER on the produced TTS audio — the WAVs are already written to `results/*/audio/`).
- LibriTTS-R text prompts via a streaming loader.
- Realtime streaming pass (chunked TTFA for STT; `--stream` latency presets).
- Orin (sm_87) side of the matrix.
- Resident-server mode for parakeet (TTFA without CLI cold-start).
