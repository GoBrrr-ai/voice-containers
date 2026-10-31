# wyoming-faster-whisper (Jetson sm_87 + sm_110)

Wyoming STT server: **wyoming 1.10.2** + **wyoming-faster-whisper 3.8.1** +
**faster-whisper 1.2.1** on **our CTranslate2 4.8.4 build** (sm_87 + sm_90a +
sm_110a SASS, sm_110a PTX; CUDA 13.2 SBSA). The STT half of the former combined
wyoming image — same `WHISPER_*` env contract, so Home Assistant config is a
drop-in.

- Models are pulled at first start (default `tiny-int8`; any HF faster-whisper
  repo works via `WHISPER_MODEL`).
- Benchmarked on AGX Thor: **7.22% WER**, RTF p50 **0.013**, **613 MiB** peak
  (LibriSpeech test-clean, 500 utts) — the speed baseline of the stack.

## Run

```sh
docker run --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 --rm -p 10300:10300 \
  -v /data/wfw:/data narandill/wyoming-faster-whisper:0.1.0
```

Home Assistant (Wyoming integration): add host `127.0.0.1`, port `10300`.

## Env

| Var | Default | What |
|---|---|---|
| `WHISPER_MODEL` | `tiny-int8` | model alias or HF repo |
| `WHISPER_BEAM` | `1` | beam size |
| `WHISPER_LANGUAGE` | `en` | language hint |

## Test

```sh
./test/verify_arch.sh narandill/wyoming-faster-whisper:0.1.0   # cubins, no GPU
./test/smoke_whisper.sh narandill/wyoming-faster-whisper:0.1.0 # GPU round-trip
```
