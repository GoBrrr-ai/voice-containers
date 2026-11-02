"""Engine client — copied/mounted INTO each engine container (spec §4).

Reads a job spec (JSON on argv), runs the manifest against one engine, and
appends per-utterance records to utterances.jsonl (+ events.jsonl for streams,
WAVs to audio/ for TTS). Never computes metrics; never times itself — the
client DOES stamp perf_counter_ns around engine calls because it is the only
process that can synchronize the device before stopping the clock (spec 5.1/4.3).

Usage: python client.py <job.json>
"""

from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path


def _now_ns() -> int:
    return time.perf_counter_ns()


def _sync_cuda() -> None:
    """Real device sync before stopping a timer (spec 4.3).

    Every engine API used here is synchronous, but the contract is enforced
    anyway via cudart so a future async adapter cannot silently under-report.
    On CPU-only images there is no cudart and this is a no-op.
    """
    try:
        import ctypes
        import ctypes.util

        rt = getattr(_sync_cuda, "_rt", None)
        if rt is None:
            name = ctypes.util.find_library("cudart") or "libcudart.so.13"
            rt = _sync_cuda._rt = ctypes.CDLL(name)
        rc = rt.cudaDeviceSynchronize()
        if rc == 0:
            return
    except Exception:
        pass  # no CUDA runtime in this image — nothing to synchronize


# ---------------------------------------------------------------- STT engines
class FasterWhisper:
    name = "faster_whisper"
    backends = {"cuda", "cpu"}
    supports_streaming = False
    required_sample_rate = 16000

    def __init__(self, opts):
        from faster_whisper import WhisperModel

        self.m = WhisperModel(
            opts["model"],
            device=opts.get("device", "cuda"),
            compute_type=opts.get("compute_type", "int8"),
        )

    def actual_backend(self) -> str:
        return self.opts_backend

    def transcribe(self, pcm, sr: int, wav_path: str | None = None):
        import numpy as np

        segs, info = self.m.transcribe(
            np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0,
            language=self.lang,
            beam_size=1,
        )
        text = " ".join(s.text for s in segs).strip()
        return text

    def load(self, opts):
        self.opts_backend = opts.get("device", "cuda")
        self.lang = opts.get("language") or opts.get("lang", "en")


class ParakeetCpp:
    """parakeet.cpp CLI — works for all 3 STT images (unified/redux/phonon)."""

    name = "parakeet_cpp"
    backends = {"cuda", "cpu"}
    supports_streaming = True
    required_sample_rate = 16000

    def __init__(self, opts):
        self.model = opts["model"]
        self.device = opts.get("device", "")
        self.stream_latency = opts.get("stream_latency", "")
        self.lang = opts.get("lang") or opts.get("language", "")
        self.vad_model = opts.get("vad_model", "")

    def load(self, opts):
        pass

    def actual_backend(self) -> str:
        return self.device or "cuda"

    def transcribe(self, pcm, sr: int, wav_path: str | None = None):
        import subprocess
        import tempfile
        import os

        if wav_path is None:
            fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            import wave

            with wave.open(wav_path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(pcm)
        cmd = [
            "/opt/parakeet-dist/bin/parakeet-cli",
            "transcribe",
            "--model",
            self.model,
            "--input",
            wav_path,
        ]
        env = {**os.environ}
        if self.device and self.device != "cuda":
            # images default to GPU (ggml CUDA0); "cuda" means "leave default".
            # Explicit values pass through as env (honored even by packed-ternary
            # models, which the --device flag alone does not switch to CPU) and
            # as the flag for the regular backends.
            cmd += ["--device", self.device]
            env["PARAKEET_DEVICE"] = self.device
        if self.lang:
            cmd += ["--lang", self.lang]
        if self.stream_latency:
            cmd += ["--stream", "--stream-latency", self.stream_latency]
        elif self.vad_model:
            cmd += ["--vad", "--vad-model", self.vad_model]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        if out.returncode != 0:
            tail = (out.stderr or out.stdout or "").strip().splitlines()[-3:]
            raise RuntimeError(
                f"parakeet-cli exit {out.returncode}: {' | '.join(tail)}"
            )
        lines = [
            line
            for line in out.stdout.splitlines()
            if line.strip() and not line.startswith("[stream]")
        ]
        return " ".join(lines).strip()

    def stream_feed(self, chunk):
        raise NotImplementedError("v1: streaming via whole-file --stream runs")


# --------------------------------------------------------- 2026-11 CPU engines
class MoonshineStreaming:
    """Moonshine Streaming Medium 245M — official moonshine-voice runtime.

    Native incremental streaming exists (create_stream), but v1 benches the
    offline per-utterance path (transcribe_without_streaming) so WER/RTF are
    comparable with the other engines; the bundle is the quantized
    medium-streaming-en .ort set baked at /models/moonshine-streaming-en.
    """

    name = "moonshine"
    backends = {"cpu"}
    supports_streaming = False
    required_sample_rate = 16000

    def __init__(self, opts):
        self.model_dir = opts.get("model_dir", "/models/moonshine-streaming-en")
        self.arch = int(opts.get("model_arch", 5))
        from moonshine_voice import ModelArch, Transcriber

        # built once: ORT session init per utterance would dominate RTF
        self._t = Transcriber(
            model_path=self.model_dir, model_arch=ModelArch(self.arch)
        )

    def load(self, opts):
        pass

    def actual_backend(self) -> str:
        return "cpu"

    def transcribe(self, pcm, sr: int, wav_path: str | None = None):
        import numpy as np

        pcm = np.asarray(pcm, dtype="float32")
        if sr != 16000:
            import soxr

            pcm = soxr.resample(pcm, sr, 16000)
            sr = 16000
        res = self._t.transcribe_without_streaming(pcm.tolist(), 16000)
        return " ".join(
            getattr(line, "text", "") for line in getattr(res, "lines", [])
        ).strip()


class CanaryFlash:
    """NVIDIA Canary-180M-Flash int8 — sherpa-onnx offline nemo-canary.

    Per-utterance after VAD (no native streaming); en/de/es/fr ASR plus
    bidirectional X↔EN translation via src_lang/tgt_lang.
    """

    name = "canary"
    backends = {"cpu"}
    supports_streaming = False
    required_sample_rate = 16000

    def __init__(self, opts):
        self.encoder = opts["encoder"]
        self.decoder = opts["decoder"]
        self.tokens = opts["tokens"]
        self.src = opts.get("src_lang", "en")
        self.tgt = opts.get("tgt_lang", "en")
        import sherpa_onnx

        self._r = sherpa_onnx.OfflineRecognizer.from_nemo_canary(
            encoder=self.encoder,
            decoder=self.decoder,
            tokens=self.tokens,
            num_threads=1,
            provider="cpu",
            src_lang=self.src,
            tgt_lang=self.tgt,
        )

    def load(self, opts):
        pass

    def actual_backend(self) -> str:
        return "cpu"

    def transcribe(self, pcm, sr: int, wav_path: str | None = None):
        import numpy as np

        stream = self._r.create_stream()
        stream.accept_waveform(sr, np.asarray(pcm, dtype="float32"))
        self._r.decode_stream(stream)
        return stream.result.text.strip()


class ParakeetTdt:
    """NVIDIA Parakeet TDT 0.6B v3 int8 — sherpa-onnx offline nemo_transducer.

    25 languages, per-utterance after VAD; ≤ ~30 s per pass (LibriSpeech
    utterances fit).
    """

    name = "parakeet_tdt"
    backends = {"cpu"}
    supports_streaming = False
    required_sample_rate = 16000

    def __init__(self, opts):
        import sherpa_onnx

        self._r = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=opts["encoder"],
            decoder=opts["decoder"],
            joiner=opts["joiner"],
            tokens=opts["tokens"],
            num_threads=1,
            provider="cpu",
            decoding_method="greedy_search",
            model_type="nemo_transducer",
        )

    def load(self, opts):
        pass

    def actual_backend(self) -> str:
        return "cpu"

    def transcribe(self, pcm, sr: int, wav_path: str | None = None):
        import numpy as np

        stream = self._r.create_stream()
        stream.accept_waveform(sr, np.asarray(pcm, dtype="float32"))
        self._r.decode_stream(stream)
        return stream.result.text.strip()


class NemotronStreaming:
    """NVIDIA Nemotron Speech Streaming EN 0.6B int8 — sherpa OnlineRecognizer.

    Cache-aware native streaming with the 160 ms latency profile baked into
    the graph; v1 benches the canonical whole-utterance feed (the graph
    chunks internally — never pre-chunk). Streaming-partial metrics are a
    fast-follow.
    """

    name = "nemotron_streaming"
    backends = {"cpu"}
    supports_streaming = False
    required_sample_rate = 16000

    def __init__(self, opts):
        import sherpa_onnx

        self._r = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=opts["encoder"],
            decoder=opts["decoder"],
            joiner=opts["joiner"],
            tokens=opts["tokens"],
            num_threads=1,
            provider="cpu",
        )

    def load(self, opts):
        pass

    def actual_backend(self) -> str:
        return "cpu"

    def transcribe(self, pcm, sr: int, wav_path: str | None = None):
        import numpy as np

        stream = self._r.create_stream()
        stream.accept_waveform(sr, np.asarray(pcm, dtype="float32"))
        stream.accept_waveform(sr, np.zeros(int(0.2 * sr), dtype=np.float32))
        stream.input_finished()
        while self._r.is_ready(stream):
            self._r.decode_stream(stream)
        result = self._r.get_result(stream)
        return (result if isinstance(result, str) else result.text).strip()


# ---------------------------------------------------------------- TTS engines
class Kokoro:
    name = "kokoro"
    backends = {"cuda", "cpu"}
    supports_streaming = True
    output_sample_rate = 24000
    voices = ["af_heart", "af_bella", "bf_isabella", "am_puck", "bm_fable"]

    def __init__(self, opts):
        import onnxruntime as rt
        from kokoro_onnx import Kokoro

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if opts.get("device", "cuda") == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.sess = rt.InferenceSession(opts["model"], providers=providers)
        self.k = Kokoro.from_session(self.sess, opts["voices"])
        self.voice = opts.get("voice", "af_heart")

    def load(self, opts):
        self._prov = self.sess.get_providers()[0]

    def actual_backend(self) -> str:
        return "cuda" if self._prov == "CUDAExecutionProvider" else "cpu"

    def synthesize(self, text: str):
        audio, sr = self.k.create(text, voice=self.voice, lang="en-us", speed=1.0)
        _sync_cuda()
        return audio, sr

    def synthesize_stream(self, text: str):
        """Drive kokoro's async generator incrementally so TTFA stays real."""
        import asyncio

        import numpy as np

        it = self.k.create_stream(text, voice=self.voice, lang="en-us")
        loop = asyncio.new_event_loop()
        try:
            while True:
                try:
                    samples, sr = loop.run_until_complete(it.__anext__())
                except StopAsyncIteration:
                    break
                _sync_cuda()
                yield np.asarray(samples), sr
        finally:
            loop.run_until_complete(it.aclose())
            loop.close()


class Ptts:
    name = "ptts"
    backends = {"cpu"}
    supports_streaming = True
    output_sample_rate = 24000
    voices = ["default"]

    def __init__(self, opts):
        from ptts import TTS

        self.tts = TTS(
            opts.get("model", "/models/phonon-31m-en"),
            lang=opts.get("lang", "en"),
            quant=opts.get("quant", "q8_0"),
        )

    def load(self, opts):
        pass

    def actual_backend(self) -> str:
        return "cpu"

    def synthesize(self, text: str):
        import numpy as np

        with self.tts.stream(text) as audio:
            chunks = [c for c in audio]
        return np.concatenate(chunks), audio.sample_rate

    def synthesize_stream(self, text: str):
        with self.tts.stream(text) as audio:
            for pcm in audio:
                yield pcm, audio.sample_rate


class Paradee:
    name = "paradee"
    backends = {"cuda", "cpu"}
    supports_streaming = False
    output_sample_rate = 24000
    voices = ["af_heart"]

    def __init__(self, opts):
        from paradee import Paradee

        device = opts.get("device", "cuda")
        if device == "cuda":
            # the paradee package hardcodes providers=["CPUExecutionProvider"]
            # in its constructor — intercept session construction in the
            # class's own module (paradee.tts) so the GPU image actually uses
            # the CUDA EP (CPU stays the fallback)
            import sys

            mod = sys.modules[Paradee.__module__]
            real = mod.ort.InferenceSession

            def _with_cuda(model_path, so=None, providers=None):
                return real(
                    model_path,
                    so,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )

            mod.ort.InferenceSession = _with_cuda
            try:
                self.tts = Paradee(
                    version=opts.get("version", "1.0"),
                    quantized=opts.get("quantized", True),
                )
            finally:
                mod.ort.InferenceSession = real
        else:
            self.tts = Paradee(
                version=opts.get("version", "1.0"),
                quantized=opts.get("quantized", True),
            )

    def load(self, opts):
        # honest provider record (spec 7A): ORT falls back to CPU silently
        try:
            self._prov = self.tts.session.get_providers()[0]
        except Exception:
            self._prov = "unknown"

    def actual_backend(self) -> str:
        return "cuda" if "CUDA" in getattr(self, "_prov", "") else "cpu"

    def synthesize(self, text: str):
        audio = self.tts(text)
        _sync_cuda()
        return audio, 24000

    def synthesize_stream(self, text: str):
        yield self.synthesize(text)


class PiperTts:
    name = "piper"
    backends = {"cuda", "cpu"}
    supports_streaming = False
    output_sample_rate = 22050
    voices = ["en_US-libritts_r-medium"]

    def __init__(self, opts):
        self.model_dir = opts.get("model_dir", "/data/models/piper")
        self.voice = opts.get("voice", "en_US-libritts_r-medium")
        self.use_cuda = opts.get("device", "cuda") == "cuda"

    def load(self, opts):
        from piper import PiperVoice
        import glob

        onnx = sorted(glob.glob(f"{self.model_dir}/**/*.onnx", recursive=True))
        if not onnx:
            raise RuntimeError(f"no piper .onnx under {self.model_dir}")
        self.v = PiperVoice.load(onnx[0], use_cuda=self.use_cuda)

    def actual_backend(self) -> str:
        return "cuda" if self.use_cuda else "cpu"

    def synthesize(self, text: str):
        import tempfile
        import wave

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as w:
            self.v.synthesize_wav(text, w)
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            data = w.readframes(w.getnframes())
        os.unlink(path)
        import numpy as np

        data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        _sync_cuda()
        return data, sr

    def synthesize_stream(self, text: str):
        yield self.synthesize(text)


class PocketTts:
    """Kyutai Pocket TTS 100M — the official pocket-tts package (torch-cpu).

    Native streaming with adaptive chunks exists in the package, but v1
    benches generate_audio per sentence (whole-utterance), comparable with
    the other non-streaming TTS engines. The english bundle is baked into
    /opt/hf-cache; the oneDNN aarch64 JIT is disabled image-wide (Thor CPU
    abort, see the package Dockerfile).
    """

    name = "pocket_tts"
    backends = {"cpu"}
    supports_streaming = False
    output_sample_rate = 24000
    voices = [
        "alba",
        "marius",
        "javert",
        "jean",
        "fantine",
        "cosette",
        "eponine",
        "azelma",
    ]

    def __init__(self, opts):
        self.language = opts.get("language", "english")
        self.voice = opts.get("voice", "alba")
        from pocket_tts import TTSModel

        self._m = TTSModel.load_model(language=self.language)
        self._voice_state = self._m.get_state_for_audio_prompt(self.voice)

    def load(self, opts):
        pass

    def actual_backend(self) -> str:
        return "cpu"

    def synthesize(self, text: str):
        import numpy as np

        audio = self._m.generate_audio(self._voice_state, text)
        _sync_cuda()
        return np.asarray(audio), self._m.sample_rate


STT_ENGINES = {
    c.name: c
    for c in (
        FasterWhisper,
        ParakeetCpp,
        MoonshineStreaming,
        CanaryFlash,
        ParakeetTdt,
        NemotronStreaming,
    )
}
TTS_ENGINES = {c.name: c for c in (Kokoro, Ptts, Paradee, PiperTts, PocketTts)}


def main() -> None:
    job = json.loads(Path(sys.argv[1]).read_text())
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audio").mkdir(exist_ok=True)
    task = job["task"]
    reg = STT_ENGINES if task == "stt" else TTS_ENGINES
    requested = job.get("requested_backend") or job.get("opts", {}).get(
        "device", "cuda"
    )
    # unsupported (engine, backend) pair: honest skip rows, never a silent
    # no-op — decided from the class attribute BEFORE any model is loaded
    if requested not in reg[job["engine"]].backends:
        utt_path = Path(job["out_dir"]) / "utterances.jsonl"
        for i, s in enumerate(job["samples"]):
            jsonl_append(
                utt_path,
                {
                    "engine": job["engine"],
                    "engine_id": job.get("engine_id", job["engine"]),
                    "task": task,
                    "utt_id": s["utt_id"],
                    "run_position": i,
                    "status": "skipped",
                    "reason": (
                        f"backend {requested!r} unsupported by {job['engine']} "
                        f"(supports {sorted(reg[job['engine']].backends)})"
                    ),
                },
            )
        jsonl_append(
            Path(job["out_dir"]) / "run_meta.jsonl",
            {
                "engine": job["engine"],
                "engine_id": job.get("engine_id"),
                "actual_backend": "skipped",
                "requested_backend": requested,
            },
        )
        return

    eng = reg[job["engine"]](job.get("opts", {}))
    eng.load(job.get("opts", {}))
    t_load0 = _now_ns()

    # load time is measured from process start; approximate with first-record
    rec = {
        "engine": eng.name,
        "engine_id": job.get("engine_id", eng.name),
        "load_ns_after_imports": t_load0,
        "actual_backend": eng.actual_backend(),
        "requested_backend": requested,
    }

    utt_path = out_dir / "utterances.jsonl"

    # warmup: the first warmup_n utterances run untimed and are not recorded
    warm = int(job.get("warmup_n", 0) or 0)
    for s in job["samples"][:warm]:
        run_sample(eng, job, s, out_dir, record=False)

    for i, s in enumerate(job["samples"]):
        rec_utt = run_sample(eng, job, s, out_dir, record=True, run_position=i)
        jsonl_append(utt_path, rec_utt)
    jsonl_append(out_dir / "run_meta.jsonl", rec)


def run_sample(
    eng, job: dict, s: dict, out_dir: Path, record: bool = True, run_position: int = 0
) -> dict | None:
    """Run one utterance; returns the record row (None for warmup passes)."""
    task = job["task"]
    out_dir = Path(out_dir)
    rec_utt = {
        "engine": eng.name,
        "engine_id": job.get("engine_id", eng.name),
        "task": task,
        "utt_id": s["utt_id"],
        "run_position": run_position,
        "ref_norm": s.get("text_norm", ""),
        "audio_s": s.get("duration_s"),
        "num_words": s.get("num_words"),
        "num_chars": s.get("num_chars"),
        "language": s.get("language"),
    }
    t0 = _now_ns()
    try:
        if task == "stt":
            import wave

            with wave.open(s["audio_path"], "rb") as w:
                pcm = w.readframes(w.getnframes())
                sr = w.getframerate()
            t1 = _now_ns()
            text = eng.transcribe(pcm, sr, wav_path=s["audio_path"])
            t2 = _now_ns()
            rec_utt.update(hyp_raw=text, proc_ns=t2 - t1, feed_mode="fast", status="ok")
        else:
            import numpy as np  # lazy: parakeet images ship no numpy

            sr_out = None
            if job.get("streaming") and eng.supports_streaming:
                chunks, first, last, n = [], None, None, 0
                for pcm, sr in eng.synthesize_stream(s["text_raw"]):
                    now = _now_ns()
                    first = first or now
                    last = now
                    n += 1
                    sr_out = sr
                    chunks.append(np.asarray(pcm))
                synth_ns = last - t0
                audio = np.concatenate(chunks) if chunks else np.zeros(1)
                rec_utt.update(
                    ttfa_ns=(first - t0) if first else None,
                    synth_ns=synth_ns,
                    chunk_count=n,
                    status="ok",
                )
            else:
                audio, sr_out = eng.synthesize(s["text_raw"])
                audio = np.asarray(audio)
                rec_utt.update(synth_ns=_now_ns() - t0, status="ok")
            rec_utt["sample_rate"] = int(sr_out or eng.output_sample_rate)
            # output duration drives TTS RTF (spec 6.2)
            rec_utt["out_audio_s"] = len(audio) / rec_utt["sample_rate"]
            if record:
                _write_wav(
                    out_dir / "audio" / f"{s['utt_id']}.wav",
                    audio,
                    rec_utt["sample_rate"],
                )
    except (
        Exception
    ) as e:  # record and continue (spec: a crash must not kill the suite)
        rec_utt.update(status="error", error=repr(e)[:500])
    return rec_utt if record else None


def _write_wav(path: Path, audio, sr: int) -> None:
    """stdlib-only WAV writer (no soundfile needed in engine images)."""
    import wave

    import numpy as np

    a = np.asarray(audio)
    if a.dtype != np.int16:
        peak = np.max(np.abs(a)) or 1.0
        a = np.clip(a / (peak if peak > 1.0 else 1.0), -1.0, 1.0)
        a = (a * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1 if a.ndim == 1 else a.shape[1])
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(a.tobytes())


def jsonl_append(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
