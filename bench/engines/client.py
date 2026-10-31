"""Engine client — copied/mounted INTO each engine container (PRD §4).

Reads a job spec (JSON on argv), runs the manifest against one engine, and
appends per-utterance records to utterances.jsonl (+ events.jsonl for streams,
WAVs to audio/ for TTS). Never computes metrics; never times itself — the
client DOES stamp perf_counter_ns around engine calls because it is the only
process that can synchronize the device before stopping the clock (PRD 5.1/4.3).

Usage: python client.py <job.json>
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path


def _now_ns() -> int:
    return time.perf_counter_ns()


def _sync_cuda() -> None:
    """Best-effort device sync before stopping a timer (PRD 4.3)."""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count():
            return  # CT2 waits are synchronous
    except Exception:
        pass
    try:
        import onnxruntime as ort

        # ORT CUDA EP run() is synchronous
        _ = ort.__version__
    except Exception:
        pass


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
        self.lang = opts.get("language", "en")


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
        self.lang = opts.get("lang", "")
        self.vad_model = opts.get("vad_model", "")

    def load(self, opts):
        pass

    def actual_backend(self) -> str:
        return self.device or "cuda"

    def transcribe(self, pcm, sr: int, wav_path: str | None = None):
        import subprocess, tempfile, os

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
        if self.device and self.device != "cuda":
            # images default to GPU (ggml CUDA0); "cuda" means "leave default".
            # Explicit values pass through: cpu, CUDA0, CUDA1, ...
            cmd += ["--device", self.device]
        if self.lang:
            cmd += ["--lang", self.lang]
        if self.stream_latency:
            cmd += ["--stream", "--stream-latency", self.stream_latency]
        elif self.vad_model:
            cmd += ["--vad", "--vad-model", self.vad_model]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        lines = [
            l
            for l in out.stdout.splitlines()
            if l.strip() and not l.startswith("[stream]")
        ]
        return " ".join(lines).strip()

    def stream_feed(self, chunk):
        raise NotImplementedError("v1: streaming via whole-file --stream runs")


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
        import numpy as np

        for samples, sr in self.k.create_stream(text, voice=self.voice, lang="en-us"):
            _sync_cuda()
            yield np.asarray(samples), sr


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
            quant=opts.get("quant", "q8"),
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

        self.tts = Paradee(
            version=opts.get("version", "1.0"), quantized=opts.get("quantized", True)
        )

    def load(self, opts):
        pass

    def actual_backend(self) -> str:
        return (
            "cuda"  # int8 graph on ORT; CPU EP if CUDA unavailable (flagged by probes)
        )

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
        import tempfile, wave

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as w:
            self.v.synthesize_wav(text, w)
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32")
        os.unlink(path)
        _sync_cuda()
        return data, sr

    def synthesize_stream(self, text: str):
        yield self.synthesize(text)


import os  # noqa: E402  (used by PiperTts)

STT_ENGINES = {c.name: c for c in (FasterWhisper, ParakeetCpp)}
TTS_ENGINES = {c.name: c for c in (Kokoro, Ptts, Paradee, PiperTts)}


def main() -> None:
    job = json.loads(Path(sys.argv[1]).read_text())
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audio").mkdir(exist_ok=True)
    task = job["task"]
    reg = STT_ENGINES if task == "stt" else TTS_ENGINES
    eng = reg[job["engine"]](job.get("opts", {}))
    eng.load(job.get("opts", {}))
    t_load0 = _now_ns()

    # load time is measured from process start; approximate with first-record
    rec = {
        "load_ns_after_imports": t_load0,
        "actual_backend": eng.actual_backend(),
        "requested_backend": job.get("opts", {}).get("device", "cuda"),
    }

    utt_path = out_dir / "utterances.jsonl"
    ev_path = out_dir / "events.jsonl"
    for i, s in enumerate(job["samples"]):
        rec_utt = {
            "engine": eng.name,
            "engine_id": job.get("engine_id", eng.name),
            "task": task,
            "utt_id": s["utt_id"],
            "run_position": i,
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
                rec_utt.update(
                    hyp_raw=text, proc_ns=t2 - t1, feed_mode="fast", status="ok"
                )
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
                # output duration drives TTS RTF (PRD 6.2)
                rec_utt["out_audio_s"] = len(audio) / rec_utt["sample_rate"]
                wav_out = out_dir / "audio" / f"{s['utt_id']}.wav"
                import soundfile as sf

                sf.write(wav_out, audio, rec_utt["sample_rate"])
        except (
            Exception
        ) as e:  # record and continue (PRD: a crash must not kill the suite)
            rec_utt.update(status="error", error=repr(e)[:500])
        jsonl_append(utt_path, rec_utt)
    jsonl_append(out_dir / "run_meta.jsonl", rec)


def jsonl_append(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
