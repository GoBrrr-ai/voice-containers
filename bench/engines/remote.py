"""Remote engine adapters — voice-api v1 (docs/voice-api.md) + wyoming TCP.

Serve mode (runner branch in bench/runner.py): the engine image runs its own
voice-server (`docker run -d`, the image CMD serves), and client.py drives it
from the HOST over loopback. These classes implement the exact in-process
adapter contract from client.py (name/backends/actual_backend + transcribe /
synthesize / synthesize_stream), so run_sample is untouched.

Timing (spec 5.1): proc_ns / synth_ns include loopback transport overhead —
identical for every engine in the roster, which is what keeps RTF comparable
(and matches how a real deployment calls these servers). No host-side
_sync_cuda: the compute happens in the server container, and the HTTP/TCP
round trip is the completion point.

Deps (bench/.venv only — images stay stdlib): openai, wyoming==1.10.2, numpy.

Wyoming wire note: AsyncTcpClient.read_event() returns GENERIC Events —
isinstance() never matches. The pattern is X.is_type(ev.type) +
X.from_event(ev).
"""

from __future__ import annotations

import asyncio
import io
import time
import wave
from pathlib import Path

import numpy as np


def _now_ns() -> int:
    return time.perf_counter_ns()


def _wav_bytes(pcm: bytes, sr: int) -> bytes:
    """Wrap raw int16 PCM in a minimal WAV container (44-byte header)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm)
    return buf.getvalue()


def _wav_read(data: bytes):
    """Parse a WAV blob -> (int16 np array, sample_rate)."""
    with wave.open(io.BytesIO(data), "rb") as w:
        sr = w.getframerate()
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype=np.int16), sr


# ------------------------------------------------------------------- OpenAI
class OpenAISTTRemote:
    """voice-api v1 STT over the OpenAI python SDK (multipart upload)."""

    backends = {"cuda", "cpu"}
    supports_streaming = False
    required_sample_rate = 16000

    def __init__(self, engine: str, host: str, port: int, opts: dict):
        self.name = engine
        self.base = f"http://{host}:{port}"
        self.lang = (opts.get("lang") or opts.get("language") or "")[:2] or None
        self._model = opts.get("model", "")
        self._client = None

    def load(self, opts: dict) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=self.base + "/v1", api_key="local")
        if not self._model:
            # /v1/models is the documented alias of /v1/voices (single model)
            self._model = self._client.models.list().data[0].id

    def actual_backend(self) -> str:
        return "remote:openai"

    def transcribe(self, pcm, sr: int, wav_path: str | None = None) -> str:
        if wav_path:
            fh = open(wav_path, "rb")
            name = Path(wav_path).name
        else:
            fh = io.BytesIO(_wav_bytes(pcm, sr))
            name = "audio.wav"
        try:
            kwargs = {"model": self._model, "file": (name, fh)}
            if self.lang:
                kwargs["language"] = self.lang
            r = self._client.audio.transcriptions.create(**kwargs)
            return (r.text if hasattr(r, "text") else str(r)).strip()
        finally:
            fh.close()


class OpenAITTSRemote:
    """voice-api v1 TTS over the OpenAI python SDK.

    synthesize() -> wav blob; synthesize_stream() -> `pcm` response read
    chunk-by-chunk (Transfer-Encoding: chunked) — honest TTFA for the
    streaming engines, derived (= completion) for single-chunk ones, matching
    the in-process adapters' semantics.
    """

    backends = {"cuda", "cpu"}
    supports_streaming = True
    required_sample_rate = 24000

    def __init__(self, engine: str, host: str, port: int, opts: dict):
        self.name = engine
        self.base = f"http://{host}:{port}"
        self.voice = opts.get("voice", "")
        self.speed = opts.get("speed")
        self._model = opts.get("model", "")
        self._client = None

    def load(self, opts: dict) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=self.base + "/v1", api_key="local")
        voices = self._client.get("/v1/voices")["data"]
        if not self._model:
            # voice entries carry the engine id == the `model` the server
            # validates against (docs/voice-api.md)
            self._model = voices[0].get("engine") or voices[0]["id"]
        self.output_sample_rate = int(voices[0].get("sample_rate") or 24000)

    def actual_backend(self) -> str:
        return "remote:openai"

    def _speech_kwargs(self, text: str, response_format: str) -> dict:
        kw = {
            "model": self._model,
            "input": text,
            "response_format": response_format,
        }
        if self.voice:
            kw["voice"] = self.voice
        if self.speed:
            kw["speed"] = self.speed
        return kw

    def synthesize(self, text: str):
        with self._client.audio.speech.with_streaming_response.create(
            **self._speech_kwargs(text, "wav")
        ) as r:
            return _wav_read(r.content)

    def synthesize_stream(self, text: str):
        with self._client.audio.speech.with_streaming_response.create(
            **self._speech_kwargs(text, "pcm")
        ) as r:
            sr = int(r.response.headers.get("X-Sample-Rate") or self.output_sample_rate)
            # chunked read may split mid-sample — keep a 1-byte carry
            buf = b""
            for chunk in r.iter_bytes(8192):
                buf += chunk
                usable = len(buf) // 2 * 2
                if usable:
                    yield np.frombuffer(buf[:usable], dtype=np.int16), sr
                    buf = buf[usable:]
            if buf:
                yield np.frombuffer(buf[: len(buf) // 2 * 2], dtype=np.int16), sr


# ------------------------------------------------------------------ Wyoming
class WyomingTTSRemote:
    """Wyoming TTS (AudioStart/AudioChunk/AudioStop) — HA-native pair."""

    backends = {"cuda", "cpu"}
    supports_streaming = True

    def __init__(self, engine: str, host: str, port: int, opts: dict):
        self.name = engine
        self.host, self.port = host, int(port)
        self.output_sample_rate = 22050

    def load(self, opts: dict) -> None:
        pass

    def actual_backend(self) -> str:
        return "remote:wyoming"

    def synthesize_stream(self, text: str):
        from wyoming.audio import AudioChunk, AudioStart, AudioStop
        from wyoming.client import AsyncTcpClient
        from wyoming.tts import Synthesize

        async def _gen():
            async with AsyncTcpClient(self.host, self.port) as client:
                await client.write_event(Synthesize(text=text).event())
                while True:
                    ev = await client.read_event()
                    if ev is None:
                        return
                    t = ev.type
                    if AudioStart.is_type(t):
                        ev = AudioStart.from_event(ev)
                        self.output_sample_rate = ev.rate
                    elif AudioChunk.is_type(t):
                        ev = AudioChunk.from_event(ev)
                        yield np.frombuffer(ev.audio, dtype=np.int16), ev.rate
                    elif AudioStop.is_type(t):
                        return

        loop = asyncio.new_event_loop()
        try:
            it = _gen().__aiter__()
            while True:
                try:
                    yield loop.run_until_complete(it.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

    def synthesize(self, text: str):
        chunks, sr = [], None
        for pcm, r in self.synthesize_stream(text):
            chunks.append(pcm)
            sr = r
        return np.concatenate(chunks), sr


class WyomingSTTRemote:
    """Wyoming ASR: AudioStart(rate) -> AudioChunk* -> AudioStop -> Transcript."""

    backends = {"cuda", "cpu"}
    supports_streaming = False
    required_sample_rate = 16000

    def __init__(self, engine: str, host: str, port: int, opts: dict):
        self.name = engine
        self.host, self.port = host, int(port)

    def load(self, opts: dict) -> None:
        pass

    def actual_backend(self) -> str:
        return "remote:wyoming"

    def transcribe(self, pcm, sr: int, wav_path: str | None = None) -> str:
        from wyoming.asr import Transcript, Transcribe
        from wyoming.audio import AudioChunk, AudioStart, AudioStop
        from wyoming.client import AsyncTcpClient

        if wav_path:
            with wave.open(wav_path, "rb") as w:
                pcm = w.readframes(w.getnframes())
                sr = w.getframerate()
        elif isinstance(pcm, np.ndarray):
            pcm = pcm.astype(np.int16).tobytes()

        async def _main() -> str:
            async with AsyncTcpClient(self.host, self.port) as client:
                await client.write_event(Transcribe().event())
                await client.write_event(
                    AudioStart(rate=int(sr), width=2, channels=1).event()
                )
                view = memoryview(pcm)
                step = 4096  # bytes; server buffers, size is arbitrary
                for off in range(0, len(view), step):
                    await client.write_event(
                        AudioChunk(
                            audio=bytes(view[off : off + step]),
                            rate=int(sr),
                            width=2,
                            channels=1,
                        ).event()
                    )
                await client.write_event(AudioStop().event())
                while True:
                    ev = await client.read_event()
                    if ev is None:
                        return ""
                    if Transcript.is_type(ev.type):
                        return Transcript.from_event(ev).text

        return asyncio.run(_main()).strip()


# ------------------------------------------------------------------ factory
def build(task: str, serve: dict, engine: str, opts: dict):
    """Construct the serve-mode adapter for one job (client.py main())."""
    transport = serve["transport"]
    host = serve.get("host", "127.0.0.1")
    port = int(serve["host_port"])
    if transport == "openai":
        cls = OpenAISTTRemote if task == "stt" else OpenAITTSRemote
    elif transport == "wyoming":
        cls = WyomingSTTRemote if task == "stt" else WyomingTTSRemote
    else:
        raise ValueError(f"unknown serve transport {transport!r}")
    return cls(engine, host, port, opts)
