"""Engine boundary layers without engine libs.

client.py (the in-container harness) via fake engine adapters through
``run_sample``/``main``; remote.py (the host-side voice-api clients) against
a real loopback HTTP server — the same wire the containers speak.
"""

from __future__ import annotations

import json
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

from bench.engines import client, remote

ENGINES_DIR = Path(__file__).resolve().parents[1] / "bench" / "engines"


# ------------------------------------------------------------------- fakes
class FakeSTT:
    name = "fake-stt"
    backends = {"cuda", "cpu"}
    supports_streaming = False

    def __init__(self, opts=None):
        self.opts = opts or {}
        self.calls = 0

    def load(self, opts):
        self.loaded = True

    def actual_backend(self):
        return "fake:1.0"

    def transcribe(self, pcm, sr, wav_path=None):
        self.calls += 1
        return "hello fake"


class FakeStreamingSTT(FakeSTT):
    supports_streaming = True

    def transcribe_stream(self, pcm, sr, wav_path=None):
        t0 = client._now_ns()
        yield "hel", False, t0 + 1
        yield "hello", False, t0 + 2
        yield "hello fake", True, t0 + 3


class FakeSTTCPU(FakeSTT):
    backends = {"cpu"}


class FakeTTS:
    name = "fake-tts"
    backends = {"cuda"}
    supports_streaming = False
    output_sample_rate = 24000

    def load(self, opts):
        pass

    def actual_backend(self):
        return "fake-tts:1.0"

    def synthesize(self, text):
        return np.zeros(2400, dtype=np.int16), 24000


def _wav(tmp_path: Path, sr=16000, n=1600) -> Path:
    p = tmp_path / "in.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.zeros(n, dtype=np.int16).tobytes())
    return p


def _out(tmp_path: Path, name="o") -> Path:
    d = tmp_path / name
    (d / "audio").mkdir(parents=True, exist_ok=True)
    return d


def _job(tmp_path: Path, **kw):
    d = tmp_path / "out"
    (d / "audio").mkdir(parents=True, exist_ok=True)
    return {
        "engine": "fake-stt",
        "task": "stt",
        "out_dir": str(d),
        "engine_id": "e0",
        "samples": [
            {
                "utt_id": "u1",
                "audio_path": str(_wav(tmp_path)),
                "text_norm": "hello",
                "duration_s": 0.1,
                "num_words": 2,
                "num_chars": 5,
                "language": "en",
            }
        ],
        **kw,
    }


# ------------------------------------------------------------- run_sample paths
def test_run_sample_stt_fast(tmp_path):
    eng, job = FakeSTT(), _job(tmp_path)
    rec = client.run_sample(eng, job, job["samples"][0], _out(tmp_path, "out"))
    assert rec["status"] == "ok" and rec["hyp_raw"] == "hello fake"
    assert rec["feed_mode"] == "fast" and rec["proc_ns"] > 0
    assert rec["engine_id"] == "e0" and rec["ref_norm"] == "hello"


def test_run_sample_stt_streaming(tmp_path):
    job = _job(tmp_path, streaming=True)
    rec = client.run_sample(FakeStreamingSTT(), job, job["samples"][0], _out(tmp_path))
    assert rec["feed_mode"] == "stream"
    assert rec["partials_n"] == 2 and rec["ttfp_ns"] > 0
    # incapable engine + streaming job -> degrade to fast, one final
    job2 = _job(tmp_path, streaming=True)
    rec2 = client.run_sample(FakeSTT(), job2, job2["samples"][0], _out(tmp_path, "o2"))
    assert rec2["feed_mode"] == "fast" and "partials_n" not in rec2


def test_run_sample_error_recorded(tmp_path):
    class Boom(FakeSTT):
        def transcribe(self, pcm, sr, wav_path=None):
            raise RuntimeError("engine on fire")

    job = _job(tmp_path)
    rec = client.run_sample(Boom(), job, job["samples"][0], _out(tmp_path))
    assert rec["status"] == "error" and "engine on fire" in rec["error"]


def test_run_sample_tts_writes_wav(tmp_path):
    job = _job(tmp_path, task="tts")
    job["samples"][0] = {"utt_id": "u1", "text_raw": "hello"}
    rec = client.run_sample(FakeTTS(), job, job["samples"][0], _out(tmp_path))
    assert rec["status"] == "ok" and rec["sample_rate"] == 24000
    assert abs(rec["out_audio_s"] - 0.1) < 0.01
    assert (tmp_path / "o" / "audio" / "u1.wav").exists()


def test_run_sample_tts_streaming(tmp_path):
    class StreamTTS(FakeTTS):
        supports_streaming = True

        def synthesize_stream(self, text):
            yield np.zeros(1200, dtype=np.int16), 24000
            yield np.zeros(1200, dtype=np.int16), 24000

    job = _job(tmp_path, task="tts", streaming=True)
    job["samples"][0] = {"utt_id": "u1", "text_raw": "hello"}
    rec = client.run_sample(StreamTTS(), job, job["samples"][0], _out(tmp_path))
    assert rec["chunk_count"] == 2 and rec["ttfa_ns"] is not None
    assert (tmp_path / "o" / "audio" / "u1.wav").exists()


# --------------------------------------------------------------- main() paths
def test_main_skips_unsupported_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "STT_ENGINES", {"fake-stt": FakeSTT})
    job = _job(tmp_path, requested_backend="cuda")
    # only cpu supported -> every sample becomes an honest skip row
    monkeypatch.setattr(
        client.FakeSTT if False else client, "STT_ENGINES", {"fake-stt": FakeSTTCPU}
    )
    monkeypatch.setattr(sys, "argv", ["client.py", str(_write_job(tmp_path, job))])
    client.main()
    rows = [
        json.loads(x)
        for x in (
            job["out_dir"] / "o" if False else Path(job["out_dir"]) / "utterances.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    assert rows and rows[0]["status"] == "skipped" and "cuda" in rows[0]["reason"]
    meta = (Path(job["out_dir"]) / "run_meta.jsonl").read_text()
    assert "skipped" in meta


def _write_job(tmp_path, job):
    p = tmp_path / "job.json"
    p.write_text(json.dumps(job))
    return p


def test_main_end_to_end_fake_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "STT_ENGINES", {"fake-stt": FakeSTT})
    job = _job(tmp_path, warmup_n=1)
    job["samples"] = job["samples"] * 3
    monkeypatch.setattr(sys, "argv", ["client.py", str(_write_job(tmp_path, job))])
    client.main()
    rows = [
        json.loads(x)
        for x in (Path(job["out_dir"]) / "utterances.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 3 and all(r["status"] == "ok" for r in rows)
    assert {r["run_position"] for r in rows} == {0, 1, 2}
    meta = json.loads((Path(job["out_dir"]) / "run_meta.jsonl").read_text())
    assert meta["actual_backend"] == "fake:1.0"
    assert "load_ns_after_imports" in meta


# ------------------------------------------------------------------ pure bits
def test_sync_cuda_no_cuda_is_noop():
    client._sync_cuda()  # no cudart on CI -> swallowed; must not raise


def test_jsonl_append_utf8(tmp_path):
    p = tmp_path / "a.jsonl"
    client.jsonl_append(p, {"x": "é"})
    client.jsonl_append(p, {"x": 2})
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    assert rows[1]["x"] == 2 and "é" == rows[0]["x"]


# ------------------------------------------------------------ remote over HTTP
class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, status=200, headers=None):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json({"status": "ok"})
        elif self.path == "/v1/models":
            self._json({"object": "list", "data": [{"id": "m1", "object": "model"}]})
        elif self.path == "/v1/voices":
            self._json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "v1",
                            "engine": "eng1",
                            "sample_rate": 24000,
                            "streaming": True,
                        }
                    ],
                }
            )
        else:
            self._json({"detail": "nope"}, 404)

    sse = True  # class attr; the plain fixture subclasses it off

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body_in = self.rfile.read(n)
        if "transcriptions" in self.path:
            if self.sse and b'name="stream"' in body_in:
                body = (
                    'event: partial\ndata: {"text": "hel"}\n\n'
                    'event: final\ndata: {"text": "hello"}\n\n'
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
            else:
                body = b'{"text": "hello"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
        else:  # speech: real wav container, odd tail exercises the carry
            pcm = bytes(range(256)) * 3 + b"\x01"
            body = remote._wav_bytes(pcm, 24000)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("X-Sample-Rate", "24000")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture()
# noqa needed? no — fixture below
def server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def test_remote_stt_roundtrip(server, tmp_path):
    stt = remote.OpenAISTTRemote("e", "127.0.0.1", server.server_address[1], {})
    stt.load({})
    assert stt._model == "m1" and stt.supports_streaming is True
    assert stt.actual_backend() == "remote:openai"
    pcm = np.zeros(1600, dtype=np.int16)
    assert stt.transcribe(pcm, 16000) == "hello"
    wav = _wav(tmp_path, n=320)
    assert stt.transcribe(None, 16000, wav_path=str(wav)) == "hello"


def test_remote_stt_language_opt(server):
    stt = remote.OpenAISTTRemote(
        "e", "127.0.0.1", server.server_address[1], {"language": "de"}
    )
    assert stt.lang == "de"
    stt2 = remote.OpenAISTTRemote("e", "127.0.0.1", 1, {})
    assert stt2.lang is None


def test_remote_stt_stream_sse_and_degrade(server, monkeypatch):
    url = f"http://127.0.0.1:{server.server_address[1]}"
    stt = remote.OpenAISTTRemote(
        "e", "127.0.0.1", server.server_address[1], {"model": "m1"}
    )
    stt._model = "m1"
    events = list(stt.transcribe_stream(np.zeros(16, dtype=np.int16), 16000))
    texts = [t for t, final, _ in events]
    finals = [f for _, f, _ in events]
    assert texts[-1] == "hello" and finals[-1] is True
    assert any(not f for f in finals)  # at least one partial
    assert url


def test_remote_tts_roundtrip(server):
    tts = remote.OpenAITTSRemote(
        "e", "127.0.0.1", server.server_address[1], {"voice": "", "speed": 1.1}
    )
    tts.load({})
    assert tts._model == "eng1" and tts.voice == "v1"
    assert tts.output_sample_rate == 24000
    kw = tts._speech_kwargs("hi", "wav")
    assert kw["voice"] == "v1" and kw["speed"] == 1.1
    pcm, sr = tts.synthesize("hi")
    assert sr == 24000 and len(pcm) > 100
    chunks = list(tts.synthesize_stream("hi"))
    assert chunks and chunks[0][1] == 24000
    total = sum(len(c) for c, _ in chunks)
    assert total % 2 == 0  # carry logic keeps frames whole


def test_wav_roundtrip(tmp_path):
    pcm = np.array([0, 100, -100, 300], dtype=np.int16)
    blob = remote._wav_bytes(pcm.tobytes(), 16000)
    data, sr = remote._wav_read(blob)
    assert sr == 16000 and list(data) == [0, 100, -100, 300]


def test_iter_sse():
    lines = [
        b"event: partial\n",
        b"data: {'x': 1}\n",
        b"\n",
        b"noise\n",
    ]
    events = list(remote._iter_sse(iter(lines)))
    assert ("partial", "{'x': 1}") in events


def test_build_dispatch():
    serve = {"transport": "openai", "host_port": 1}
    assert isinstance(remote.build("stt", serve, "e", {}), remote.OpenAISTTRemote)
    assert isinstance(remote.build("tts", serve, "e", {}), remote.OpenAITTSRemote)
    serve["transport"] = "wyoming"
    assert isinstance(remote.build("stt", serve, "e", {}), remote.WyomingSTTRemote)
    serve["transport"] = "carrier-pigeon"
    with pytest.raises(ValueError):
        remote.build("stt", serve, "e", {})


def test_wyoming_light_paths():
    w = remote.WyomingTTSRemote("e", "127.0.0.1", 10200, {})
    w.load({})
    assert w.actual_backend() == "remote:wyoming"
    assert w.output_sample_rate == 22050
    ws = remote.WyomingSTTRemote("e", "127.0.0.1", 10300, {})
    ws.load({})
    assert ws.actual_backend() == "remote:wyoming"


# ------------------------------------------------------------ wyoming wire
def _wyoming_tts_server():
    """Real wyoming-wire server: Synthesize -> AudioStart/Chunk/Stop."""
    import asyncio

    from wyoming.audio import AudioChunk, AudioStart, AudioStop
    from wyoming.event import async_read_event, async_write_event
    from wyoming.tts import Synthesize

    pcm = (np.ones(2205, dtype=np.int16) * 100).tobytes()

    async def handle(reader, writer):
        try:
            while True:
                ev = await async_read_event(reader)
                if ev is None:
                    return
                if Synthesize.is_type(ev.type):
                    await async_write_event(
                        AudioStart(rate=22050, width=2, channels=1).event(), writer
                    )
                    await async_write_event(
                        AudioChunk(audio=pcm, rate=22050, width=2, channels=1).event(),
                        writer,
                    )
                    await async_write_event(AudioStop().event(), writer)
        except Exception:
            return

    async def main():
        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        return server

    loop = asyncio.new_event_loop()
    srv = loop.run_until_complete(main())
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return srv, loop


def test_wyoming_tts_roundtrip():
    srv, loop = _wyoming_tts_server()
    try:
        port = srv.sockets[0].getsockname()[1]
        tts = remote.WyomingTTSRemote("e", "127.0.0.1", port, {})
        chunks = list(tts.synthesize_stream("hello"))
        assert len(chunks) == 1
        pcm, sr = chunks[0]
        assert sr == 22050 and len(pcm) == 2205 and tts.output_sample_rate == 22050
        # synthesize() folds the stream
        pcm2, sr2 = tts.synthesize("hello")
        assert sr2 == 22050 and len(pcm2) == 2205
    finally:
        loop.call_soon_threadsafe(loop.stop)
        srv.close()


def _wyoming_stt_server(text="hello wy"):
    import asyncio

    from wyoming.asr import Transcript
    from wyoming.audio import AudioStop
    from wyoming.event import async_read_event, async_write_event

    async def handle(reader, writer):
        try:
            while True:
                ev = await async_read_event(reader)
                if ev is None:
                    return
                if AudioStop.is_type(ev.type):
                    await async_write_event(Transcript(text=text).event(), writer)
        except Exception:
            return

    async def main():
        return await asyncio.start_server(handle, "127.0.0.1", 0)

    loop = asyncio.new_event_loop()
    srv = loop.run_until_complete(main())
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return srv, loop


def test_wyoming_stt_roundtrip(tmp_path):
    srv, loop = _wyoming_stt_server()
    try:
        port = srv.sockets[0].getsockname()[1]
        stt = remote.WyomingSTTRemote("e", "127.0.0.1", port, {})
        assert stt.actual_backend() == "remote:wyoming"
        wav = _wav(tmp_path)
        assert stt.transcribe(None, 16000, wav_path=str(wav)) == "hello wy"
        assert stt.transcribe(np.zeros(16, dtype=np.int16), 16000) == "hello wy"
    finally:
        loop.call_soon_threadsafe(loop.stop)
        srv.close()
