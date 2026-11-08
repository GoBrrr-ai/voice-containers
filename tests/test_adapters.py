"""client.py engine adapters via injected stub modules — no engine wheels.

Each adapter's glue (option handling, device/env selection, result shaping)
is the logic that silently breaks; the native libs themselves are exercised
in-image. Plus the chain-probe engine loop with every container faked.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from bench.engines import client


# ------------------------------------------------------------ FasterWhisper
def test_faster_whisper_adapter(monkeypatch):
    created = {}

    class WhisperModel:
        def __init__(self, model, device=None, compute_type=None):
            created.update(model=model, device=device, compute_type=compute_type)

        def transcribe(self, audio, language=None, beam_size=None):
            assert audio.dtype == np.float32
            return iter(
                [SimpleNamespace(text="hi"), SimpleNamespace(text="there")]
            ), None

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)

    eng = client.FasterWhisper({"model": "m", "device": "cpu", "compute_type": "int8"})
    assert created == {"model": "m", "device": "cpu", "compute_type": "int8"}
    eng.load({"device": "cpu", "language": "de"})
    assert eng.actual_backend() == "cpu"
    assert eng.transcribe(b"\0\0" * 8, 16000) == "hi there"


# ------------------------------------------------------------- ParakeetCpp
def test_parakeet_cpp_transcribe_happy(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, capture_output, text, timeout, env=None):
        seen.update(cmd=cmd, env=env)
        return subprocess.CompletedProcess([], 0, stdout="hello\n[stream] x\nworld\n")

    monkeypatch.setattr("subprocess.run", fake_run)
    eng = client.ParakeetCpp(
        {"model": "/m", "device": "cpu", "lang": "en", "stream_latency": "2"}
    )
    assert eng.actual_backend() == "cpu"
    text = eng.transcribe(b"\0\0" * 4, 16000)  # no wav -> temp wav written
    assert text == "hello world"  # [stream] chatter stripped, joined
    assert "--device" in seen["cmd"] and "--stream-latency" in seen["cmd"]
    assert seen["env"]["PARAKEET_DEVICE"] == "cpu"


def test_parakeet_cpp_vad_and_default_device(tmp_path, monkeypatch):
    seen = {}

    def fake_run(cmd, **k):
        seen.update(cmd=cmd)
        return subprocess.CompletedProcess([], 0, stdout="t")

    monkeypatch.setattr("subprocess.run", fake_run)
    eng = client.ParakeetCpp({"model": "/m", "vad_model": "/v"})
    eng.transcribe(b"\0\0" * 4, 16000, wav_path=str(tmp_path / "x.wav"))
    assert "--vad" in seen["cmd"] and "--device" not in seen["cmd"]


def test_parakeet_cpp_failure_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess([], 2, stdout="", stderr="seg"),
    )
    eng = client.ParakeetCpp({"model": "/m"})
    with pytest.raises(RuntimeError, match="parakeet-cli exit 2"):
        eng.transcribe(b"\0\0" * 4, 16000, wav_path="/nonexistent.wav")


# ------------------------------------------------------------------ Kokoro
def test_kokoro_synth_paths(monkeypatch):
    eng = client.Kokoro.__new__(client.Kokoro)
    eng.voice = "af_heart"

    class FakeK:
        def create(self, text, voice=None, lang=None, speed=None):
            return np.zeros(16, dtype=np.float32), 24000

        async def create_stream(self, text, voice=None, lang=None):
            yield np.zeros(8, dtype=np.float32), 24000
            yield np.zeros(8, dtype=np.float32), 24000

    eng.k = FakeK()
    audio, sr = eng.synthesize("hi")
    assert sr == 24000 and len(audio) == 16

    chunks = list(eng.synthesize_stream("hi"))
    assert len(chunks) == 2 and all(s == 24000 for _, s in chunks)
    # the async loop closes cleanly on generator close
    g = eng.synthesize_stream("bye")
    next(g)
    g.close()


# ------------------------------------------------------------- chain probe
def test_probe_engine_loop(tmp_path, monkeypatch, capsys):
    import bench.chain_probe as cp

    manifest = tmp_path / "mf.jsonl"
    manifest.write_text(
        json.dumps({"audio_path": "a.wav", "utt_id": "u1"})
        + "\n"
        + json.dumps({"audio_path": "b.wav", "utt_id": "u2"})
        + "\n"
    )
    monkeypatch.setattr(cp, "MANIFEST", manifest)

    class FakeContainer:
        def __init__(self, *a, **k):
            self.host_port = 59999

        def down(self):
            pass

    monkeypatch.setattr(cp, "Container", FakeContainer)
    monkeypatch.setattr(cp, "_wait_http", lambda *a, **k: True)
    monkeypatch.setattr(cp.time, "sleep", lambda s: None)

    calls = {"warmup": 0, "n": 0}

    def fake_run_once(eng, llm, tts, s, model):
        calls["warmup" if s is None else "n"] += 1
        if s and s["utt_id"] == "u2":
            return None  # a failed pass
        return {"utt_id": (s or {}).get("utt_id", "w")}

    monkeypatch.setattr(cp, "_run_once", fake_run_once)

    # real remote adapter against a dead port: load would fail -> patch build
    import bench.engines.remote as remote_mod

    class FakeRemote:
        def __init__(self, *a, **k):
            pass

        def load(self, opts):
            pass

    monkeypatch.setattr(remote_mod, "OpenAISTTRemote", FakeRemote)

    rows = []
    cp.probe_engine(
        "k1", {"image": "i", "engine": "e", "env": {}}, None, None, "m", rows, 2
    )
    assert calls["warmup"] == 1 and calls["n"] == 2
    assert len(rows) == 1  # u2 returned None -> not recorded
    assert rows[0]["stt_engine"] == "k1" and rows[0]["run_position"] == 0
    assert "1/2 ok" in capsys.readouterr().out

    # server never comes up -> early return, no rows
    monkeypatch.setattr(cp, "_wait_http", lambda *a, **k: False)
    rows2 = []
    cp.probe_engine(
        "k2", {"image": "i", "engine": "e", "env": None}, None, None, "m", rows2, 2
    )
    assert rows2 == [] and "never came up" in capsys.readouterr().out


# ------------------------------------------------------- harness pure helpers
def test_to_float_bytes_and_array():
    out = client._to_float(b"\x00\x00\x00\x40\x00\x80")  # 3 int16 samples, LE
    assert out.dtype == np.float32
    assert out.tolist() == pytest.approx([0.0, 0.5, -1.0], abs=1e-6)
    arr = np.ones(4, dtype=np.float32)
    assert client._to_float(arr) is arr or np.array_equal(client._to_float(arr), arr)


def test_sync_cuda_caches_and_survives_garbage(monkeypatch):
    import ctypes

    from bench.engines import client as c

    # no cudart on the host -> silent no-op, twice (cached-failure path)
    c._sync_cuda()
    c._sync_cuda.__dict__.pop("_rt", None)
    c._sync_cuda()

    # fake cudart: success then nonzero rc
    calls = {"n": 0}

    class FakeRT:
        def cudaDeviceSynchronize(self):
            calls["n"] += 1
            return 0 if calls["n"] == 1 else 3

    fake_cdll = SimpleNamespace(cudaDeviceSynchronize=FakeRT().cudaDeviceSynchronize)
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: "libcudart.so.13")
    monkeypatch.setattr(ctypes, "CDLL", lambda name: fake_cdll)
    c._sync_cuda.__dict__.pop("_rt", None)
    c._sync_cuda()  # rc == 0 -> return
    assert calls["n"] == 1
    c._sync_cuda()  # rc != 0 -> falls through, no raise
    assert calls["n"] == 2
    c._sync_cuda.__dict__.pop("_rt", None)


# --------------------------------------------------- edge-score vector math
def test_edge_vector_score_branches():
    from bench.charts import _edge_vector_score as s

    # lower-is-better (target < worst)
    assert s(0.05, 0.1, 1.0) == 1.0  # at/better than target
    assert s(1.0, 0.1, 1.0) == 0.0  # at worst
    assert 0.0 < s(0.5, 0.1, 1.0) < 1.0  # log-linear between
    # higher-is-better (MOS-like, target > worst)
    assert s(4.5, 4.0, 2.0) == 1.0
    assert s(2.0, 4.0, 2.0) == 0.0
    assert 0.0 < s(3.0, 4.0, 2.0) < 1.0
