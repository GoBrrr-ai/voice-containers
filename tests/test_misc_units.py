"""Small pure units: env record, cachectl, probe parsers, chain-probe helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------- env
def test_collect_parses_compute_cap(monkeypatch):
    from bench import env

    monkeypatch.setattr(
        env,
        "_run",
        lambda cmd: "Orin, 8.7, 580.00" if "nvidia-smi" in cmd else "x",
    )
    env_rec = env.collect(git_commit="abc1234")
    assert env_rec["compute_cap"] == "8.7" and env_rec["compute_cap_ok"] is True
    assert env_rec["git_commit"] == "abc1234"


def test_collect_no_gpu(monkeypatch):
    from bench import env

    monkeypatch.setattr(env, "_run", lambda cmd: "")
    env_rec = env.collect()
    assert env_rec["compute_cap"] == "" and env_rec["compute_cap_ok"] is False


def test_selfcheck_writes_json(tmp_path: Path, monkeypatch):
    import bench.env as env

    monkeypatch.setattr(env, "collect", lambda git_commit="": {"compute_cap": "8.7"})
    monkeypatch.setattr(env, "_run", lambda cmd: "ok")
    report = env.selfcheck(tmp_path)
    assert (
        json.loads((tmp_path / "selfcheck.json").read_text())["docker_gpu_runtime"]
        == "ok"
    )
    assert report["docker_gpu_runtime"] == "ok"


# ------------------------------------------------------------------ cachectl
def test_user_and_rule(monkeypatch):
    from bench import cachectl

    monkeypatch.setenv("SUDO_USER", "benchy")
    assert cachectl._user() == "benchy"
    assert "benchy ALL=(ALL) NOPASSWD:" in cachectl._rule()
    monkeypatch.delenv("SUDO_USER")
    assert cachectl._user()


def test_drop_caches_granted(monkeypatch):
    from bench import cachectl

    def ok_run(*a, **k):
        return subprocess.CompletedProcess(
            [], 0, stdout="... sysctl /usr/sbin/sysctl -w vm.drop_caches=3 ..."
        )

    monkeypatch.setattr(cachectl.subprocess, "run", ok_run)
    assert cachectl.drop_caches_granted() is True

    monkeypatch.setattr(
        cachectl.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 1, stdout=""),
    )
    assert cachectl.drop_caches_granted() is False

    def boom(*a, **k):
        raise OSError("no sudo here")

    monkeypatch.setattr(cachectl.subprocess, "run", boom)
    assert cachectl.drop_caches_granted() is False


def test_drop_caches_call(monkeypatch):
    from bench import cachectl

    seen = {}
    monkeypatch.setattr(
        cachectl.subprocess,
        "call",
        lambda cmd: (seen.__setitem__("cmd", cmd), 0)[1],
    )
    assert cachectl.drop_caches() == 0
    assert seen["cmd"][:3] == ["sudo", "-n", "/usr/sbin/sysctl"]


def test_grant_idempotent_and_abort(capsys, monkeypatch):
    from bench import cachectl

    monkeypatch.setattr(cachectl, "drop_caches_granted", lambda: True)
    assert cachectl.grant_drop_caches(assume_yes=False) == 0

    monkeypatch.setattr(cachectl, "drop_caches_granted", lambda: False)
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    assert cachectl.grant_drop_caches(assume_yes=False) == 1  # aborted


def test_grant_visudo_rejected(monkeypatch):
    from bench import cachectl

    monkeypatch.setattr(cachectl, "drop_caches_granted", lambda: True)
    monkeypatch.setattr(cachectl, "drop_caches_granted", lambda: False)

    def fake_run(cmd, **k):
        return subprocess.CompletedProcess([], 1, stdout="", stderr="bad rule")

    monkeypatch.setattr(cachectl.subprocess, "run", fake_run)
    assert cachectl.grant_drop_caches(assume_yes=True) == 1


def test_prepare_bench_paths(capsys, monkeypatch):
    from bench import cachectl

    monkeypatch.setattr(cachectl, "drop_caches_granted", lambda: False)
    cachectl.prepare_bench()  # quiet=False -> hint printed
    assert "grant missing" in capsys.readouterr().err
    cachectl.prepare_bench(quiet=True)

    monkeypatch.setattr(cachectl, "drop_caches_granted", lambda: True)
    monkeypatch.setattr(cachectl, "drop_caches", lambda: 0)
    cachectl.prepare_bench()
    assert "page cache cleared" in capsys.readouterr().err

    monkeypatch.setattr(cachectl, "drop_caches", lambda: 3)
    cachectl.prepare_bench()
    assert "failed" in capsys.readouterr().err


# -------------------------------------------------------------------- probes
def test_parse_mem_mib_units():
    from bench.probes import _parse_mem_mib

    assert _parse_mem_mib("512MiB") == 512.0
    assert _parse_mem_mib("1.5GiB") == 1536.0
    assert _parse_mem_mib("100kB") == pytest.approx(100 / 1024)
    assert _parse_mem_mib("junk") == 0.0


def test_parse_tegrastats(tmp_path: Path):
    from bench.probes import parse_tegrastats

    log = tmp_path / "tegrastats.log"
    log.write_text(
        "11-07-2026 12:00:00 RAM 2000/8000MB (lft) GR3D 45% VDD_GPU 9000mW junk\n"
        "11-07-2026 12:00:01 RAM 2100/8000MB GR3D 55% VDD_GPU 10000mW\n"
        "not a ts line\n"
    )
    rows = parse_tegrastats(log)
    assert len(rows) == 2
    assert (
        rows[0]["ram_mib"] == 2000
    )  # raw regex value; MiB conversion is aggregate-side
    assert rows[0]["gpu_mw"] == 9000 and rows[0]["gpu_util_pct"] == 45
    assert rows[1]["gpu_util_pct"] == 55
    assert parse_tegrastats(tmp_path / "nope.log") == []


def test_parse_tegrastats_gz(tmp_path: Path):
    import gzip

    from bench.probes import parse_tegrastats

    p = tmp_path / "tegrastats.log"
    with gzip.open(f"{p}.gz", "wt") as f:
        f.write("11-07-2026 12:00:00 RAM 2000/8000MB GR3D 45%\n")
    rows = parse_tegrastats(p)
    assert len(rows) == 1 and rows[0]["gpu_util_pct"] == 45


def test_window_and_docker_stats():
    from bench.probes import window_aggregate

    rows = [
        {"epoch": 100.0, "gpu_mw": 1000, "gpu_util_pct": 10, "ram_mib": 100},
        {"epoch": 200.0, "gpu_mw": 3000, "gpu_util_pct": 30, "ram_mib": 300},
    ]
    win = window_aggregate(iter(rows), 0, 1000)
    assert win["gpu_power_w_mean"] == pytest.approx(2.0)
    assert win["n_samples"] == 2
    assert window_aggregate([], 0, 10) is None or window_aggregate([], 0, 10) == {}


# -------------------------------------------------------------- chain probe
def test_stream_first_sentence():
    from bench.chain_probe import _stream_first_sentence

    def chunks(words):
        return [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=w))])
            for w in words
        ]

    ttft, sent_t, sent, done = _stream_first_sentence(
        chunks(["Hel", "lo", " wor", "ld.", " more"])
    )
    assert sent == "Hello world." and done is False
    assert ttft <= sent_t

    ttft2, _, sent2, done2 = _stream_first_sentence(chunks(["no", "punct"]))
    assert done2 is True and sent2 == "nopunct"

    # empty deltas skipped; empty stream -> done, empty text
    _, _, sent3, done3 = _stream_first_sentence(
        [
            SimpleNamespace(choices=[]),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]
            ),
        ]
    )
    assert sent3 == "" and done3 is True


def test_run_once_full_chain(tmp_path, monkeypatch):
    import bench.chain_probe as cp

    manifest = tmp_path / "mf.jsonl"
    manifest.write_text(json.dumps({"audio_path": "a.wav", "utt_id": "u1"}) + "\n")
    (tmp_path / "a.wav").write_bytes(b"RIFF....")  # placeholder; wave-read patched

    monkeypatch.setattr(cp, "MANIFEST", manifest)
    monkeypatch.setattr(cp, "CACHE_DIR", tmp_path)

    class FakeWave:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def getnframes(self):
            return 2

        def getframerate(self):
            return 16000

        def readframes(self, n):
            return b"\x00\x00\x00\x00"

    monkeypatch.setattr("wave.open", lambda *a, **k: FakeWave())

    class FakeStream:
        def __iter__(self):
            return iter(
                [
                    SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi."))]
                    )
                ]
            )

    eng = SimpleNamespace(transcribe=lambda pcm, sr, wav_path=None: "hello there. ok")
    llm = SimpleNamespace(create=lambda **kw: FakeStream())
    tts = SimpleNamespace(synthesize_stream=lambda text: iter([(b"\x00\x00", 24000)]))
    rec = cp._run_once(eng, llm, tts, None, "m")
    assert rec["utt_id"] == "warmup"
    assert rec["first_sentence"] == "Hi."
    for k in ("stt_ns", "llm_ttft_ns", "llm_1sent_ns", "tts_ttfa_ns", "chain_ns"):
        assert rec[k] >= 0

    # empty hypothesis -> None
    silent = SimpleNamespace(transcribe=lambda *a, **k: "  ")
    assert cp._run_once(silent, llm, tts, None, "m") is None
    # tts producing no chunks -> None
    dry = SimpleNamespace(synthesize_stream=lambda text: iter([]))
    assert cp._run_once(eng, llm, dry, None, "m") is None


def test_container_down_and_wait_http(monkeypatch):
    import bench.chain_probe as cp

    ok = subprocess.CompletedProcess([], 0, stdout="", stderr="")
    monkeypatch.setattr(cp, "_sh", lambda *a: ok)
    c = cp.Container("n", "img", 8080)
    assert c.host_port > 0
    ran = {}
    monkeypatch.setattr(cp, "_sh", lambda *a: ran.setdefault("cmd", a))
    c.down()
    assert "rm" in ran["cmd"]
