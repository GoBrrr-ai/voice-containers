"""runner.py pure/patched units + one fake-docker end-to-end of run_config.

No GPU, no docker daemon: the docker/tegrastats/cachectl/env collaborators
are stubbed; everything downstream of "container ran" is real (summary,
history append, raw publish) — that's the spec 9.1 protocol regression.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from bench import runner


# ---------------------------------------------------------------- validate_config
def _cfg(**kw):
    base = {
        "task": "stt",
        "engines": [{"name": "parakeet-tdt-cpp-stt", "image": "narandill/x:0.1"}],
    }
    base.update(kw)
    return base


def test_validate_config_ok():
    runner.validate_config(_cfg())  # silent on success
    runner.validate_config(_cfg(repeats=1, probes={"tegrastats": "auto"}))


def test_validate_config_rejects():
    bad = [
        _cfg(unknown_key=1),
        _cfg(normalizer="per_engine"),
        _cfg(probes={"tegrastats": "maybe"}),
        _cfg(repeats=2),
        _cfg(task="asr"),
        _cfg(engines=[]),
        _cfg(engines=[{"name": "x"}]),
        _cfg(
            engines=[
                {
                    "name": "x",
                    "image": "i",
                    "serve": {"transport": "grpc", "port": 1},
                }
            ]
        ),
        _cfg(engines=[{"name": "x", "image": "i", "serve": "openai"}]),
    ]
    for cfg in bad:
        with pytest.raises(SystemExit):
            runner.validate_config(cfg)


# ---------------------------------------------------------- container path mapping
def test_container_audio_path_relative_contract(tmp_path):
    assert (
        runner._container_audio_path("a/b.wav", tmp_path) == "/data/bench-cache/a/b.wav"
    )
    # relative-only contract: absolute paths are rejected with a regenerate hint
    with pytest.raises(ValueError, match="cache-relative"):
        runner._container_audio_path("/elsewhere/x.wav", tmp_path)


# ------------------------------------------------------------------ engine summary
def _row(**kw):
    r = {"status": "ok", "engine_id": "e", "proc_ns": 2.0e8, "audio_s": 1.0}
    r.update(kw)
    return r


def test_engine_summary_stt():
    from bench.runner import _engine_summary

    rows = [
        _row(
            ref_norm="hello world", hyp_raw="hello world", ttfp_ns=1.0e8, partials_n=2
        ),
        _row(ref_norm="a b", hyp_raw="a c", ttfp_ns=1.1e8, partials_n=3),
        _row(status="skipped", reason="no audio"),
    ]
    s = _engine_summary(rows, "stt")
    assert s["n_ok"] == 2  # corpus n is wer's ref count; row count via n_ok
    assert s["n_skipped"] == 1 and s["skip_reason"] == "no audio"
    for k in ("wer", "cer", "rtf_p90", "offline_latency_s_p90"):
        assert k in s, k
    assert "hyp_norm" in rows[0]
    # realtime branch engaged: every ok row streamed
    assert "ttfp_s_p90" in s and "partials_n_p90" in s


def test_engine_summary_tts_and_mixed_streaming():
    from bench.runner import _engine_summary

    rows = [
        _row(synth_ns=5.0e8, out_audio_s=2.0),
        _row(synth_ns=7.0e8, out_audio_s=2.0, ttfa_ns=1.0e8),
    ]
    s = _engine_summary([dict(r) for r in rows], "tts")
    assert "synth_latency_s_p90" in s
    assert "ttfa_s_p90" not in s  # one row lacks ttfa -> branch skipped
    assert "rtf_p90" in s
    rows[0]["ttfa_ns"] = 9.0e8
    s2 = _engine_summary([dict(r) for r in rows], "tts")
    assert "ttfa_s_p90" in s2


def test_write_summary_json_csv(compare_tmp: Path):
    run_dir = compare_tmp
    rows = [
        _row(engine_id="a", ref_norm="x", hyp_raw="x"),
        _row(engine_id="b", status="skipped", reason="r"),
    ]
    with open(run_dir / "utterances.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    s = runner.write_summary(run_dir, "stt")
    assert set(s["per_engine"]) == {"a", "b"}
    assert s["per_engine"]["a"]["n_ok"] == 1
    assert "wer" in s["per_engine"]["a"]
    csv_text = (run_dir / "summary.csv").read_text()
    assert "engine_id,status" in csv_text or "engine_id" in csv_text


def test_compare_skips_missing_and_renders_wer(compare_tmp: Path):
    other = compare_tmp.parent / "other"
    other.mkdir()
    (other / "summary.json").write_text(
        json.dumps({"wer": 0.031, "n_ok": 5, "task": "stt"})
    )
    out = runner.compare(
        [str(compare_tmp), str(other), str(compare_tmp / "nope")],
        str(compare_tmp / "cmp.md"),
    )
    assert "0.031" in Path(out).read_text()


# ------------------------------------------------------------------- socket probes
class _Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def log_message(self, *a):
        pass


def test_wait_served_http():
    srv = HTTPServer(("127.0.0.1", 0), _Health)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert runner._wait_served("openai", srv.server_address[1], timeout_s=3)
        assert not runner._wait_served("openai", runner._free_port(), timeout_s=1)
    finally:
        srv.shutdown()
        srv.server_close()


def test_wait_for_free_gpu_instant_clean(monkeypatch, capsys):
    monkeypatch.setattr(runner, "_gpu_used_mib", lambda: 10)
    monkeypatch.setattr(runner, "_gpu_busy_pct", lambda: 0)
    runner._wait_for_free_gpu(100, 0.01)
    assert "GPU clean" in capsys.readouterr().out


def test_wait_for_free_gpu_defers(monkeypatch):
    monkeypatch.setattr(runner, "_gpu_used_mib", lambda: 5000)
    monkeypatch.setattr(runner, "_gpu_busy_pct", lambda: 80)
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit):
        runner._wait_for_free_gpu(100, 0.01, confirm=2, poll_s=0)


# ------------------------------------------------- run_config end-to-end, fake docker
@pytest.fixture()
def compare_tmp(tmp_path: Path) -> Path:
    d = tmp_path / "run"
    d.mkdir()
    return d


def test_run_config_fake_docker(tmp_path: Path, monkeypatch):
    root = tmp_path / "repo"
    (root / "bench" / "configs").mkdir(parents=True)
    manifest = root / "mf.jsonl"
    manifest.write_text('{"audio_path": "a1.wav"}\n{"audio_path": "a2.wav"}\n')
    cfg = root / "bench" / "configs" / "smoke.yaml"
    cfg.write_text(
        json.dumps(
            {
                "task": "stt",
                "run_name": "baseline_smoke",
                "device_label": "ci",
                "dataset": {"manifest": "mf.jsonl"},
                "engines": [
                    {"name": "e1", "image": "img:1"},
                    {"name": "e2", "image": "img:2", "id": "e2"},
                ],
            }
        )
    )

    ran = []

    class FakeTegra:
        def __init__(self, out_path, interval_ms=1000):
            ran.append(("teg", str(out_path)))

        def start(self):
            ran.append(("start", None))

        def stop(self):
            ran.append(("stop", None))

    def fake_job(img, job, out, bench_dir, cache_dir, gpu=True):
        ran.append(("job", job["engine_id"]))
        with open(out / "utterances.jsonl", "a") as f:
            f.write(
                json.dumps(
                    _row(engine_id=job["engine_id"], ref_norm="hi", hyp_raw="hi")
                )
                + "\n"
            )
        return {"engine_id": job["engine_id"], "exit": 0, "container": "c-1.0"}

    monkeypatch.setattr(runner, "TegraStats", FakeTegra)
    monkeypatch.setattr(runner, "run_container_job", fake_job)
    monkeypatch.setattr(runner, "_wait_for_free_gpu", lambda *a, **k: None)
    import bench.cachectl as cachectl

    monkeypatch.setattr(cachectl, "prepare_bench", lambda quiet=False: None)
    import bench.env as env_mod

    monkeypatch.setattr(
        env_mod,
        "collect",
        lambda git_commit="": {
            "git_commit": "c" * 7,
            "device_label": "ci",
            "compute_cap": "8.7",
        },
    )
    run_id = runner.run_config(str(cfg))
    assert run_id.endswith("ci_stt")  # ts device task; run_name lives in history
    out = next((root / "results").iterdir())
    env = json.loads((out / "env.json").read_text())
    assert env["config"]["task"] == "stt"
    assert ("job", "e1_0") in ran and ("job", "e2") in ran  # default id gains _0
    assert (out / "container_e1_0.json").exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["per_engine"]["e1_0"]["n_ok"] == 1
    # baseline run_name -> raw publish, path-scrubbed
    raw = root / "benchmarks" / "raw" / out.name
    assert (raw / "env.json").exists()
    assert str(tmp_path) not in (raw / "env.json").read_text()
    hist = root / "benchmarks" / "history.jsonl"
    rows = [json.loads(x) for x in hist.read_text().splitlines()]
    assert {r["engine_id"] for r in rows} == {"e1_0", "e2"}
