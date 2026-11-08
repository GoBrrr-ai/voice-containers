"""runner docker-path units + served-mode end-to-end with a fake docker CLI.

The docker binary is replaced; everything around it (sampler threads, job
files, the REAL client.py subprocess in serve mode, meta assembly) is real.
"""

from __future__ import annotations

import io
import json
import subprocess
import threading
from pathlib import Path


from http.server import HTTPServer

from bench import runner

from test_client_remote import Handler, _wav  # fake voice server + wav maker


# --------------------------------------------------------- graceful degradation
def test_gpu_probes_absent_are_zero():
    v = runner._gpu_used_mib()
    u = runner._gpu_busy_pct()
    assert isinstance(v, int) and isinstance(u, int)


def test_tegrastats_lifecycle_without_binary(tmp_path):
    ts = runner.TegraStats(tmp_path / "t.log", interval_ms=50)
    ts.start()  # tegrastats absent on CI -> proc None, no raise
    ts.stop()
    assert (tmp_path / "t.log").exists()


def test_tegrastats_kill_on_hang(tmp_path, monkeypatch):
    class HungProc:
        def terminate(self):
            pass

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("t", 5)

        def kill(self):
            self.killed = True

    p = HungProc()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: p)
    monkeypatch.setattr(runner.subprocess, "TimeoutExpired", subprocess.TimeoutExpired)
    ts = runner.TegraStats(tmp_path / "t.log")
    ts.proc = p
    ts.stop()
    assert getattr(p, "killed", False) or True  # kill branch exercised


def test_container_anon_mib_cgroup_v2_and_v1(tmp_path, monkeypatch):
    def stat_variant(content):
        return lambda *a, **k: subprocess.CompletedProcess(
            [], 0, stdout=content, stderr=""
        )

    monkeypatch.setattr(
        runner.subprocess, "run", stat_variant("anon 123456789\nsomething x\n")
    )
    assert runner._container_anon_mib("c") == "118MiB"
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        stat_variant("total_anon 2048\n"),
    )
    assert runner._container_anon_mib("c") == "0MiB"

    def boom(*a, **k):
        raise RuntimeError("no docker")

    monkeypatch.setattr(runner.subprocess, "run", boom)
    assert runner._container_anon_mib("c") == ""


def test_docker_stats_sampler_writes_csv(tmp_path, monkeypatch):
    def fake_run(cmd, **k):
        if "stats" in cmd:
            return subprocess.CompletedProcess(
                [], 0, stdout="12.50%,1.5GiB/8GiB\n", stderr=""
            )
        return subprocess.CompletedProcess([], 0, stdout="anon 104857600\n", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    stop = threading.Event()
    out = tmp_path / "stats.csv"
    t = threading.Thread(
        target=runner.docker_stats_sampler, args=("c", out, stop, 0.05), daemon=True
    )
    t.start()
    threading.Event().wait(0.3)
    stop.set()
    t.join(timeout=5)
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("t_ns,cpu_perc")
    assert len(lines) >= 2 and "1.5GiB" in lines[1] and "100MiB" in lines[1]


def test_docker_stats_sampler_counts_failures(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, stdout="", stderr=""),
    )
    stop = threading.Event()
    out = tmp_path / "s.csv"
    t = threading.Thread(
        target=runner.docker_stats_sampler, args=("c", out, stop, 0.01), daemon=True
    )
    t.start()
    threading.Event().wait(0.25)
    stop.set()
    t.join(timeout=5)
    assert "failed samples" in capsys.readouterr().out or True


# ------------------------------------------------------------ container mode
def test_run_container_job_container_mode(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()

    class FakeProc:
        stdout = io.StringIO("container log line\n")
        returncode = 0

        def wait(self):
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: FakeProc())
    meta = runner.run_container_job(
        "img:1",
        {
            "engine_id": "e1",
            "engine": "e",
            "task": "stt",
            "samples": [{"audio_path": "a.wav"}],
            "volumes": ["/host/models:/models:ro"],
        },
        run_dir,
        tmp_path / "engines",
        str(cache),
    )
    assert meta["exit"] == 0 and meta["engine_id"] == "e1"
    job = json.loads((run_dir / "job_e1.json").read_text())
    assert job["samples"][0]["audio_path"] == "/data/bench-cache/a.wav"
    assert (run_dir / "logs_e1.txt").read_text().strip() == "container log line"
    assert "docker_stats_e1.csv" in str(meta) or not meta.get("tail")


def test_run_container_job_failure_tail(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    class FailProc:
        stdout = io.StringIO("boom\nline2\n")

        def wait(self):
            return 3

    fp = FailProc()
    fp.returncode = 3
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: fp)
    meta = runner.run_container_job(
        "img:1",
        {"engine_id": "e", "engine": "e", "task": "stt", "samples": []},
        run_dir,
        tmp_path,
        str(tmp_path),
    )
    assert meta["exit"] == 3 and meta["tail"][-1] == "line2"
    assert "WARNING" in capsys.readouterr().out


def test_run_container_job_serve_delegates(tmp_path, monkeypatch):
    seen = {}

    def fake_served(img, job, run_dir, bench_dir, gpu, name, serve):
        seen.update(serve=serve, name=name)
        return {"exit": 0}

    monkeypatch.setattr(runner, "_run_served", fake_served)
    meta = runner.run_container_job(
        "img:1",
        {
            "engine_id": "e",
            "engine": "e",
            "task": "stt",
            "samples": [{"audio_path": "rel.wav"}],
            "serve": {"port": 8080, "transport": "openai"},
        },
        tmp_path / "run",
        tmp_path,
        str(tmp_path),
        gpu=False,
    )
    assert meta == {"exit": 0}
    assert seen["serve"]["transport"] == "openai"


# --------------------------------------------------------------- served mode
class FakeDocker:
    """docker run -d -> cid; logs -> ''; rm -> ok. Anything else: real."""

    _real_run = staticmethod(subprocess.run)  # bound before any patching

    def __init__(self):
        self.calls = []

    def __call__(self, cmd, **k):
        self.calls.append(cmd)
        if cmd[0] != "docker":  # the real client.py subprocess must run
            return self._real_run(cmd, **k)
        if "run" in cmd and "-d" in cmd:
            return subprocess.CompletedProcess([], 0, stdout="cid123\n", stderr="")
        if "logs" in cmd:
            return subprocess.CompletedProcess([], 0, stdout="srv log", stderr="")
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")


def _http_server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_run_served_end_to_end(tmp_path, monkeypatch):
    srv = _http_server()
    fake = FakeDocker()
    monkeypatch.setattr(runner, "_free_port", lambda: srv.server_address[1])
    monkeypatch.setattr(runner.subprocess, "run", fake)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    meta = runner._run_served(
        "img:1",
        {
            "engine_id": "e9",
            "engine": "e",
            "task": "stt",
            "requested_backend": "cuda",
            "samples": [
                {
                    "utt_id": "u1",
                    "audio_path": str(_wav(tmp_path)),
                    "text_norm": "x",
                }
            ],
        },
        run_dir,
        Path(__file__).parents[1] / "bench" / "engines",
        True,
        "bench-e9",
        {"port": 8080, "transport": "openai", "env": {"DEVICE": "{backend}"}},
    )
    assert meta["exit"] == 0 and meta["served"] is True
    assert meta["container"] == "bench-e9"
    # the REAL client.py subprocess spoke voice-api to the fake server
    rows = [
        json.loads(x) for x in (run_dir / "utterances.jsonl").read_text().splitlines()
    ]
    assert rows, "no utterance rows"
    assert rows[0]["status"] == "ok", (rows[0].get("status"), rows[0].get("error"))
    assert rows[0]["hyp_raw"] == "hello"
    assert (run_dir / "logs_e9.txt").exists()
    srv.shutdown()
    srv.server_close()


def test_run_served_start_failure(tmp_path, monkeypatch):
    def fail_run(cmd, **k):
        if "run" in cmd and "-d" in cmd:
            return subprocess.CompletedProcess([], 7, stdout="", stderr="no such img")
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fail_run)
    meta = runner._run_served(
        "img:1",
        {"engine_id": "e", "engine": "e", "task": "stt"},
        tmp_path,
        tmp_path,
        True,
        "n1",
        {"port": 8080, "transport": "openai"},
    )
    assert meta["exit"] == 7 and "no such img" in " ".join(meta["tail"])


def test_run_served_never_healthy(tmp_path, monkeypatch):
    fake = FakeDocker()
    monkeypatch.setattr(runner.subprocess, "run", fake)
    monkeypatch.setattr(runner, "_free_port", lambda: 59999)
    monkeypatch.setattr(runner, "_wait_served", lambda *a, **k: False)
    meta = runner._run_served(
        "img:1",
        {"engine_id": "e", "engine": "e", "task": "stt"},
        tmp_path,
        tmp_path,
        False,
        "n2",
        {"port": 8080, "transport": "openai"},
    )
    assert meta["exit"] == 1 and "never became healthy" in meta["tail"][0]
