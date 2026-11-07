"""cli.py — every command through typer's CliRunner with patched edges."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from bench.cli import app

runner = CliRunner()


def test_no_args_shows_help():
    r = runner.invoke(app, [])
    assert "prepare" in r.output and "run" in r.output


def test_validate_ok_and_bad(tmp_path: Path):
    good = tmp_path / "ok.json"
    good.write_text(
        json.dumps(
            {"task": "stt", "engines": [{"name": "e", "image": "i:1"}]},
        )
    )
    r = runner.invoke(app, ["validate", str(good)])
    assert r.exit_code == 0 and "config OK" in r.output and "1 engine(s)" in r.output

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"task": "stt", "engines": []}))
    r = runner.invoke(app, ["validate", str(bad)])
    assert r.exit_code != 0


def test_run_invokes_runner(tmp_path: Path, monkeypatch):
    import bench.runner as runner_mod

    monkeypatch.setattr(runner_mod, "run_config", lambda p: "123_device_stt")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("task: stt")
    r = runner.invoke(app, ["run", str(cfg)])
    assert r.exit_code == 0 and "run_id: 123_device_stt" in r.output


def test_plot_patches(tmp_path: Path, monkeypatch):
    import bench.charts as charts
    import bench.readme as readme_mod
    import bench.cli as cli_mod

    calls = []
    monkeypatch.setattr(
        charts, "render_all", lambda root, include: calls.append(("c", root, include))
    )
    monkeypatch.setattr(
        readme_mod,
        "render_readme",
        lambda root, include: calls.append(("r", root, include)) or tmp_path / "r.md",
    )
    r = runner.invoke(app, ["plot", "--no-readme", "--include", "x"])
    assert r.exit_code == 0
    assert calls and calls[0][0] == "c"
    assert not any(c[0] == "r" for c in calls)

    calls.clear()
    monkeypatch.setattr(cli_mod, "BENCH", tmp_path)
    r = runner.invoke(app, ["plot", "--include", "y"])
    assert r.exit_code == 0
    assert any(c[0] == "r" for c in calls)


def test_selfcheck_patches(tmp_path, monkeypatch):
    import bench.env as env_mod

    monkeypatch.setattr(env_mod, "selfcheck", lambda root: {"ok": True})
    r = runner.invoke(app, ["selfcheck"])
    assert r.exit_code == 0 and '"ok": true' in r.output


def test_grant_and_drop_caches_patches(tmp_path, monkeypatch):
    import bench.cachectl as cachectl

    monkeypatch.setattr(cachectl, "grant_drop_caches", lambda assume_yes: 0)
    r = runner.invoke(app, ["grant-permissions", "--yes"])
    assert r.exit_code == 0

    monkeypatch.setattr(cachectl, "drop_caches_granted", lambda: False)
    r = runner.invoke(app, ["drop-caches"])
    assert r.exit_code == 4 and "grant missing" in r.output

    monkeypatch.setattr(cachectl, "drop_caches_granted", lambda: True)
    monkeypatch.setattr(cachectl, "drop_caches", lambda: 0)
    r = runner.invoke(app, ["drop-caches"])
    assert r.exit_code == 0 and "page cache cleared" in r.output


def test_prune_dry_run_and_empty(tmp_path, monkeypatch):
    import bench.cli as cli_mod

    # no raw dir at all -> clean exit
    monkeypatch.setattr(cli_mod, "BENCH", tmp_path)
    r = runner.invoke(app, ["prune", "--yes"])
    assert r.exit_code == 0

    raw = tmp_path / "benchmarks" / "raw"
    for name in ("20260101_stt", "20260102_stt", "20260103_tts"):
        (raw / name).mkdir(parents=True)
        (raw / name / "x.json").write_text("{}")
    (tmp_path / "benchmarks" / "history.jsonl").write_text(
        json.dumps(
            {
                "run_id": "20260102_stt",
                "run_name": "baseline",
                "task": "stt",
                "engine_id": "e",
                "epoch": 1,
                "metrics": {"n_ok": 1},
            }
        )
        + "\n"
    )
    r = runner.invoke(app, ["prune"])  # dry-run default
    assert r.exit_code == 0 and "would remove" in r.output
    assert (raw / "20260101_stt").exists()

    r = runner.invoke(app, ["prune", "--yes"])
    assert r.exit_code == 0 and "removed" in r.output
    assert (raw / "20260102_stt").exists()  # protected: matrix row
    assert (raw / "20260103_tts").exists()  # newest of its task
    assert not (raw / "20260101_stt").exists()  # superseded, untracked -> rmtree
