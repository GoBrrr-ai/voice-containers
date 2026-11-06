"""history — persisted-artifact hygiene: path scrubbing + jsonl readback."""

import json

from bench.history import _scrub_paths, read_history


def test_scrub_paths_makes_artifacts_repo_relative(tmp_path):
    root = tmp_path.resolve()
    text = f"py={root}/bench/.venv/bin/python out={root}/results/x.json"
    assert (
        _scrub_paths(text, tmp_path) == "py=bench/.venv/bin/python out=results/x.json"
    )


def test_scrub_paths_is_idempotent_and_foreign_safe(tmp_path):
    once = _scrub_paths(f"{tmp_path.resolve()}/a.csv", tmp_path)
    assert once == "a.csv"
    assert _scrub_paths(once, tmp_path) == once
    foreign = "/home/someone-else/elsewhere/x.csv"
    assert _scrub_paths(foreign, tmp_path) == foreign


def test_read_history_roundtrip(tmp_path):
    hist = tmp_path / "benchmarks" / "history.jsonl"
    hist.parent.mkdir()
    rows = [{"engine_id": "a", "epoch": 2}, {"engine_id": "b", "epoch": 1}]
    hist.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert [r["engine_id"] for r in read_history(tmp_path)] == ["a", "b"]
