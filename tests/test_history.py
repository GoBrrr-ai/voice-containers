"""history — persisted-artifact hygiene: path scrubbing + jsonl readback."""

import json

import pytest

from pathlib import Path

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


def _run_dir(tmp_path: Path):
    """A finished run dir: summary + container map + env + derived metrics."""
    from bench.history import _load_run_dir

    run = tmp_path / "20260101_dev_stt"
    run.mkdir()
    (run / "summary.json").write_text(
        json.dumps({"per_engine": {"e1": {"n_ok": 2, "wer": 0.05}}})
    )
    (run / "container_e1.json").write_text(
        json.dumps(
            {
                "engine_id": "e1",
                "engine": "e",
                "container": "bench-e1-1798000000",
                "wall_ns": 1.5e9,
                "exit": 0,
            }
        )
    )
    (run / "env.json").write_text(
        json.dumps(
            {
                "device_label": "dev",
                "power_profile": "default",
                "git_commit": "abc1234",
                "compute_cap": "8.7",
                "config": {
                    "task": "tts",
                    "run_name": "baseline_x",
                    "dataset": {"manifest": "mf.jsonl", "name": "libritts_r"},
                    "engines": [{"name": "e", "image": "img:1"}],
                },
            }
        )
    )
    with open(run / "utterances.jsonl", "w") as f:
        for i in range(4):
            f.write(
                json.dumps(
                    {
                        "engine_id": "e1",
                        "status": "ok",
                        "synth_ns": 1.0e9 + i * 1e8,
                    }
                )
                + "\n"
            )
        f.write(json.dumps({"engine_id": "other", "status": "ok"}) + "\n")
        f.write("\n")  # blank line tolerated
    doc = _load_run_dir(run)
    assert doc["run_id"] == "20260101_dev_stt"
    (r,) = doc["rows"]
    assert r["engine_id"] == "e1" and r["task"] == "tts"
    # _derive_metrics: no ttfa anywhere -> synth p50 becomes ttfa (documented)
    assert r["metrics"]["synth_latency_s"] == pytest.approx(1.2e9 / 1e9)
    assert r["metrics"]["ttfa_s"] == r["metrics"]["synth_latency_s"]
    assert "image" in r and r["requested_backend"] == ""
    return run, doc


def test_load_run_dir_and_append(tmp_path):
    from bench.history import append_run, read_history

    run, doc = _run_dir(tmp_path)
    bench_root = tmp_path / "bench"
    bench_root.mkdir()
    rows = append_run(run, bench_root)
    assert len(rows) == 1
    hist = read_history(bench_root)
    assert hist[0]["host"] and hist[0]["epoch"] > 0
    assert hist[0]["metrics"]["n_ok"] == 2


def test_append_run_empty_is_noop(tmp_path):
    from bench.history import append_run

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "summary.json").write_text(json.dumps({"per_engine": {}}))
    assert append_run(empty, tmp_path) == []
    assert not (tmp_path / "benchmarks" / "history.jsonl").exists()


def test_publish_raw_scrubs_and_slims(tmp_path):
    from bench.history import publish_raw

    run = Path(_run_dir.__wrapped__(tmp_path)[0]) if False else _run_dir(tmp_path)[0]
    (run / "audio").mkdir(exist_ok=True)
    (run / "x.wav").write_bytes(b"RIFF")
    (run / "tegrastats.log").write_text("11-07-2026 12:00:00 x\n")
    bench_root = tmp_path / "bench2"
    bench_root.mkdir()
    dst = publish_raw(run, bench_root)
    names = {p.name for p in dst.iterdir()}
    assert "summary.json" in names and "env.json" in names
    assert "x.wav" not in names and "tegrastats.log" not in names
    text = (dst / "env.json").read_text()
    assert str(tmp_path) not in text  # scrubbed to repo-relative
    # republish is idempotent (rmtree first)
    dst2 = publish_raw(run, bench_root)
    assert dst2 == dst


def test_publish_raw_writes_slim_utterance_extract(tmp_path):
    """The committed extract carries chart fields only — the full record
    (refs, ttfa, utt ids) stays in results/; charts still get their
    ok-filter and the exact numbers."""
    import json

    from bench.history import UTT_EXTRACT_KEYS, publish_raw

    run = _run_dir(tmp_path)[0]
    full = {  # the results/ shape — none of the extra keys are needed
        "engine_id": "e1",
        "status": "ok",
        "utt_id": "6930-75918-0019",
        "proc_ns": 1_200_000_000,
        "synth_ns": 2_500_000_000,
        "audio_s": 6.5,
        "ref_norm": "a long reference transcript",
        "ttfa_ns": 300_000_000,
    }
    (run / "utterances.jsonl").write_text(
        "".join(json.dumps(full, ensure_ascii=False) + "\n" for _ in range(3)),
        encoding="utf-8",
    )
    dst = publish_raw(run, tmp_path / "b")
    lines = (dst / "utterances.jsonl").read_text().splitlines()
    assert len(lines) == 3
    row = json.loads(lines[0])
    assert set(row) == set(UTT_EXTRACT_KEYS)
    assert row["proc_ns"] == 1_200_000_000 and row["audio_s"] == 6.5
    assert row["status"] == "ok"  # charts' ok-filter stays expressible
    # no utterances -> no extract file, publish still fine
    (run / "utterances.jsonl").unlink()
    dst2 = publish_raw(run, tmp_path / "b")
    assert not (dst2 / "utterances.jsonl").exists()


def test_load_run_dir_tolerates_odd_container_names(tmp_path):
    from bench.history import _load_run_dir, _derive_metrics

    run = tmp_path / "r"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"per_engine": {"e1": {"n_ok": 1}}}))
    # container name without a trailing epoch -> start_epoch 0.0 (no crash)
    (run / "container_e1.json").write_text(
        json.dumps({"engine_id": "e1", "container": "x"})
    )
    (run / "job_e1.json").write_text(json.dumps({"opts": {"model": "/m"}}))
    m = {"n_ok": 1}
    _derive_metrics(m, run, "e1")  # no utterances file -> unchanged
    assert "synth_latency_s" not in m
    # utterances with ttfa already present -> derive skipped
    (run / "utterances.jsonl").write_text(
        json.dumps({"engine_id": "e1", "status": "ok", "synth_ns": 2e9, "ttfa_ns": 3e8})
        + "\n"
    )
    m2 = {"n_ok": 1}
    _derive_metrics(m2, run, "e1")
    assert "ttfa_s" not in m2 and "synth_latency_s" in m2
    doc = _load_run_dir(run)
    assert doc["rows"][0]["opts"] == {"model": "/m"}
