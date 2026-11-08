"""Repo-root conftest: keeps ``bench.*`` imports resolvable under any pytest
invocation. The Makefile uses ``python -m pytest`` (which adds the cwd to
sys.path itself); the root conftest also covers a bare ``pytest`` run.

Also hosts the shared synthetic-bench fixture: a complete tmp bench tree
(history + manifests + raw utterances) that drives the full render pipeline
(``select_rows → score_cohort → every chart``) with zero GPU/docker/network.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest


# Two tiers of "engine quality" so every norm/medal/pareto branch has spread.
def _stt_metrics(tier: int) -> dict:
    return {
        "n_ok": 2,
        "wer": 0.02 + 0.04 * tier,
        "rtf": 0.20 + 0.30 * tier,
        "rtf_p90": 0.25 + 0.35 * tier,
        "rtf_p95": 0.30 + 0.40 * tier,
        "offline_latency_s": 0.25 + 0.45 * tier,
        "offline_latency_s_p95": 0.30 + 0.50 * tier,
        "mem_peak_mib": 900 + 500 * tier,
        "anon_peak_mib": 700 + 500 * tier,
        "gpu_power_w_mean": 14.0 - 4.0 * tier,
    }


def _tts_metrics(tier: int) -> dict:
    return {
        "n_ok": 2,
        "rtf": 0.10 + 0.20 * tier,
        "rtf_p90": 0.15 + 0.25 * tier,
        "rtf_p95": 0.18 + 0.30 * tier,
        "synth_latency_s": 0.4 + 0.6 * tier,
        "synth_latency_s_p90": 0.5 + 0.7 * tier,
        "ttfa_s": 0.3 + 0.5 * tier,
        "ttfa_s_p90": 0.35 + 0.55 * tier,
        "mem_peak_mib": 700 + 400 * tier,
        "anon_peak_mib": 600 + 300 * tier,
        "gpu_power_w_mean": 9.0 - 2.0 * tier,
    }


def _row(task: str, run_id: str, engine: str, eid: str, tier: int, **kw) -> dict:
    r = {
        "epoch": 1_798_000_000.0 + tier,
        "run_id": run_id,
        "run_name": "baseline_smoke",
        "host": "pytest",
        "device_label": "jetson-test",
        "power_profile": "default",
        "compute_cap": "8.7",
        "utc_ts": "2026-11-07T00:00:00Z",
        "task": task,
        "manifest": f"manifests/{task}_smoke.jsonl",
        "engine": engine,
        "engine_id": eid,
        "image": f"narandill/{engine}:0.1.0",
        "requested_backend": kw.pop("requested_backend", ""),
        "model": "m",
        "exit": 0,
        "wall_s": 12.5,
        "opts": {},
        "dataset": {"name": "librispeech"},
        "metrics": _stt_metrics(tier) if task == "stt" else _tts_metrics(tier),
    }
    r.update(kw)
    return r


def _utterances(task: str, eid: str) -> list[dict]:
    if task == "stt":
        return [
            {
                "engine_id": eid,
                "status": "ok",
                "proc_ns": 1.2e8 + i * 1e7,
                "audio_s": 4.0,
            }
            for i in range(6)
        ] + [{"engine_id": eid, "status": "skip"}]
    return [
        {
            "engine_id": eid,
            "status": "ok",
            "synth_ns": 3.0e8 + i * 2e7,
            "ttfa_ns": 1.1e8 + i * 1e7,
        }
        for i in range(6)
    ]


@pytest.fixture()
def bench_env(tmp_path: Path):
    """A complete miniature bench root: 2 STT + 2 TTS rows across two runs.

    The second run of the same engines is incomplete (n_ok < manifest) so
    ``select_rows`` has a superseded row to drop, and one STT engine has a
    CPU twin so label/medal dedup branches engage.
    """
    root = tmp_path / "bench"
    (root / "manifests").mkdir(parents=True)
    for task, n in (("stt", 2), ("tts", 2)):
        (root / "manifests" / f"{task}_smoke.jsonl").write_text(
            "utt-1\nutt-2\n" if n == 2 else "utt-1\n"
        )

    stt = [
        _row("stt", "run_a", "parakeet-tdt-cpp-stt", "parakeet_cpp_0", 0),
        _row(
            "stt",
            "run_a",
            "parakeet-tdt-cpp-stt",
            "parakeet_cpp_1",
            1,
            requested_backend="cpu",
        ),
        # same engine, later incomplete run -> superseded by run_a
        _row("stt", "run_b", "parakeet-tdt-cpp-stt", "parakeet_cpp_0", 1),
    ]
    tts = [
        _row("tts", "run_a", "kokoro-tts", "kokoro_0", 0),
        _row("tts", "run_a", "pocket-tts", "pocket_0", 1),
    ]
    lines = []
    for r in stt + tts:
        r["metrics"]["n_ok"] = 2 if r["run_id"] == "run_a" else 1
        lines.append(json.dumps(r))
    (root / "benchmarks").mkdir()
    (root / "benchmarks" / "history.jsonl").write_text("\n".join(lines) + "\n")

    for run_id, task, rows in (
        ("run_a", "stt", [r for r in stt if r["run_id"] == "run_a"]),
        ("run_a", "tts", list(tts)),
    ):
        d = root / "benchmarks" / "raw" / run_id
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "utterances.jsonl", "a") as f:
            for r in rows:
                for line in _utterances(task, r["engine_id"]):
                    f.write(json.dumps(line) + "\n")

    # packages/ mirror: only the build.sh files, so collect_packages sees the
    # full roster (it cross-checks names.PACKAGES) without copying trees
    repo = Path(__file__).resolve().parents[1]
    for task in ("stt", "tts", "llm"):
        for build_sh in sorted((repo / "packages" / task).glob("*/build.sh")):
            dst = root / "packages" / task / build_sh.parent.name
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(build_sh, dst / "build.sh")

    return SimpleNamespace(root=root, stt=stt[:2], tts=tts)
