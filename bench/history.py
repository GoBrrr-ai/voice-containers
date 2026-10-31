"""Append-only benchmark history (mirrors mjolnir's history.jsonl).

Every ``bench run`` appends one line per (engine, backend) config to
``benchmarks/history.jsonl`` — ts, host, device, commit, run_id, config and
all headline metrics. ``bench plot`` renders charts and the README from this
file plus the raw run dirs it references.
"""

from __future__ import annotations
import json
import platform
import shutil
import time
from pathlib import Path

from . import probes as _probes

RAW_EXCLUDE = {"audio"}  # TTS wavs stay out of the committed raws


def _load_run_dir(run_dir: Path) -> dict:
    """Collect per-engine rows from a finished run dir (summary + jobs + env)."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    env = (
        json.loads((run_dir / "env.json").read_text())
        if (run_dir / "env.json").exists()
        else {}
    )
    rows = []
    ts_rows = _probes.parse_tegrastats(run_dir / "tegrastats.log")
    for cid in sorted(run_dir.glob("container_*.json")):
        meta = json.loads(cid.read_text())
        eid = meta.get("engine_id")
        if not eid or eid not in summary.get("per_engine", {}):
            continue
        job_p = run_dir / f"job_{eid}.json"
        job = json.loads(job_p.read_text()) if job_p.exists() else {}
        opts = job.get("opts", {})
        # probe windows: the container name embeds its start epoch
        try:
            start_epoch = float(meta.get("container", "").rsplit("-", 1)[-1])
        except ValueError:
            start_epoch = 0.0
        win = _probes.window_aggregate(
            ts_rows, start_epoch, (meta.get("wall_ns") or 0) / 1e9
        )
        mem = _probes.parse_docker_stats(run_dir / f"docker_stats_{eid}.csv")
        eng_cfg = next(
            (
                e
                for e in (env.get("config", {}) or {}).get("engines", [])
                if e.get("name") == meta.get("engine")
                and (
                    e.get("id")
                    or f"{e['name']}_{(env.get('config', {}).get('engines') or []).index(e)}"
                )
                == eid
            ),
            {},
        )
        rows.append(
            {
                "engine_id": eid,
                "engine": meta.get("engine"),
                "image": meta.get("image", ""),
                "model": opts.get("model")
                or opts.get("voice")
                or opts.get("model_dir", ""),
                "requested_backend": meta.get("requested_backend", ""),
                "exit": meta.get("exit"),
                "wall_s": (meta.get("wall_ns") or 0) / 1e9,
                "metrics": summary["per_engine"][eid],
                "opts": {k: v for k, v in opts.items() if k != "device"},
                "dataset": (env.get("config", {}).get("dataset", {}) or {}),
                "manifest": (env.get("config", {}).get("dataset", {}) or {}).get(
                    "manifest", ""
                ),
                "run_name": (env.get("config", {}).get("run_name", "")),
                "mem_peak_mib": (mem or {}).get("mem_peak_mib"),
                # CPU rows: GPU power during their window is idle draw — not a metric
                "gpu_power_w_mean": (win or {}).get("gpu_power_w_mean")
                if (meta.get("requested_backend") or "cuda") != "cpu"
                else None,
                "gpu_util_pct_mean": (win or {}).get("gpu_util_pct_mean"),
                "n_probe_samples": (win or {}).get("n_samples", 0) if win else 0,
            }
        )
    return {
        "run_id": run_dir.name,
        "device_label": env.get("device_label", platform.node()),
        "power_profile": env.get("power_profile", "default"),
        "git_commit": env.get("git_commit", ""),
        "utc_ts": env.get("utc_ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        "compute_cap": env.get("compute_cap", ""),
        "rows": rows,
    }


def append_run(run_dir: Path, bench_root: Path) -> list[dict]:
    """Append one line per engine config; returns the appended rows."""
    run_dir = Path(run_dir)
    bench_root = Path(bench_root)
    doc = _load_run_dir(run_dir)
    if not doc["rows"]:
        return []
    host = platform.node()
    hist = bench_root / "benchmarks" / "history.jsonl"
    hist.parent.mkdir(parents=True, exist_ok=True)
    with open(hist, "a", encoding="utf-8") as f:
        for r in doc["rows"]:
            line = {k: v for k, v in doc.items() if k != "rows"} | r
            line["host"] = host
            line["epoch"] = time.time()
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return doc["rows"]


def publish_raw(run_dir: Path, bench_root: Path) -> Path:
    """Copy a run's raw artifacts (no audio) into benchmarks/raw/<run_id>/."""
    run_dir = Path(run_dir)
    bench_root = Path(bench_root)
    dst = bench_root / "benchmarks" / "raw" / run_dir.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir()
    for p in run_dir.iterdir():
        if p.name in RAW_EXCLUDE or p.suffix == ".wav":
            continue
        if p.is_file():
            shutil.copy2(p, dst / p.name)
    return dst


def read_history(bench_root: Path) -> list[dict]:
    hist = Path(bench_root) / "benchmarks" / "history.jsonl"
    if not hist.exists():
        return []
    rows = []
    for line in hist.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows
