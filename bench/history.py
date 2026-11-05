"""Append-only benchmark history.

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


def _derive_metrics(m: dict, run_dir: Path, eid: str) -> None:
    """Fill metric gaps from the raw utterances so tables/charts are complete.

    - synth p50/p90 when an older runner didn't aggregate them;
    - TTFA for single-chunk engines: no ttfa_ns on any utterance means the
      client receives the whole clip in one response, so time-to-first-audio
      equals the synthesis completion time (documented in the README).
    """
    utt = run_dir / "utterances.jsonl"
    synth, ttfa = [], 0
    if utt.exists():
        for line in utt.read_text().splitlines():
            if not line.strip():
                continue
            u = json.loads(line)
            if u.get("engine_id") != eid or u.get("status") != "ok":
                continue
            if u.get("synth_ns"):
                synth.append(u["synth_ns"] / 1e9)
            if u.get("ttfa_ns"):
                ttfa += 1
    if "synth_latency_s" not in m and synth:
        synth.sort()
        m["synth_latency_s"] = synth[len(synth) // 2]
        m["synth_latency_s_p90"] = synth[min(len(synth) - 1, int(len(synth) * 0.9))]
    if m.get("ttfa_s") is None and not ttfa and m.get("synth_latency_s") is not None:
        m["ttfa_s"] = m["synth_latency_s"]
        m["ttfa_s_p90"] = m.get("synth_latency_s_p90", m["synth_latency_s"])


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
        m = summary["per_engine"][eid]
        _derive_metrics(m, run_dir, eid)
        # probe windows: the container name embeds its start epoch
        try:
            start_epoch = float(meta.get("container", "").rsplit("-", 1)[-1])
        except ValueError:
            start_epoch = 0.0
        win = _probes.window_aggregate(
            ts_rows, start_epoch, (meta.get("wall_ns") or 0) / 1e9
        )
        mem = _probes.parse_docker_stats(run_dir / f"docker_stats_{eid}.csv")
        # multi-backend engines get a _cpu/_cuda suffix on their engine_id
        import re as _re

        base_eid = _re.sub(r"_(cpu|cuda)$", "", eid)
        eng_cfg = next(
            (
                e
                for e in (env.get("config", {}) or {}).get("engines", [])
                if e.get("name") == meta.get("engine")
                and (
                    e.get("id")
                    or f"{e['name']}_{(env.get('config', {}).get('engines') or []).index(e)}"
                )
                == base_eid
            ),
            {},
        )
        rows.append(
            {
                "engine_id": eid,
                "engine": meta.get("engine"),
                "image": meta.get("image") or (eng_cfg.get("image") or ""),
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
                "task": (env.get("config", {}).get("task", "")),
                "mem_peak_mib": (mem or {}).get("mem_peak_mib"),
                "anon_peak_mib": (mem or {}).get("anon_peak_mib"),
                "anon_p50_mib": (mem or {}).get("anon_p50_mib"),
                "gpu_power_w_mean": (win or {}).get("gpu_power_w_mean"),
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


def _scrub_paths(text: str, bench_root: Path) -> str:
    """Anonymize absolute host paths in persisted artifacts: the repo prefix
    becomes repo-relative (e.g. datasets/cache/...), so committed raws carry
    no username/home layout. Idempotent; anything else passes through."""
    prefix = f"{bench_root.resolve()}/"
    return text.replace(prefix, "")


def publish_raw(run_dir: Path, bench_root: Path) -> Path:
    """Copy a run's raw artifacts (no audio) into benchmarks/raw/<run_id>/.

    Text artifacts (json/jsonl/csv/logs) get host-path scrubbing — committed
    raws stay repo-relative (no /home/<user> leakage); the results/ copies
    keep runtime-absolute paths.
    """
    run_dir = Path(run_dir)
    bench_root = Path(bench_root)
    dst = bench_root / "benchmarks" / "raw" / run_dir.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir()
    scrub_suffixes = {".json", ".jsonl", ".csv", ".txt"}
    for p in run_dir.iterdir():
        if p.name in RAW_EXCLUDE or p.suffix == ".wav":
            continue
        if p.is_file():
            target = dst / p.name
            if p.suffix in scrub_suffixes:
                text = p.read_text(errors="replace")
                target.write_text(_scrub_paths(text, bench_root))
            else:
                shutil.copy2(p, target)
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
