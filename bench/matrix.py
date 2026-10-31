"""Results matrix (PRD §10.1–10.2): cohorts, scores, gates, Pareto, outputs.

Reads benchmarks/history.jsonl (written by every ``bench run``); never
re-runs inference. Rows are grouped into cohorts by
(task, device, power_profile, manifest) — rankings never span cohorts.
"""

from __future__ import annotations
import csv
import json
import math
from pathlib import Path

from . import probes as _probes
from .history import read_history

# v1 composite weights (PRD 10.2; quality fields absent until 7.3 lands)
WEIGHTS = {
    "stt": {
        "default": {
            "accuracy": 0.35,
            "latency": 0.25,
            "throughput": 0.10,
            "memory": 0.15,
            "power": 0.15,
        },
        "realtime_assistant": {
            "accuracy": 0.20,
            "latency": 0.40,
            "throughput": 0.10,
            "memory": 0.15,
            "power": 0.15,
        },
        "batch_transcription": {
            "accuracy": 0.50,
            "latency": 0.10,
            "throughput": 0.25,
            "memory": 0.08,
            "power": 0.07,
        },
        "low_power": {
            "accuracy": 0.20,
            "latency": 0.10,
            "throughput": 0.05,
            "memory": 0.30,
            "power": 0.35,
        },
    },
    "tts": {
        "default": {
            "latency": 0.35,
            "throughput": 0.15,
            "memory": 0.20,
            "power": 0.20,
            "ttfa": 0.10,
        },
        "realtime_assistant": {
            "latency": 0.30,
            "throughput": 0.10,
            "memory": 0.20,
            "power": 0.20,
            "ttfa": 0.20,
        },
        "batch_transcription": {
            "latency": 0.20,
            "throughput": 0.40,
            "memory": 0.20,
            "power": 0.20,
            "ttfa": 0.00,
        },
        "low_power": {
            "latency": 0.10,
            "throughput": 0.10,
            "memory": 0.30,
            "power": 0.40,
            "ttfa": 0.10,
        },
    },
}

# direction: True = lower is better
METRICS = {
    "stt": [
        ("wer", True),
        ("cer", True),
        ("ser", True),
        ("hallucination_rate", True),
        ("rtf", True),
        ("offline_latency_s", True),
        ("mem_peak_mib", True),
        ("gpu_power_w_mean", True),
    ],
    "tts": [
        ("rtf", True),
        ("synth_latency_s", True),
        ("ttfa_s", True),
        ("mem_peak_mib", True),
        ("gpu_power_w_mean", True),
    ],
}
LOG_METRICS = {
    "rtf",
    "offline_latency_s",
    "synth_latency_s",
    "ttfa_s",
    "mem_peak_mib",
    "gpu_power_w_mean",
}

# hard gates (pass/fail; None = not enforced) — PRD 10.2.1
GATES = {
    "stt": {"wer": ("<=", 0.10), "rtf": ("<=", 1.0)},
    "tts": {"rtf": ("<=", 1.0), "synth_latency_s": ("<=", 2.0)},
}


def _metric(row: dict, key: str):
    """Row metric value: plain keys + metrics-dict keys."""
    if key in row:
        return row[key]
    return (row.get("metrics") or {}).get(key)


def cohort_of(row: dict) -> tuple:
    return (
        row.get("task") or ("stt" if "wer" in (row.get("metrics") or {}) else "tts"),
        row.get("device_label", "?"),
        row.get("power_profile", "default"),
        row.get("manifest", "?"),
    )


def select_rows(bench_root: Path, include: str = "baseline") -> list[dict]:
    """Latest complete row per (cohort, engine_id) from history.

    ``include`` is a substring filter on run_name ("baseline" by default;
    "" / "all" selects everything). Completeness = n_ok equals the manifest
    line count (the frozen subset is the truth, not dataset.n).
    """
    rows = []
    seen: set[tuple] = set()
    manifest_n: dict[str, int] = {}

    def expected(manifest: str) -> int | None:
        if not manifest:
            return None
        if manifest not in manifest_n:
            p = Path(bench_root) / manifest
            manifest_n[manifest] = (
                sum(1 for line in p.read_text().splitlines() if line.strip())
                if p.exists()
                else -1
            )
        return manifest_n[manifest] or None

    for r in sorted(read_history(bench_root), key=lambda x: x.get("epoch", 0)):
        run_name = r.get("run_name", "")
        if include and include not in ("all", "") and include not in run_name:
            continue
        n_expected = expected(r.get("manifest", ""))
        m = r.get("metrics") or {}
        if n_expected and m.get("n_ok") != n_expected:
            continue  # incomplete run — superseded by a later full one
        key = (cohort_of(r), r.get("engine_id"), r.get("requested_backend"))
        if key in seen:
            rows = [
                x
                for x in rows
                if (cohort_of(x), x.get("engine_id"), x.get("requested_backend")) != key
            ]
        rows.append(r)
        seen.add(key)
    return rows


def _norm(values: list[float], lower_better: bool) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.5] * len(values)
    out = [(v - lo) / (hi - lo) for v in values]
    return [1 - x for x in out] if lower_better else out


def score_cohort(rows: list[dict], task: str, preset: str = "default") -> list[dict]:
    """Attach score/rank/pareto/gates to each row (in place)."""
    weights = WEIGHTS[task].get(preset, WEIGHTS[task]["default"])
    contrib = {}
    for mname, lower in METRICS[task]:
        vals = [_metric(r, mname) for r in rows]
        if any(v is None for v in vals):
            continue  # metric absent for some row — excluded from scoring
        if mname in LOG_METRICS:
            vals = [math.log10(max(v, 1e-9)) for v in vals]
        normed = _norm(vals, lower)
        w = weights.get(mname if mname != "wer" else "accuracy", 0)
        for r, nv in zip(rows, normed):
            contrib.setdefault(id(r), 0.0)
            contrib[id(r)] += w * nv
    for r in rows:
        r["score"] = contrib.get(id(r), 0.0)
        r["gates"] = []
        failed = False
        for gname, (op, bound) in GATES[task].items():
            v = _metric(r, gname)
            if v is None:
                continue
            ok = v <= bound if op == "<=" else v >= bound
            if not ok:
                failed = True
                r["gates"].append(f"{gname}={v:.4g} > {bound}")
        if m_err := (_metric(r, "n") or 0) - (_metric(r, "n_ok") or 0):
            if _metric(r, "n"):
                pass
        r["gate_pass"] = not failed
    rows.sort(key=lambda r: -r["score"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    # Pareto front on (accuracy/latency, memory) — non-dominated set
    def axes(r):
        acc = 1 - _metric(r, "wer") if task == "stt" else None
        lat = (
            _metric(r, "offline_latency_s")
            if task == "stt"
            else _metric(r, "synth_latency_s")
        )
        lat = math.log10(max(lat or 1e-9, 1e-9))
        mem = math.log10(max(_metric(r, "mem_peak_mib") or 1e-9, 1e-9))
        return (acc if acc is not None else -lat, -lat, -mem)

    pareto = []
    for i, r in enumerate(rows):
        a = axes(r)
        dominated = any(
            all(x >= y for x, y in zip(axes(o), a))
            and any(x > y for x, y in zip(axes(o), a))
            for j, o in enumerate(rows)
            if j != i
        )
        if not dominated:
            pareto.append(r["engine_id"])
    for r in rows:
        r["pareto"] = r["engine_id"] in pareto
    return rows


def cohorts(rows: list[dict]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for r in rows:
        out.setdefault(cohort_of(r), []).append(r)
    return out


def matrix_table(rows: list[dict], task: str) -> tuple[list[str], list[list[str]]]:
    """(headers, cells) for the markdown/HTML matrix of one cohort."""
    heads = ["engine", "model", "backend", "rank", "score", "pareto"]
    heads += [h for h, _ in METRICS[task]]
    heads += ["n_ok", "gate"]
    body = []
    for r in sorted(rows, key=lambda x: x.get("rank", 999)):
        m = r.get("metrics") or {}

        def fmt(k):
            v = _metric(r, k)
            if v is None:
                return "—"
            return f"{v:.4f}" if isinstance(v, float) and v < 10 else f"{v:,.1f}"

        cells = [
            r.get("engine_id", "?"),
            str(r.get("model", ""))[-38:],
            r.get("requested_backend", "?"),
            str(r.get("rank", "")),
            f"{r.get('score', 0):.3f}",
            "✓" if r.get("pareto") else "",
        ]
        cells += [fmt(h) for h, _ in METRICS[task]]
        cells += [str(m.get("n_ok", "")), "; ".join(r.get("gates", [])) or "pass"]
        body.append(cells)
    return heads, body


def write_outputs(
    coh: dict[tuple, list[dict]], task: str, out_dir: Path, preset: str = "default"
) -> dict:
    """matrix.csv + matrix.json (+ returns per-cohort tables for docs)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    tables = {}
    for cohort, rows in sorted(coh.items()):
        scored = score_cohort([dict(r) for r in rows], task, preset)
        heads, body = matrix_table(scored, task)
        tables[cohort] = {
            "heads": heads,
            "body": body,
            "device": cohort[1],
            "power": cohort[2],
            "manifest": cohort[3],
        }
        for r in scored:
            all_rows.append(
                {k: v for k, v in r.items() if k not in ("opts", "dataset")}
            )
    with open(out_dir / f"matrix_{task}.csv", "w", newline="") as f:
        if all_rows:
            keys = sorted({k for r in all_rows for k in r})
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in all_rows:
                w.writerow(
                    {
                        k: (
                            json.dumps(v, ensure_ascii=False)
                            if isinstance(v, (dict, list))
                            else v
                        )
                        for k, v in r.items()
                        if k in keys
                    }
                )
    (out_dir / f"matrix_{task}.json").write_text(
        json.dumps(all_rows, indent=1, default=str)
    )
    return tables
