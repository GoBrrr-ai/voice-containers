"""README rendering from raw benchmark data (mirrors mjolnir's readme.py).

``render_readme`` renders ``README.md`` from ``README.md.j2`` with:
- ``ctx``          — headline numbers + per-task colored tables from history
- ``chart(name)``  — ``<img>`` resolved from assets/benchmarks/manifest.json
- ``color(text, c)`` / ``good()`` / ``bad()`` — LaTeX colored table cells
- ``root``         — relative asset prefix

StrictUndefined: a missing key is a hard error, never a silent blank.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any

import jinja2

from . import matrix as _matrix

GREEN = "#3fb950"
RED = "#f85149"
GREY = "grey"


def color(text: str, c: str = GREY) -> str:
    """GitHub-renderable colored text (LaTeX math mode, as in mjolnir)."""
    return rf"$\color{{{c}}}{{\text{{{text}}}}}$"


def good(text: str) -> str:
    return color(text, GREEN)


def bad(text: str) -> str:
    return color(text, RED)


def _table(header: list[str], rows: list[list[str]]) -> str:
    out = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + ["--:"] * (len(header) - 1)) + "|",
    ]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _colored_table(rows: list[dict], task: str) -> str:
    """Markdown matrix with direction-aware coloring: best-in-column green,
    worst red (PRD 10.1)."""
    # mem/power have dedicated formatted columns — skip their raw twins
    skip = {"mem_peak_mib", "gpu_power_w_mean"}
    metric_keys = [h for h, _ in _matrix.METRICS[task] if h not in skip]
    heads = (
        ["engine", "model", "backend", "rank"]
        + [_pretty(k) for k in metric_keys]
        + ["mem MiB", "W", "gates"]
    )
    # per-column best/worst for coloring
    best: dict[str, float] = {}
    worst: dict[str, float] = {}
    for k, lower in _matrix.METRICS[task]:
        vals = [_matrix._metric(r, k) for r in rows if _metric_v(r, k) is not None]
        if len(vals) < 2:
            continue
        best[k], worst[k] = (min(vals), max(vals)) if lower else (max(vals), min(vals))
    body = []
    for r in sorted(rows, key=lambda x: x.get("rank", 999)):
        cells = [
            r.get("engine_id", "?"),
            _short_model(r.get("model", "")) if r.get("model") else "baked",
            r.get("requested_backend", "?"),
            str(r.get("rank", "")),
        ]
        for k, _ in _matrix.METRICS[task]:
            if k in skip:
                continue
            v = _metric_v(r, k)
            if v is None:
                cells.append("—")
                continue
            txt = f"{v:.4f}" if isinstance(v, float) and v < 10 else f"{v:,.1f}"
            if k in best and v == best[k]:
                cells.append(good(txt))
            elif k in worst and v == worst[k]:
                cells.append(bad(txt))
            else:
                cells.append(txt)
        mem = _metric_v(r, "mem_peak_mib")
        cells.append(f"{mem:,.0f}" if mem else "—")
        pw = _metric_v(r, "gpu_power_w_mean")
        cells.append(f"{pw:,.0f}" if pw else "—")
        cells.append(
            "; ".join(r.get("gates", []))
            or ("✓" if r.get("gate_pass", True) else bad("fail"))
        )
        body.append(cells)
    return _table(heads, body)


def _metric_v(r: dict, k: str):
    return _matrix._metric(r, k)


def _pretty(k: str) -> str:
    return {
        "wer": "WER",
        "cer": "CER",
        "ser": "SER",
        "hallucination_rate": "halluc.",
        "rtf": "RTF p50",
        "offline_latency_s": "lat p50 (s)",
        "synth_latency_s": "synth p50 (s)",
        "ttfa_s": "TTFA p50 (s)",
    }.get(k, k)


def _short_model(model: str) -> str:
    m = str(model).rsplit("/", 1)[-1]
    return m.replace(".gguf", "")[-30:] or "—"


def collect_context(bench_root: Path, include: str = "baseline") -> dict:
    bench_root = Path(bench_root)
    rows = _matrix.select_rows(bench_root, include)
    ctx: dict[str, Any] = {
        "date": time.strftime("%Y-%m-%d"),
        "commit": (rows[0].get("git_commit", "") if rows else ""),
        "device": (rows[0].get("device_label", "") if rows else ""),
        "power": (rows[0].get("power_profile", "") if rows else ""),
        "stt": None,
        "tts": None,
    }
    for task in ("stt", "tts"):
        task_rows = [
            r
            for r in rows
            if (
                r.get("task") or ("stt" if "wer" in (r.get("metrics") or {}) else "tts")
            )
            == task
        ]
        if not task_rows:
            continue
        scored = _matrix.score_cohort([dict(r) for r in task_rows], task)
        winner = scored[0]
        section = {
            "table": _colored_table(scored, task),
            "n_configs": len(scored),
            "winner_id": winner.get("engine_id"),
            "winner_model": _short_model(winner.get("model", "")),
        }
        if task == "stt":
            section["best_wer"] = f"{_metric_v(winner, 'wer') * 100:.2f}%"
            section["best_rtf"] = (
                f"{min(_metric_v(r, 'rtf') for r in scored if _metric_v(r, 'rtf')):.3f}"
            )
            section["n_utt"] = _metric_v(winner, "n") or 0
        else:
            section["best_rtf"] = (
                f"{min(_metric_v(r, 'rtf') for r in scored if _metric_v(r, 'rtf')):.3f}"
            )
            ttfas = [_metric_v(r, "ttfa_s") for r in scored if _metric_v(r, "ttfa_s")]
            section["best_ttfa_ms"] = f"{min(ttfas) * 1000:.0f}" if ttfas else None
            section["n_sent"] = _metric_v(winner, "n") or 0
        ctx[task] = section
    return ctx


def _manifest(bench_root: Path) -> dict:
    p = Path(bench_root) / "assets" / "benchmarks" / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _env(bench_root: Path, out_path: Path) -> jinja2.Environment:
    manifest = _manifest(bench_root)

    def chart(name: str, alt: str = "", width: str = "100%") -> str:
        entries = manifest.get(name) or []
        if not entries:
            return ""
        return f'<img src="{entries[0]["dark"]}" alt="{alt or name}" width="{width}">'

    env = jinja2.Environment(
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
        autoescape=False,
        loader=jinja2.FileSystemLoader(str(bench_root)),
    )
    env.globals.update(
        chart=chart,
        good=good,
        bad=bad,
        color=color,
        ctx=None,  # set per-render
    )
    return env


def render_readme(
    bench_root: Path,
    template: Path | None = None,
    out: Path | None = None,
    include: str = "baseline",
) -> Path:
    bench_root = Path(bench_root)
    template = template or bench_root / "README.md.j2"
    out = out or bench_root / "README.md"
    env = _env(bench_root, out)
    ctx = collect_context(bench_root, include)
    tpl = env.from_string(Path(template).read_text())
    out.write_text(tpl.render(ctx=ctx, root=""))
    return out
