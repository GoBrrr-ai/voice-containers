"""README rendering from raw benchmark data.

``render_readme`` renders ``README.md`` from ``README.md.j2`` with:
- ``ctx``          — headline numbers + per-task colored tables from history,
                     plus the Packages tables (versions parsed from build.sh)
- ``chart(name)``  — ``<img>`` resolved from assets/benchmarks/manifest.json
- ``root``         — relative asset prefix

StrictUndefined: a missing key is a hard error, never a silent blank.
"""

from __future__ import annotations
import json
import re
import time
from pathlib import Path
from typing import Any

import jinja2

from . import matrix as _matrix
from . import names as _names

GREEN = "#3fb950"
RED = "#f85149"
GREY = "grey"


def color(text: str, c: str = GREY) -> str:
    """GitHub-renderable colored text (LaTeX math mode)."""
    return rf"$\color{{{c}}}{{\text{{{text}}}}}$"


def good(text: str) -> str:
    return color(text, GREEN)


def bad(text: str) -> str:
    return color(text, RED)


def _table(header: list[str], rows: list[list[str]], align: str = "right") -> str:
    sep = "--:" if align == "right" else "---"
    out = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + [sep] * (len(header) - 1)) + "|",
    ]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# table columns per task: (metric key, header). All lower-is-better.
COLS = {
    "stt": [
        ("wer", "WER"),
        ("cer", "CER"),
        ("ser", "SER"),
        ("hallucination_rate", "halluc."),
        ("rtf", "RTF p50"),
        ("rtf_p90", "RTF p90"),
        ("offline_latency_s", "lat p50 (s)"),
        ("offline_latency_s_p90", "lat p90 (s)"),
    ],
    "tts": [
        ("rtf", "RTF p50"),
        ("rtf_p90", "RTF p90"),
        ("synth_latency_s", "synth p50 (s)"),
        ("synth_latency_s_p90", "synth p90 (s)"),
        ("ttfa_s", "TTFA p50 (s)"),
        ("ttfa_s_p90", "TTFA p90 (s)"),
    ],
}


def _colored_table(rows: list[dict], task: str) -> str:
    """Markdown matrix: original engine + model names, p50/p90 columns,
    direction-aware coloring (best-in-column green, worst red)."""
    heads = (
        ["engine", "model", "backend"]
        + [h for _, h in COLS[task]]
        + ["cgroup MiB", "anon MiB", "GPU W"]
    )
    # per-column best/worst for coloring
    best: dict[str, float] = {}
    worst: dict[str, float] = {}
    for k, _ in COLS[task]:
        vals = [_metric_v(r, k) for r in rows if _metric_v(r, k) is not None]
        if len(vals) < 2:
            continue
        best[k], worst[k] = min(vals), max(vals)  # every column is lower-is-better
    body = []
    for r in sorted(rows, key=lambda x: x.get("rank", 999)):
        is_cpu = (r.get("requested_backend") or "").lower() == "cpu"
        cells = [
            _names.medal(r) + _names.project(r),  # 🥇🥈🥉 + NBSP
            _names.display(r),
            r.get("requested_backend", "?"),
        ]
        for k, _ in COLS[task]:
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
        anon = _metric_v(r, "anon_peak_mib")
        cells.append(f"{anon:,.0f}" if anon else "—")
        if is_cpu:
            cells.append("— (CPU)")  # module-wide idle draw, not the engine's
        else:
            pw = _metric_v(r, "gpu_power_w_mean")
            cells.append(f"{pw:,.0f}" if pw else "—")
        body.append(cells)
    return _table(heads, body)


def _metric_v(r: dict, k: str):
    return _matrix._metric(r, k)


def _package_version(build_sh: Path) -> str:
    """Image version from a package's build.sh — the single source of truth."""
    m = re.search(r'\bVERSION="\$\{VERSION:-(?P<v>[^"}]+)\}"', build_sh.read_text())
    if not m:
        raise ValueError(f"cannot parse VERSION from {build_sh}")
    return m.group("v")


def _packages_table(rows: list[dict], task: str) -> str:
    heads = [
        "Family",
        "Package",
        "Image (`narandill/<name>`)",
        "What it is",
    ]
    body = [
        [
            f"{r['emoji']}\u00a0{r['family']}",
            f"[`{r['dir']}`](packages/{task}/{r['dir']}/)",
            f"`{r['dir']}:{r['version']}`",
            f"{r['what']}; {r['streaming']}",
        ]
        for r in rows
    ]
    return _table(heads, body, align="left")


def collect_packages(bench_root: Path) -> dict[str, str]:
    """Markdown Packages tables per task.

    Image name + version are parsed from each package's ``build.sh`` (the
    README can never drift from the build); descriptions and streaming notes
    come from ``names.PACKAGES`` — every new package must be registered there.
    ``packages/llm`` is outside the voice stack and not listed."""
    root = Path(bench_root) / "packages"
    dirs = {
        task: sorted(
            (d for d in (root / task).iterdir() if (d / "build.sh").exists()),
            key=lambda d: (_names.family({"image": d.name}), d.name),
        )
        for task in ("stt", "tts")
    }
    stale = set(_names.PACKAGES) - {d.name for ds in dirs.values() for d in ds}
    if stale:
        raise KeyError(f"names.PACKAGES entries with no package dir: {sorted(stale)}")
    out: dict[str, str] = {}
    for task, pkgs in dirs.items():
        rows = []
        for d in pkgs:
            name = d.name
            if name not in _names.PACKAGES:
                raise KeyError(f"packages/{task}/{name} missing from names.PACKAGES")
            fam = _names.family({"image": name})
            rows.append(
                {
                    "dir": name,
                    "version": _package_version(d / "build.sh"),
                    "emoji": _names.EMOJI.get(fam, "🔧"),
                    "family": fam,
                    **_names.PACKAGES[name],
                }
            )
        out[task] = _packages_table(rows, task)
    return out


def _data_date(rows: list[dict]) -> str:
    """Render date derived from the data: the newest utc_ts among the selected
    runs. Keeps README.md reproducible on any day (CI re-renders and diffs
    it); wall clock only as last resort for empty selections."""
    stamps = sorted(str(r.get("utc_ts") or "") for r in rows)
    return stamps[-1][:10] if stamps and stamps[-1] else time.strftime("%Y-%m-%d")


def collect_context(bench_root: Path, include: str = "baseline") -> dict:
    bench_root = Path(bench_root)
    rows = _matrix.select_rows(bench_root, include)
    ctx: dict[str, Any] = {
        "date": _data_date(rows),
        "commit": (rows[0].get("git_commit", "") if rows else ""),
        "device": (rows[0].get("device_label", "") if rows else ""),
        "power": (rows[0].get("power_profile", "") if rows else ""),
        "packages": collect_packages(bench_root),
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
            "winner": _names.display(winner),
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


def _env(bench_root: Path) -> jinja2.Environment:
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
    env.globals.update(chart=chart)
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
    env = _env(bench_root)
    ctx = collect_context(bench_root, include)
    tpl = env.from_string(Path(template).read_text())
    out.write_text(tpl.render(ctx=ctx))
    return out
