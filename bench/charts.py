"""Chart generation from the benchmark history (mirrors mjolnir's charts.py).

``render_all`` builds every chart that has data into
``assets/benchmarks/<task>/<name>.png`` (dark theme, 2x) and writes
``manifest.json``; ``readme.py`` resolves ``{{ chart("name") }}`` from it.
Charts needing per-utterance data read the raw run dirs under
``benchmarks/raw/`` (fallback: results/).
"""

from __future__ import annotations
import json
import time
from pathlib import Path

from . import matrix as _matrix
from . import names as _names
from . import theme
from .theme import DARK, Mode, apply, footer, log_y_if_spanning, row_color

DPI = 200  # 2x PNG per spec 10.3 (vs matplotlib's default 100)


def _plt():
    return theme._plt()


def _resolve_raw(bench_root: Path, run_id: str) -> Path | None:
    for cand in (
        bench_root / "benchmarks" / "raw" / run_id,
        bench_root / "results" / run_id,
    ):
        if (cand / "utterances.jsonl").exists():
            return cand
    return None


def _load_utterances(bench_root: Path, run_id: str, engine_id: str) -> list[dict]:
    raw = _resolve_raw(bench_root, run_id)
    if not raw:
        return []
    rows = []
    for line in (raw / "utterances.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("engine_id") == engine_id and r.get("status") == "ok":
            rows.append(r)
    return rows


def _label(row: dict, star: bool = False) -> str:
    """Short series label from the original engine + model (never the config id).
    ``star`` marks the best row FOR THE CHART BEING DRAWN — the metric shown
    decides (lower is better everywhere), never the composite rank."""
    return ("★ " if star else "") + _names.display(row)


def _best_id(rows, key, lower=True):
    """id() of the row with the best (direction-aware) value of ``key``."""
    rows = [r for r in rows if _matrix._metric(r, key) is not None]
    if not rows:
        return None
    fn = min if lower else max
    return id(fn(rows, key=lambda r: _matrix._metric(r, key)))


def _title(ax, text: str, mode: Mode = DARK) -> None:
    """Consistent chart title: big, bold, left-aligned, light on dark."""
    ax.set_title(text, loc="left", fontsize=15, fontweight="bold",
                 color=mode.text, pad=22)


def _backend_marker(row: dict) -> str:
    return theme.BACKEND_MARKERS.get((row.get("requested_backend") or "cuda"), "D")


def _footer_common(fig, rows: list[dict], extra: str = "") -> None:
    r0 = rows[0]
    left = (
        f"edge-containers bench · {r0.get('device_label', '?')} · "
        f"{r0.get('power_profile', 'default')} power"
    )
    if fig.get_figwidth() >= 9:  # narrow figures: drop the manifest or the strings collide
        left += f" · …{r0.get('manifest', '?')[-20:]}"
    footer(
        fig,
        DARK,
        left=left,
        right=f"commit {r0.get('git_commit', '?')}{extra}",
    )


def _hbar(
    ax,
    names,
    values,
    colors,
    mode: Mode,
    fmt: str,
    xlabel: str,
    log: bool | None = None,
    label_x: list | None = None,
):
    """Horizontal bar with in-axes value labels + footer-safe layout."""
    if log is None:
        vals = [v for v in values if v and v > 0]
        log = bool(vals) and max(vals) / max(min(vals), 1e-12) >= 10
    ax.barh(names, values, color=colors, edgecolor=mode.grid)
    ax.invert_yaxis()
    if log:
        ax.set_xscale("log")
        lo = min(v for v in values if v > 0)
        hi = max(values)
        ax.set_xlim(lo * 0.8, hi * 1.45)
        offset = hi * 0.06
    else:
        hi = max(values)
        ax.set_xlim(0, hi * 1.18)
        offset = hi * 0.015
    for i, v in enumerate(values):
        default_x = (v * 1.06) if (log and v > 0) else (v + offset)
        x = label_x[i] if label_x else default_x
        ax.text(
            x,
            i,
            fmt.format(v),
            va="center",
            fontsize=9,
            color=mode.text,
            bbox=dict(boxstyle="round,pad=0.15", fc=mode.panel, ec="none", alpha=0.7),
        )
    ax.set_xlabel(xlabel)


# --------------------------------------------------------------------- STT
def stt_ranking(rows: list[dict], mode: Mode = DARK):
    rows = sorted(rows, key=lambda r: _matrix._metric(r, "wer") or 9)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    apply(fig, ax)
    best = _best_id(rows, "wer")
    names = [_label(r, star=id(r) == best) for r in rows]
    wers = [(_matrix._metric(r, "wer") or 0) * 100 for r in rows]
    colors = [row_color(r) for r in rows]
    _hbar(ax, names, wers, colors, mode, "{:.2f}%", "corpus WER % (lower is better ↓)")
    _title(ax, "Speech recognition — corpus WER")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _footer_common(fig, rows)
    return fig


def stt_wer_vs_rtf(rows: list[dict], mode: Mode = DARK):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    apply(fig, ax)
    pts = []
    for r in rows:
        wer, rtf = _matrix._metric(r, "wer"), _matrix._metric(r, "rtf")
        if wer is None or rtf is None:
            continue
        pts.append((r, wer * 100, rtf))
    if not pts:
        return None
    mems = [(_matrix._metric(r, "mem_peak_mib") or 0) for r, _, _ in pts]
    mmax = max(mems) or 1
    x_lo, x_hi = min(p[2] for p in pts), max(p[2] for p in pts)
    ax.set_xlim(x_lo * 0.55, x_hi * 2.6)

    # when most points crowd into a small region, blow it up in an inset
    import statistics

    med = statistics.median(p[2] for p in pts)
    cluster = [p for p in pts if med / 3 <= p[2] <= med * 3]
    zoomed = set()
    axins = None
    if len(cluster) >= 3 and len(cluster) < len(pts):
        c_lo, c_hi = min(p[2] for p in cluster), max(p[2] for p in cluster)
        w_lo, w_hi = min(p[1] for p in cluster), max(p[1] for p in cluster)
        if c_hi / c_lo < 3 and (x_hi / x_lo) > 3:  # tight cluster, wide span
            axins = ax.inset_axes([0.07, 0.10, 0.42, 0.52])
            apply(fig, axins)
            axins.set_facecolor("#202a36")  # lighter than the panel behind it
            axins.set_xlim(c_lo * 0.9, c_hi * 1.1)
            axins.set_ylim(
                w_lo - (w_hi - w_lo) * 0.35 - 0.05, w_hi + (w_hi - w_lo) * 0.35 + 0.05
            )
            axins.set_title(
                "parakeet cluster (zoom)", fontsize=8.5, color=mode.text_dim, loc="left"
            )
            axins.tick_params(labelsize=7)
            ax.indicate_inset_zoom(axins, edgecolor=mode.text_dim, alpha=0.6)
            zoomed = {id(p[0]) for p in cluster}

    def _dot(axes, r, wer, rtf, mem, size_scale=1.0, label=True):
        axes.scatter(
            rtf,
            wer,
            s=(60 + 240 * (mem / mmax)) * size_scale,
            color=row_color(r),
            marker=_backend_marker(r),
            edgecolor=mode.text_dim,
            linewidth=0.6,
            zorder=3,
        )
        if label:
            axes.annotate(
                _label(r),
                (rtf, wer),
                textcoords="offset points",
                xytext=(7, 5),
                fontsize=8.5 if size_scale == 1.0 else 8,
                color=mode.text,
            )

    for (r, wer, rtf), mem in zip(pts, mems):
        if id(r) in zoomed:
            _dot(ax, r, wer, rtf, mem, label=False)  # labeled in the inset
        else:
            _dot(ax, r, wer, rtf, mem)
    if axins is not None:
        for (r, wer, rtf), mem in zip(pts, mems):
            if id(r) in zoomed:
                _dot(axins, r, wer, rtf, mem, size_scale=1.15)
    # Pareto front (minimize WER then RTF) drawn as a step line
    front = sorted(
        _matrix.score_cohort([dict(r) for r in rows], "stt"),
        key=lambda r: r["pareto"] is False,
    )
    pf = sorted(
        [r for r in front if r.get("pareto")],
        key=lambda r: _matrix._metric(r, "rtf") or 0,
    )
    if len(pf) > 1:
        ax.step(
            [_matrix._metric(r, "rtf") for r in pf],
            [_matrix._metric(r, "wer") * 100 for r in pf],
            where="post",
            color=mode.green,
            linewidth=1.4,
            linestyle="--",
            alpha=0.8,
            zorder=2,
        )
        ax.annotate(
            "Pareto front",
            (0.99, 0.03),
            xycoords="axes fraction",
            ha="right",
            fontsize=8.5,
            color=mode.green,
        )
    ax.set_xlabel("RTF p50 — audio-s per s (lower is better ↓, log)")
    ax.set_ylabel("corpus WER % (lower is better ↓)")
    ax.set_xscale("log")
    _title(ax, "Speech recognition — accuracy vs speed")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _footer_common(fig, rows, extra=" · point size = peak container memory")
    return fig


def stt_latency_box(bench_root: Path, rows: list[dict], mode: Mode = DARK):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    apply(fig, ax)
    data, colors, rows_used = [], [], []
    for r in rows:
        utts = _load_utterances(bench_root, r["run_id"], r["engine_id"])
        rtfs = [
            u["proc_ns"] / 1e9 / max(u.get("audio_s") or 1, 1e-9)
            for u in utts
            if u.get("proc_ns")
        ]
        if not rtfs:
            continue
        data.append(rtfs)
        rows_used.append(r)
    if not data:
        return None
    best_i = min(
        range(len(data)), key=lambda i: sorted(data[i])[len(data[i]) // 2]
    )
    labels = [_label(r, star=i == best_i) for i, r in enumerate(rows_used)]
    bp = ax.boxplot(
        data, tick_labels=labels, vert=False, patch_artist=True, showfliers=False
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    for el in ("whiskers", "caps", "medians"):
        for line in bp[el]:
            line.set_color(mode.text_dim)
    ax.invert_yaxis()
    _title(ax, "Speech recognition — per-utterance RTF distribution")
    ax.set_xlabel("RTF per utterance (lower is better ↓, log)")
    ax.set_xscale("log")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _footer_common(fig, rows)
    return fig


def stt_memory(rows: list[dict], mode: Mode = DARK):
    return _memory_bars(rows, mode)


def stt_heatmap(rows: list[dict], mode: Mode = DARK):
    return _heatmap(rows, "stt", mode)


def stt_power(rows: list[dict], mode: Mode = DARK):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    apply(fig, ax)
    pts = [
        (r, _matrix._metric(r, "gpu_power_w_mean"), _matrix._metric(r, "wer"))
        for r in rows
        if _matrix._metric(r, "gpu_power_w_mean")
    ]
    if len(pts) < 2:
        return None
    w_lo, w_hi = min(p[1] for p in pts), max(p[1] for p in pts)
    ax.set_xlim(w_lo * 0.85, w_hi * 1.18)
    for r, w, wer in pts:
        is_cpu = (r.get("requested_backend") or "").lower() == "cpu"
        ax.scatter(
            w,
            (wer or 0) * 100,
            s=90,
            facecolor="none" if is_cpu else row_color(r),
            marker=_backend_marker(r),
            edgecolor=row_color(r),
            linewidth=1.6,
            zorder=3,
        )
        lbl = _label(r) + (" — CPU (GPU idle)" if is_cpu else "")
        right_side = w > w_lo + (w_hi - w_lo) * 0.7
        ax.annotate(
            lbl,
            (w, (wer or 0) * 100),
            textcoords="offset points",
            xytext=(-6, 5) if right_side else (6, 5),
            ha="right" if right_side else "left",
            fontsize=8.5,
            color=mode.text,
        )
    ax.set_xlabel(
        "mean GPU power over the engine window (W) — hollow = CPU engine (idle draw)"
    )
    ax.set_ylabel("corpus WER % (lower is better ↓)")
    _title(ax, "Speech recognition — GPU power vs accuracy")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _footer_common(fig, rows, extra=" · power from tegrastats job windows")
    return fig


# --------------------------------------------------------------------- TTS
def tts_ranking(rows: list[dict], mode: Mode = DARK):
    rows = sorted(rows, key=lambda r: _matrix._metric(r, "rtf") or 9)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    apply(fig, ax)
    best = _best_id(rows, "rtf")
    names = [_label(r, star=id(r) == best) for r in rows]
    rtfs = [_matrix._metric(r, "rtf") for r in rows]
    colors = [row_color(r) for r in rows]
    _hbar(
        ax,
        names,
        rtfs,
        colors,
        mode,
        "{:.3f}",
        "RTF p50 — s audio per s wall (lower is better ↓)",
    )
    _title(ax, "Speech synthesis — real-time factor")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _footer_common(fig, rows)
    return fig


def tts_ttfa(rows: list[dict], mode: Mode = DARK):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    apply(fig, ax)
    pts = [
        (r, _matrix._metric(r, "ttfa_s")) for r in rows if _matrix._metric(r, "ttfa_s")
    ]
    if not pts:
        return None
    pts.sort(key=lambda x: x[1])
    best = _best_id([r for r, _ in pts], "ttfa_s")
    names = [_label(r, star=id(r) == best) for r, _ in pts]
    vals = [v * 1000 for _, v in pts]
    colors = [row_color(r) for r, _ in pts]
    p90s = [(_matrix._metric(r, "ttfa_s_p90") or 0) * 1000 for r, _ in pts]
    label_x = [max(v, p) * 1.12 for v, p in zip(vals, p90s)]
    _hbar(
        ax,
        names,
        vals,
        colors,
        mode,
        "{:.0f} ms",
        "TTFA (ms, lower is better ↓)",
        log=True,
        label_x=label_x,
    )
    ax.set_xlim(min(vals) * 0.7, max(max(vals), max(p90s)) * 1.25)
    # p90 whisker ticks on each bar
    for i, p90 in enumerate(p90s):
        if p90:
            ax.plot(
                [p90, p90],
                [i - 0.28, i + 0.28],
                color=mode.text,
                linewidth=1.6,
                alpha=0.85,
                zorder=4,
            )
    if any(p90s):
        ax.plot([], [], color=mode.text, linewidth=1.6, label="p90")
        ax.legend(
            loc="upper right", frameon=False, fontsize=8, labelcolor=mode.text_dim
        )
    ax.axvline(300, color=mode.red, linestyle=":", linewidth=1.2)
    ax.annotate("300 ms gate", (300 * 1.03, -0.45), fontsize=8, color=mode.red)
    _title(ax, "Speech synthesis — time to first audio")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _footer_common(fig, rows)
    return fig


def tts_synth_box(bench_root: Path, rows: list[dict], mode: Mode = DARK):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    apply(fig, ax)
    data, colors, rows_used = [], [], []
    for r in rows:
        utts = _load_utterances(bench_root, r["run_id"], r["engine_id"])
        vals = [u["synth_ns"] / 1e9 for u in utts if u.get("synth_ns")]
        if not vals:
            continue
        data.append(vals)
        rows_used.append(r)
    if not data:
        return None
    best_i = min(
        range(len(data)), key=lambda i: sorted(data[i])[len(data[i]) // 2]
    )
    labels = [_label(r, star=i == best_i) for i, r in enumerate(rows_used)]
    bp = ax.boxplot(
        data, tick_labels=labels, vert=False, patch_artist=True, showfliers=False
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    for el in ("whiskers", "caps", "medians"):
        for line in bp[el]:
            line.set_color(mode.text_dim)
    ax.invert_yaxis()
    _title(ax, "Speech synthesis — synthesis latency distribution")
    ax.set_xlabel("synthesis latency per sentence (s, lower is better ↓, log)")
    ax.set_xscale("log")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _footer_common(fig, rows)
    return fig


def tts_memory(rows: list[dict], mode: Mode = DARK):
    return _memory_bars(rows, mode)


def tts_heatmap(rows: list[dict], mode: Mode = DARK):
    return _heatmap(rows, "tts", mode)


# -------------------------------------------------------------- shared
def _memory_bars(rows: list[dict], mode: Mode):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    apply(fig, ax)
    pts = [
        (r, _matrix._metric(r, "mem_peak_mib"))
        for r in rows
        if _matrix._metric(r, "mem_peak_mib")
    ]
    if not pts:
        return None
    pts.sort(key=lambda x: -x[1])
    best = _best_id([r for r, _ in pts], "mem_peak_mib")
    names = [_label(r, star=id(r) == best) for r, _ in pts]
    vals = [v for _, v in pts]
    colors = [row_color(r) for r, _ in pts]
    _hbar(
        ax,
        names,
        vals,
        colors,
        mode,
        "{:,.0f} MiB",
        "peak container memory (MiB, lower is better ↓)",
    )
    _title(ax, "Peak container memory")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _footer_common(fig, rows)
    return fig


def _heatmap(rows: list[dict], task: str, mode: Mode):
    """Configurations × metrics, cells colored by normalized score (spec chart 5)."""
    import numpy as np

    plt = _plt()
    # only metrics present in EVERY row (absent columns dropped, not rows)
    metric_keys = [
        h
        for h, _ in _matrix.METRICS[task]
        if all(_matrix._metric(r, h) is not None for r in rows)
    ]
    if len(metric_keys) < 3:
        return None
    lower_of = dict(_matrix.METRICS[task])
    raw_vals, labels = [], []
    for r in sorted(rows, key=lambda x: x.get("rank", 999)):
        raw_vals.append([_matrix._metric(r, k) for k in metric_keys])
        labels.append(_label(r, star=r.get("rank") == 1))  # aggregate view -> composite winner
    if len(raw_vals) < 2:
        return None
    grid = []
    for j, k in enumerate(metric_keys):
        lower = lower_of[k]
        col = [rv[j] for rv in raw_vals]
        if k in _matrix.LOG_METRICS:
            col = [max(v, 1e-9) for v in col]
            col = [__import__("math").log10(v) for v in col]
        grid.append(_matrix._norm(col, lower))
    g = np.array(grid).T  # rows=configs, cols=metrics
    # narrow cells + bigger fonts: the README displays the PNG at ~1/2 scale,
    # so an over-wide figure renders the labels unreadably small
    fig, ax = plt.subplots(
        figsize=(1.5 + 0.95 * len(metric_keys), 1.1 + 0.62 * len(labels))
    )
    apply(fig, ax)
    im = ax.imshow(g, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    pretty = {
        "wer": "WER",
        "cer": "CER",
        "ser": "SER",
        "hallucination_rate": "halluc.",
        "rtf": "RTF p50",
        "offline_latency_s": "lat p50 (s)",
        "synth_latency_s": "synth p50 (s)",
        "ttfa_s": "TTFA p50 (s)",
        "mem_peak_mib": "mem (MiB)",
        "gpu_power_w_mean": "GPU power (W)",
    }
    ax.set_xticks(
        range(len(metric_keys)),
        [pretty.get(k, k) for k in metric_keys],
        rotation=30,
        ha="right",
        fontsize=10,
    )
    ax.set_yticks(range(len(labels)), labels, fontsize=10.5)
    for i in range(len(labels)):
        for j in range(len(metric_keys)):
            raw = raw_vals[i][j]
            txt = (
                f"{raw:.3f}" if (isinstance(raw, float) and raw < 10) else f"{raw:,.0f}"
            )
            ax.text(j, i, txt, ha="center", va="center", fontsize=9.5, color=mode.text)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(colors=mode.text_dim, labelsize=8)
    cb.set_label("normalized score (1 = best)", color=mode.text_dim, fontsize=8)
    _title(ax, "Speech "
           + ("recognition" if task == "stt" else "synthesis")
           + " — normalized metric heatmap", mode)
    _footer_common(fig, rows, extra=" · green = better (direction-aware)")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


TASK_CHARTS = {
    "stt": {
        "stt-ranking": stt_ranking,
        "stt-wer-vs-rtf": stt_wer_vs_rtf,
        "stt-latency-box": stt_latency_box,
        "stt-memory": stt_memory,
        "stt-heatmap": stt_heatmap,
        "stt-power": stt_power,
    },
    "tts": {
        "tts-ranking": tts_ranking,
        "tts-ttfa": tts_ttfa,
        "tts-synth-box": tts_synth_box,
        "tts-memory": tts_memory,
        "tts-heatmap": tts_heatmap,
    },
}


def render_all(bench_root: Path, include: str = "baseline") -> dict:
    """Render every chart that has data; write manifest.json. Returns it."""
    plt = _plt()
    bench_root = Path(bench_root)
    out_dir = bench_root / "assets" / "benchmarks"
    rows = _matrix.select_rows(bench_root, include)
    manifest: dict[str, list[dict]] = {}
    keep = set()

    def save(name: str, task: str, build) -> bool:
        fig = build()
        if fig is None:
            return False
        d = out_dir / task
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{name}.png"
        fig.savefig(p, dpi=DPI, facecolor=fig.get_facecolor(),
                    bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)
        manifest.setdefault(name, []).append(
            {"task": task, "dark": str(p.relative_to(bench_root))}
        )
        keep.add(p.resolve())
        return True

    for task, charts in TASK_CHARTS.items():
        task_rows = [r for r in rows if (r.get("task") or cohort_task(r)) == task]
        if not task_rows:
            continue
        scored = _matrix.score_cohort([dict(r) for r in task_rows], task)
        for name, build in charts.items():
            if "bench_root" in build.__code__.co_varnames[: build.__code__.co_argcount]:
                save(name, task, lambda b=build: b(bench_root, scored))
            else:
                save(name, task, lambda b=build: b(scored))
    # prune stale renders
    for png in out_dir.rglob("*.png"):
        if png.resolve() not in keep:
            png.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    )
    print(f"[bench] {sum(len(v) for v in manifest.values())} charts -> {out_dir}")
    return manifest


def cohort_task(row: dict) -> str:
    return "stt" if "wer" in (row.get("metrics") or {}) else "tts"
