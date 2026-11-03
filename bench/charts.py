"""Chart generation from the benchmark history.

``render_all`` builds every chart that has data into
``assets/benchmarks/<task>/<name>.png`` (dark theme, 2x) and writes
``manifest.json``; ``readme.py`` resolves ``{{ chart("name") }}`` from it.
Charts needing per-utterance data read the raw run dirs under
``benchmarks/raw/`` (fallback: results/).

Label discipline (the 28-config roster made these non-negotiable):
- one engine can hold several rows (cuda + cpu twins) — every chart gets
  UNIQUE per-row labels via ``_labels``; categorical axes merge duplicate
  names, which silently drops bars and strands annotations;
- every chart marks its top-3 places with ❶❷❸ (gold/silver/bronze, bold)
  by the metric that chart ranks by — scatters and the heatmap use the
  composite rank from ``score_cohort``;
- comparisons run on lollipops (stems + dots), never bars: a bar on a
  log axis starts at an arbitrary axis minimum and reads as length
  anyway, and stems leave the value-label lane clear on any roster size;
- the two dense scatters label via a numbered rank legend column —
  direct labels (repelled or not) are unreadable at README half-scale
  once the cluster passes a handful of points;
- figure heights scale with the row count so every label stays legible.
"""

from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import matplotlib.patheffects as patheffects

from . import matrix as _matrix
from . import names as _names
from . import theme
from .theme import DARK, Mode, apply, footer, row_color

DPI = 200  # 2x PNG per spec 10.3 (vs matplotlib's default 100)

# top-3 places: dingbat medals (DejaVu Sans) + classic medal colors
_MEDAL = {1: "❶", 2: "❷", 3: "❸"}
MEDAL_COLORS = {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}  # gold silver bronze


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


def _labels(rows: list[dict], ranks: dict | None = None) -> list[str]:
    """Unique per-chart series labels; ``ranks`` maps id(row) -> 1|2|3 for
    the top-3 places of the chart's own ranking metric (medal-prefixed).

    Rows sharing a display name (one engine, several backends) get an
    explicit "(GPU)"/"(CPU)" suffix — matching the ``paradee 8M int8
    (GPU)/(CPU)`` convention in ``names.display`` — so categorical axes
    keep every row distinct.
    """
    ranks = ranks or {}
    base = [_names.display(r) for r in rows]
    counts = Counter(base)
    out: list[str] = []
    for r, b in zip(rows, base):
        if counts[b] > 1:
            b = f"{b} ({(r.get('requested_backend') or '?').upper()})"
        while b in out:  # same name AND backend — never expected; still unique
            b = f"{b} · {r.get('engine_id')}"
        rank = ranks.get(id(r))
        out.append(f"{_MEDAL[rank]} {b}" if rank else b)
    return out


def _top3(rows: list[dict], key: str, lower: bool = True) -> dict[int, int]:
    """{id(row): 1|2|3} by the chart's own metric (direction-aware)."""
    rs = sorted(
        (r for r in rows if _matrix._metric(r, key) is not None),
        key=lambda r: _matrix._metric(r, key),
        reverse=not lower,
    )
    return {id(r): i for i, r in enumerate(rs[:3], 1)}


def _style_medals(ax, labels: list[str]) -> None:
    """Bold + gold/silver/bronze the y-tick labels of the medal rows."""
    for t, lab in zip(ax.get_yticklabels(), labels):
        for rank, medal in _MEDAL.items():
            if lab.startswith(medal):
                t.set_fontweight("bold")
                t.set_color(MEDAL_COLORS[rank])


def _title(ax, text: str, mode: Mode = DARK) -> None:
    """Consistent chart title: big, bold, left-aligned, light on dark."""
    ax.set_title(
        text, loc="left", fontsize=15, fontweight="bold", color=mode.text, pad=22
    )


def _backend_marker(row: dict) -> str:
    return theme.BACKEND_MARKERS.get((row.get("requested_backend") or "cuda"), "D")


def _is_cpu(row: dict) -> bool:
    return (row.get("requested_backend") or "").lower() == "cpu"


def _pretty_device(label: str) -> str:
    """jetson-agx-thor-128gb -> Jetson AGX Thor 128GB."""
    import re as _re

    out = []
    for tok in label.split("-"):
        if tok.lower() in ("jetson",):
            out.append("Jetson")
        elif tok.lower() in ("agx", "nx"):
            out.append(tok.upper())
        elif m := _re.match(r"^(\d+)([a-z]+)$", tok):
            out.append(m.group(1) + m.group(2).upper())
        else:
            out.append(tok.capitalize())
    return " ".join(out)


def _pretty_dataset(row: dict) -> str:
    """Manifest filename -> 'LibriSpeech test-clean · 500 utts'."""
    import re as _re

    m = (row.get("manifest") or "").rsplit("/", 1)[-1].removesuffix(".jsonl")
    m = _re.sub(r"^(stt|tts)_", "", m).rsplit("_seed", 1)[0]
    parts = m.split("_")  # librispeech, test-clean, en, 500
    name = {"librispeech": "LibriSpeech", "libritts": "LibriTTS-R"}.get(
        parts[0], parts[0].capitalize()
    )
    split = parts[1] if len(parts) > 1 else ""
    task = row.get("task") or ("stt" if "wer" in (row.get("metrics") or {}) else "tts")
    n = (row.get("metrics") or {}).get("n")
    unit = "utts" if task == "stt" else "sents"
    out = f"{name} {split}" if split else name
    return f"{out} · {n:,} {unit}" if n else out


def _footer_common(fig, rows: list[dict], extra: str = "") -> None:
    r0 = rows[0]
    footer(
        fig,
        DARK,
        left=(
            f"edge-containers bench · {_pretty_device(r0.get('device_label', '?'))} · "
            f"{_pretty_dataset(r0)}"
        ),
        right=f"commit {r0.get('git_commit', '?')}{extra}",
    )


def _figsize(n_rows: int, base: float, per_row: float) -> tuple[float, float]:
    """Height that keeps every y-label legible as the roster grows."""
    return (9.5, max(base, 1.5 + per_row * n_rows))


def _auto_log(values: list[float]) -> bool:
    vals = [v for v in values if v and v > 0]
    return bool(vals) and max(vals) / max(min(vals), 1e-12) >= 10


def _text_on(hex_color: str, mode: Mode) -> str:
    """Readable text color on top of a filled marker of ``hex_color``."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return mode.bg if lum > 0.55 else mode.text


def _value_label(ax, x, y, txt, mode: Mode, ha: str = "left") -> None:
    ax.text(
        x,
        y,
        txt,
        va="center",
        ha=ha,
        fontsize=9,
        color=mode.text,
        bbox=dict(boxstyle="round,pad=0.15", fc=mode.panel, ec="none", alpha=0.7),
    )


def _lollipop(
    ax,
    names,
    values,
    colors,
    mode: Mode,
    fmt: str,
    xlabel: str,
    ticks: list | None = None,
    log: bool | None = None,
    value_labels: list | None = None,
):
    """Stem + dot chart — the one honest form for any roster size: on a log
    axis a bar starts at an arbitrary minimum and reads as length anyway,
    and stems leave the value-label lane clear. Medals (from ``names``)
    are bolded/colored on the y-tick labels."""
    assert len(set(names)) == len(names), "duplicate labels would merge rows"
    if log is None:
        log = _auto_log(values)
    y = list(range(len(names)))
    if log:
        ax.set_xscale("log")
        lo = min(v for v in values if v > 0)
        hi = max(max(values), max(ticks or [0]))
        ax.set_xlim(lo * 0.7, hi * 1.55)
        pad = 0.055 * (__import__("math").log10(hi) - __import__("math").log10(lo))
        label_x = [
            max(v, (ticks or [0])[i] if ticks else 0) * 10**pad
            for i, v in enumerate(values)
        ]
    else:
        hi = max(max(values), max(ticks or [0]))
        ax.set_xlim(0, hi * 1.22)
        label_x = [
            max(v, (ticks or [0])[i] if ticks else 0) + hi * 0.015
            for i, v in enumerate(values)
        ]
    ax.hlines(y, ax.get_xlim()[0], values, color=colors, linewidth=2.4, zorder=2)
    ax.scatter(
        values, y, s=72, color=colors, edgecolor=mode.grid, linewidth=0.8, zorder=3
    )
    for i, t in enumerate(ticks or []):
        if t:
            ax.plot(
                [t, t],
                [i - 0.28, i + 0.28],
                color=mode.text,
                linewidth=1.6,
                alpha=0.85,
                zorder=4,
            )
    for i, v in enumerate(values):
        txt = value_labels[i] if value_labels else fmt.format(v)
        _value_label(ax, label_x[i], i, txt, mode)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    _style_medals(ax, names)
    ax.set_xlabel(xlabel)


def _scatter_fig(n: int, mode: Mode):
    """Two-panel layout for the dense scatters: dot plot + ranked legend.

    Direct labels cannot survive an 18-point cluster — at README half-scale
    even repelled 8pt text is unreadable and leader lines mislead. Markers
    carry their rank number; the legend column holds every package name at
    full size, collision-free, whatever the roster grows to.
    """
    plt = _plt()
    fig, (ax, axl) = plt.subplots(
        1,
        2,
        figsize=(12.8, max(5.4, 1.2 + 0.30 * n)),
        gridspec_kw={"width_ratios": [2.55, 1.0], "wspace": 0.05},
    )
    apply(fig, ax)
    apply(fig, axl)
    axl.set_xticks([])
    axl.set_yticks([])
    axl.set_facecolor(mode.bg)
    for sp in axl.spines.values():
        sp.set_visible(False)
    return fig, ax, axl


def _legend_panel(axl, entries: list[tuple[dict, str]], mode: Mode) -> None:
    """Ranked legend column. ``entries``: (row, label) top-to-bottom by rank."""
    n = len(entries)
    axl.set_xlim(0, 1)
    axl.set_ylim(0, n + 0.6)
    axl.text(
        0.02,
        n + 0.15,
        "composite rank · ■ GPU ○ CPU",
        fontsize=8,
        color=mode.text_dim,
        va="center",
    )
    for i, (r, lab) in enumerate(entries):
        y = n - i - 0.5
        rank = i + 1
        axl.text(
            0.02,
            y,
            "○" if _is_cpu(r) else "■",
            color=row_color(r),
            fontsize=9,
            va="center",
            ha="left",
        )
        if rank <= 3:
            axl.text(
                0.065,
                y,
                f"{_MEDAL[rank]} {lab}",
                color=MEDAL_COLORS[rank],
                fontsize=8.8,
                fontweight="bold",
                va="center",
                ha="left",
            )
        else:
            axl.text(
                0.065,
                y,
                f"{rank}. {lab}",
                color=mode.text,
                fontsize=8.8,
                va="center",
                ha="left",
            )


def _rank_text(
    ax, x, y, rank: int, face: str | None, mode: Mode, fontsize: float = 6
) -> None:
    """Rank number centered on a marker; top-3 keep their medal colors
    (with a bg-colored stroke so gold/silver/bronze survive any fill)."""
    if rank < 1:
        return
    if rank <= 3:
        txt, color = _MEDAL[rank], MEDAL_COLORS[rank]
        effects = [patheffects.withStroke(linewidth=1.2, foreground=mode.bg)]
    else:
        txt = str(rank)
        color = mode.text if face is None else _text_on(face, mode)
        effects = None
    ax.text(
        x,
        y,
        txt,
        fontsize=fontsize,
        fontweight="bold",
        ha="center",
        va="center",
        color=color,
        zorder=4,
        path_effects=effects,
    )


def _add_lens(fig, ax, pts, mode: Mode, pareto=None, xlog: bool = False) -> None:
    """Magnifier lens over the densest marker cluster.

    Rank numbers stay readable even where markers pile up: the lens
    re-renders the crowded region at ~4x with uniform marker size and
    larger rank texts, wired to the source region by indicate_inset_zoom
    connectors. ``pts``: [(x, y, rank, face_or_None)]; ``pareto``:
    optional (xs, ys) step arrays drawn through the lens for context.
    """
    import numpy as np

    if len(pts) < 5:
        return
    fig.canvas.draw()
    disp = np.array([ax.transData.transform((x, y)) for x, y, _rk, _f in pts])
    d = np.sqrt(((disp[:, None, :] - disp[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    # density seed: the point with the most neighbors within 45 px — a
    # min-distance seed locks onto duplicate-position twins (cpu/cuda pairs
    # of one engine sit at nearly identical coords) far from the real crowd
    R = 45.0
    counts = (d <= R).sum(axis=1)
    seed = int(counts.argmax())
    if counts[seed] < 3:  # no meaningful crowd anywhere
        return
    axw = ax.get_window_extent().width
    sx, sy = disp[seed]
    half = 60.0
    for _ in range(4):  # widen just enough to hold the whole crowd
        inside = ((abs(disp[:, 0] - sx) <= half) & (abs(disp[:, 1] - sy) <= half)).sum()
        if inside >= counts[seed] or half >= 0.24 * axw:
            break
        half *= 1.3
    inv = ax.transData.inverted()
    dx0, dy0 = inv.transform((sx - half, sy - half))
    dx1, dy1 = inv.transform((sx + half, sy + half))
    xmin, xmax = min(dx0, dx1), max(dx0, dx1)
    ymin, ymax = min(dy0, dy1), max(dy0, dy1)
    src = (sx - half, sy - half, 2 * half, 2 * half)

    def _inter(a, b) -> float:
        ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
        return ix * iy

    lw, lh = 0.42, 0.46
    best = None  # (covered, src overlap, preference) -> position; a lens may
    # graze its source region when every free slot does — least overlap wins
    for pref, (px, py) in enumerate(
        (
            (0.52, 0.48),
            (0.06, 0.48),
            (0.52, 0.06),
            (0.06, 0.06),
            (0.29, 0.48),
            (0.29, 0.06),
        )
    ):
        c0 = ax.transAxes.transform((px, py))
        c1 = ax.transAxes.transform((px + lw, py + lh))
        crect = (c0[0], c0[1], c1[0] - c0[0], c1[1] - c0[1])
        covered = int(
            (
                (disp[:, 0] >= crect[0])
                & (disp[:, 0] <= crect[0] + crect[2])
                & (disp[:, 1] >= crect[1])
                & (disp[:, 1] <= crect[1] + crect[3])
            ).sum()
        )
        key = (covered, _inter(crect, src), pref)
        if best is None or key < best[0]:
            best = (key, (px, py))
    if best is None:
        return

    axins = ax.inset_axes([best[1][0], best[1][1], lw, lh])
    apply(fig, axins)
    axins.set_facecolor("#202a36")  # lighter than the panel behind it
    if xlog:
        axins.set_xscale("log")
    axins.set_xlim(xmin, xmax)
    axins.set_ylim(ymin, ymax)
    axins.set_xticks([])
    axins.set_yticks([])
    # log axes grow minor tick labels ("2 × 10⁻¹") inside the lens — kill all
    axins.tick_params(
        which="both", bottom=False, left=False, labelbottom=False, labelleft=False
    )
    if pareto:
        axins.step(
            pareto[0],
            pareto[1],
            where="post",
            color=mode.green,
            linewidth=1.4,
            linestyle="--",
            alpha=0.8,
            zorder=2,
        )
    for x, y, rank, face in pts:
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            continue
        cpu = face is None
        axins.scatter(
            x,
            y,
            s=70,
            facecolor="none" if cpu else face,
            edgecolor=face if cpu else mode.text_dim,
            linewidth=1.4 if cpu else 0.6,
            marker="o" if cpu else "s",
            zorder=3,
        )
        _rank_text(axins, x, y, rank, face, mode, fontsize=7.5)
    ax.indicate_inset_zoom(axins, edgecolor=mode.text_dim, alpha=0.6)


# --------------------------------------------------------------------- STT
def stt_ranking(rows: list[dict], mode: Mode = DARK):
    rows = sorted(
        (r for r in rows if _matrix._metric(r, "wer") is not None),
        key=lambda r: _matrix._metric(r, "wer"),
    )
    plt = _plt()
    fig, ax = plt.subplots(figsize=_figsize(len(rows), 4.2, 0.27))
    apply(fig, ax)
    names = _labels(rows, ranks=_top3(rows, "wer"))
    wers = [_matrix._metric(r, "wer") * 100 for r in rows]
    colors = [row_color(r) for r in rows]
    _lollipop(
        ax, names, wers, colors, mode, "{:.2f}%", "corpus WER % (lower is better ↓)"
    )
    _title(ax, "Speech recognition — corpus WER")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _footer_common(fig, rows)
    return fig


def stt_wer_vs_rtf(rows: list[dict], mode: Mode = DARK):
    pts = []
    for r in rows:
        wer, rtf = _matrix._metric(r, "wer"), _matrix._metric(r, "rtf")
        if wer is None or rtf is None:
            continue
        pts.append((r, wer * 100, rtf))
    if not pts:
        return None
    fig, ax, axl = _scatter_fig(len(pts), mode)
    ordered = sorted(pts, key=lambda p: p[0].get("rank") or 999)
    leg_rows = [r for r, _, _ in ordered]
    _legend_panel(axl, list(zip(leg_rows, _labels(leg_rows))), mode)

    mems = [(_matrix._metric(r, "mem_peak_mib") or 0) for r, _, _ in pts]
    mmax = max(mems) or 1
    x_lo, x_hi = min(p[2] for p in pts), max(p[2] for p in pts)
    w_lo, w_hi = min(p[1] for p in pts), max(p[1] for p in pts)
    ax.set_xscale("log")
    ax.set_xlim(x_lo * 0.5, x_hi * 3.2)
    ax.set_ylim(w_lo - (w_hi - w_lo) * 0.16 - 0.1, w_hi + (w_hi - w_lo) * 0.16 + 0.1)

    def _whisker(wer, rtf, p90):
        if p90 and p90 > rtf:
            ax.plot(
                [rtf, p90],
                [wer, wer],
                color=mode.text_dim,
                linewidth=1.2,
                alpha=0.7,
                zorder=2,
            )

    for r, wer, rtf in pts:
        _whisker(wer, rtf, _matrix._metric(r, "rtf_p90"))
    lens_pts = []
    for (r, wer, rtf), mem in zip(pts, mems):
        face = None if _is_cpu(r) else row_color(r)
        ax.scatter(
            rtf,
            wer,
            s=60 + 240 * (mem / mmax),
            facecolor="none" if _is_cpu(r) else row_color(r),
            marker=_backend_marker(r),
            edgecolor=mode.text_dim if not _is_cpu(r) else row_color(r),
            linewidth=0.6 if not _is_cpu(r) else 1.4,
            zorder=3,
        )
        _rank_text(ax, rtf, wer, r.get("rank") or 0, face, mode)
        lens_pts.append((rtf, wer, r.get("rank") or 0, face))

    # Pareto front (minimize WER then RTF) drawn as a step line
    front = sorted(
        _matrix.score_cohort([dict(r) for r in rows], "stt"),
        key=lambda r: r["pareto"] is False,
    )
    pf = sorted(
        [r for r in front if r.get("pareto")],
        key=lambda r: _matrix._metric(r, "rtf") or 0,
    )
    pf_xy = None
    if len(pf) > 1:
        pf_x = [_matrix._metric(r, "rtf") for r in pf]
        pf_y = [_matrix._metric(r, "wer") * 100 for r in pf]
        ax.step(
            pf_x,
            pf_y,
            where="post",
            color=mode.green,
            linewidth=1.4,
            linestyle="--",
            alpha=0.8,
            zorder=2,
        )
        pf_xy = (pf_x, pf_y)
        ax.annotate(
            "Pareto front",
            (0.99, 0.03),
            xycoords="axes fraction",
            ha="right",
            fontsize=8.5,
            color=mode.green,
        )
    _add_lens(fig, ax, lens_pts, mode, pareto=pf_xy, xlog=True)
    ax.set_xlabel("RTF p50 — audio-s per s (lower is better ↓, log)")
    ax.set_ylabel("corpus WER % (lower is better ↓)")
    _title(ax, "Speech recognition — accuracy vs speed")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _footer_common(
        fig, rows, extra=" · marker = rank (❶❷❸ top-3) · size = memory · line = p50→p90"
    )
    return fig


def stt_latency_box(bench_root: Path, rows: list[dict], mode: Mode = DARK):
    plt = _plt()
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
        colors.append(row_color(r))
    if not data:
        return None
    fig, ax = plt.subplots(figsize=_figsize(len(data), 4.6, 0.31))
    apply(fig, ax)
    medals = {}
    for pos, i in enumerate(
        sorted(range(len(data)), key=lambda i: sorted(data[i])[len(data[i]) // 2])[:3],
        1,
    ):
        medals[id(rows_used[i])] = pos
    labels = _labels(rows_used, ranks=medals)
    bp = ax.boxplot(
        data, tick_labels=labels, vert=False, patch_artist=True, showfliers=False
    )
    _style_medals(ax, labels)
    for i, vals_i in enumerate(data):
        p90 = sorted(vals_i)[min(len(vals_i) - 1, int(len(vals_i) * 0.9))]
        ax.plot(
            [p90, p90], [i + 0.75, i + 1.25], color=mode.green, linewidth=1.8, zorder=4
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
    _footer_common(fig, rows, extra=" · green tick = p90 · ❶❷❸ = best median RTF")
    return fig


def stt_memory(rows: list[dict], mode: Mode = DARK):
    return _memory_chart(rows, mode)


def stt_heatmap(rows: list[dict], mode: Mode = DARK):
    return _heatmap(rows, "stt", mode)


def stt_power(rows: list[dict], mode: Mode = DARK):
    pts = [
        (r, _matrix._metric(r, "gpu_power_w_mean"), _matrix._metric(r, "wer"))
        for r in rows
        if _matrix._metric(r, "gpu_power_w_mean")
    ]
    if len(pts) < 2:
        return None
    fig, ax, axl = _scatter_fig(len(pts), mode)
    ordered = sorted(pts, key=lambda p: p[0].get("rank") or 999)
    leg_rows = [r for r, _, _ in ordered]
    _legend_panel(axl, list(zip(leg_rows, _labels(leg_rows))), mode)

    w_lo, w_hi = min(p[1] for p in pts), max(p[1] for p in pts)
    wers = [(p[2] or 0) * 100 for p in pts]
    ax.set_xlim(w_lo * 0.8, w_hi * 1.3)
    ax.set_ylim(min(wers) - 0.5, max(wers) + 0.5)
    lens_pts = []
    for r, w, wer in pts:
        face = None if _is_cpu(r) else row_color(r)
        ax.scatter(
            w,
            (wer or 0) * 100,
            s=90,
            facecolor=face,
            marker=_backend_marker(r),
            edgecolor=row_color(r),
            linewidth=1.6 if _is_cpu(r) else 0.6,
            zorder=3,
        )
        _rank_text(ax, w, (wer or 0) * 100, r.get("rank") or 0, face, mode)
        lens_pts.append((w, (wer or 0) * 100, r.get("rank") or 0, face))
    _add_lens(fig, ax, lens_pts, mode)
    ax.set_xlabel(
        "mean GPU power over the engine window (W) — hollow = CPU engine (idle draw)"
    )
    ax.set_ylabel("corpus WER % (lower is better ↓)")
    _title(ax, "Speech recognition — GPU power vs accuracy")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _footer_common(
        fig,
        rows,
        extra=" · marker = rank (❶❷❸ top-3) · power from tegrastats job windows",
    )
    return fig


# --------------------------------------------------------------------- TTS
def tts_ranking(rows: list[dict], mode: Mode = DARK):
    rows = sorted(
        (r for r in rows if _matrix._metric(r, "rtf") is not None),
        key=lambda r: _matrix._metric(r, "rtf"),
    )
    plt = _plt()
    fig, ax = plt.subplots(figsize=_figsize(len(rows), 4.4, 0.30))
    apply(fig, ax)
    names = _labels(rows, ranks=_top3(rows, "rtf"))
    rtfs = [_matrix._metric(r, "rtf") for r in rows]
    p90s = [_matrix._metric(r, "rtf_p90") or 0 for r in rows]
    colors = [row_color(r) for r in rows]
    _lollipop(
        ax,
        names,
        rtfs,
        colors,
        mode,
        "{:.3f}",
        "RTF p50 — s audio per s wall (lower is better ↓, tick = p90)",
        ticks=p90s,
    )
    if any(p90s):
        ax.plot([], [], color=mode.text, linewidth=1.6, label="p90")
        ax.legend(
            loc="upper right", frameon=False, fontsize=8, labelcolor=mode.text_dim
        )
    _title(ax, "Speech synthesis — real-time factor")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _footer_common(fig, rows)
    return fig


def tts_ttfa(rows: list[dict], mode: Mode = DARK):
    plt = _plt()
    pts = [
        (r, _matrix._metric(r, "ttfa_s")) for r in rows if _matrix._metric(r, "ttfa_s")
    ]
    if not pts:
        return None
    pts.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=_figsize(len(pts), 4.0, 0.30))
    apply(fig, ax)
    names = _labels([r for r, _ in pts], ranks=_top3([r for r, _ in pts], "ttfa_s"))
    vals = [v * 1000 for _, v in pts]
    colors = [row_color(r) for r, _ in pts]
    p90s = [(_matrix._metric(r, "ttfa_s_p90") or 0) * 1000 for r, _ in pts]
    _lollipop(
        ax,
        names,
        vals,
        colors,
        mode,
        "{:.0f} ms",
        "TTFA (ms, lower is better ↓)",
        ticks=p90s,
    )
    if any(p90s):
        ax.plot([], [], color=mode.text, linewidth=1.6, label="p90")
        ax.legend(
            loc="upper right", frameon=False, fontsize=8, labelcolor=mode.text_dim
        )
    ax.axvline(300, color=mode.red, linestyle=":", linewidth=1.2)
    ax.annotate(
        "300 ms gate", (300 * 1.06, len(vals) - 1.1), fontsize=8, color=mode.red
    )
    _title(ax, "Speech synthesis — time to first audio")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _footer_common(fig, rows)
    return fig


def tts_synth_box(bench_root: Path, rows: list[dict], mode: Mode = DARK):
    plt = _plt()
    data, colors, rows_used = [], [], []
    for r in rows:
        utts = _load_utterances(bench_root, r["run_id"], r["engine_id"])
        vals = [u["synth_ns"] / 1e9 for u in utts if u.get("synth_ns")]
        if not vals:
            continue
        data.append(vals)
        rows_used.append(r)
        colors.append(row_color(r))
    if not data:
        return None
    fig, ax = plt.subplots(figsize=_figsize(len(data), 4.2, 0.31))
    apply(fig, ax)
    medals = {}
    for pos, i in enumerate(
        sorted(range(len(data)), key=lambda i: sorted(data[i])[len(data[i]) // 2])[:3],
        1,
    ):
        medals[id(rows_used[i])] = pos
    labels = _labels(rows_used, ranks=medals)
    bp = ax.boxplot(
        data, tick_labels=labels, vert=False, patch_artist=True, showfliers=False
    )
    _style_medals(ax, labels)
    for i, vals_i in enumerate(data):
        p90 = sorted(vals_i)[min(len(vals_i) - 1, int(len(vals_i) * 0.9))]
        ax.plot(
            [p90, p90], [i + 0.75, i + 1.25], color=mode.green, linewidth=1.8, zorder=4
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
    _footer_common(fig, rows, extra=" · green tick = p90 · ❶❷❸ = best median latency")
    return fig


def tts_memory(rows: list[dict], mode: Mode = DARK):
    return _memory_chart(rows, mode)


def tts_heatmap(rows: list[dict], mode: Mode = DARK):
    return _heatmap(rows, "tts", mode)


# -------------------------------------------------------------- shared
# -------------------------------------------------------------- shared
# Edge deployment profiles — VERSIONED. Changing anchors, weights, gates or
# methodology means a NEW profile name (orin-nano-8gb-v2), never a silent
# edit: a 98-point engine must stay a meaningful 98 six months and twenty
# engines later.
#
# Vector tuple: (target, worst, weight). Score is ABSOLUTE, not
# cohort-relative — 1.0 at/beyond the target, 0.0 at the worst-acceptable
# bound, log-linear in between — so adding engines never rewrites history
# and the anchors are deployment facts (HA's 300 ms gate, the Orin Nano
# 8 GB voice budget), not chart artifacts. target > worst marks a
# higher-is-better metric (e.g. MOS).
#
# Latency/RTF vectors bind the TAIL (p95; TTFA p90 is the only percentile
# recorded) — averages hide the multi-second spikes that make an
# interactive voice assistant feel broken. Memory is the measured peak.
#
# Coverage: weights renormalize over the vectors a row MEASURES, and the
# chart reports coverage next to the score — an unmeasured vector makes a
# score provisional, it is never silently treated as irrelevant. The TTS
# quality vector (MOS) is pending a listening test: until then every TTS
# score is a deployment-PERFORMANCE score at 75% coverage.
#
# Hard gates are separate from the score: violating one means UNDEPLOYABLE
# for the target regardless of the aggregate (a 99-quality engine that
# does not fit in memory must not outrank one that fits). Passing engines
# rank by score; failures sink below them, marked ✗.
#
# The synth-latency vector was dropped from TTS: it is RTF re-measured per
# sentence (same measurement, same bottleneck) and double-counted
# throughput. RTF + TTFA + (pending) MOS capture different aspects.
EDGE_PROFILES = {
    "orin-nano-8gb-v1": {
        "stt": {
            "vectors": {
                # quality: corpus WER; ≤2% production grade, ≥10% unusable
                "wer": (0.02, 0.10, 0.35),
                # responsiveness, p95 tail: instant follow-ups … lost user.
                # Conservative final-latency proxy: true streaming engines'
                # final latency is their chunk lookahead (160 ms), so this
                # over-penalizes them until a streaming-latency probe lands.
                "offline_latency_s_p95": (0.3, 1.5, 0.20),
                # throughput headroom, p95 tail: 10× realtime … barely realtime
                "rtf_p95": (0.10, 1.0, 0.20),
                # VRAM: fits the Orin Nano 8 GB voice budget … half the board
                "mem_peak_mib": (1024, 4096, 0.25),
            },
            "gates": {  # hard upper bounds (lower-is-better metrics)
                "wer": 0.10,
                "offline_latency_s_p95": 1.5,
                "mem_peak_mib": 4096,
            },
        },
        "tts": {
            "vectors": {
                # naturalness: MOS ≥4.0 good, ≤2.5 unusable — PENDING a
                # listening test (drives the 75% coverage on TTS scores)
                "tts_mos": (4.0, 2.5, 0.25),
                # responsiveness on the tail (p90 = only percentile recorded)
                "ttfa_s_p90": (0.3, 1.5, 0.25),
                # sustained generation efficiency, p95 tail
                "rtf_p95": (0.10, 1.0, 0.25),
                "mem_peak_mib": (1024, 4096, 0.25),
            },
            "gates": {
                "ttfa_s_p90": 1.5,
                "mem_peak_mib": 4096,
            },
        },
    },
}
EDGE_PROFILE = "orin-nano-8gb-v1"


def _edge_vector_score(v: float, target: float, worst: float) -> float:
    """Log-linear anchor score in [0, 1]; direction from target/worst order."""
    import math

    if target > worst:  # higher is better (MOS)
        if v >= target:
            return 1.0
        if v <= worst:
            return 0.0
        return math.log(v / worst) / math.log(target / worst)
    if v <= target:
        return 1.0
    if v >= worst:
        return 0.0
    return math.log(worst / v) / math.log(worst / target)


def _edge_eval(row: dict, task: str) -> tuple[float, float, list[str]]:
    """(score 0-100, coverage 0-1, failed gate descriptions)."""
    prof = EDGE_PROFILES[EDGE_PROFILE][task]
    total = wsum = total_weight = 0.0
    for metric, (target, worst, weight) in prof["vectors"].items():
        total_weight += weight
        v = _matrix._metric(row, metric)
        if v is None:
            continue
        wsum += weight
        total += weight * _edge_vector_score(v, target, worst)
    score = 100 * total / wsum if wsum else None
    failed = []
    for metric, limit in prof["gates"].items():
        v = _matrix._metric(row, metric)
        if v is not None and v > limit:
            failed.append(f"{metric}={v:.4g}>{limit:g}")
    return score, (wsum / total_weight if total_weight else 0.0), failed


def _edge_chart(rows: list[dict], task: str, mode: Mode):
    evals = []
    for r in rows:
        score, cov, failed = _edge_eval(r, task)
        if score is not None:
            evals.append((r, score, cov, failed))
    if len(evals) < 2:
        return None
    # passing engines rank by score; gate failures sink below, marked ✗
    evals.sort(key=lambda e: (bool(e[3]), -e[1]))
    plt = _plt()
    fig, ax = plt.subplots(figsize=_figsize(len(evals), 4.2, 0.27))
    apply(fig, ax)
    passing = [e for e in evals if not e[3]]
    ranks = {id(e[0]): i for i, e in enumerate(passing[:3], 1)}
    labels_raw = _labels([e[0] for e in evals], ranks=ranks)
    names = [("✗ " if e[3] else "") + b for e, b in zip(evals, labels_raw)]
    vals = [e[1] for e in evals]
    colors = [row_color(e[0]) for e in evals]
    value_labels = [
        f"{s:.0f}" + ("" if cov > 0.999 else f" · {cov:.0%}") for _, s, cov, _f in evals
    ]
    if task == "stt":
        xlabel = (
            "edge deployment score (0–100; WER 35% · latency p95 20% · RTF p95 "
            "20% · VRAM 25% — targets ≤2% / ≤0.3 s / ≤0.1 / ≤1 GiB)"
        )
        title = "Speech recognition — edge deployment score"
        footer = f" · profile {EDGE_PROFILE} · ✗ = hard-gate FAIL"
    else:
        xlabel = (
            "edge deployment-performance score (0–100; TTFA p90 25% · RTF p95 "
            "25% · VRAM 25% · quality 25% pending MOS — targets ≤0.3 s / "
            "≤0.1 / ≤1 GiB)"
        )
        title = "Speech synthesis — edge deployment-performance score"
        footer = f" · profile {EDGE_PROFILE} · ✗ = hard-gate FAIL · MOS pending"
    _lollipop(
        ax,
        names,
        vals,
        colors,
        mode,
        "{:.0f}",
        xlabel,
        log=False,
        value_labels=value_labels,
    )
    for t, lab in zip(ax.get_yticklabels(), names):
        if lab.startswith("✗"):
            t.set_color(mode.red)
            t.set_fontweight("bold")
        else:
            for rank, medal in _MEDAL.items():
                if lab.startswith(medal):
                    t.set_fontweight("bold")
                    t.set_color(MEDAL_COLORS[rank])
    _title(ax, title)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _footer_common(fig, rows, extra=footer)
    return fig


def stt_edge(rows: list[dict], mode: Mode = DARK):
    return _edge_chart(rows, "stt", mode)


def tts_edge(rows: list[dict], mode: Mode = DARK):
    return _edge_chart(rows, "tts", mode)


def _memory_chart(rows: list[dict], mode: Mode):
    pts = [
        (r, _matrix._metric(r, "mem_peak_mib"))
        for r in rows
        if _matrix._metric(r, "mem_peak_mib")
    ]
    if not pts:
        return None
    pts.sort(key=lambda x: -x[1])
    plt = _plt()
    fig, ax = plt.subplots(figsize=_figsize(len(pts), 4.2, 0.27))
    apply(fig, ax)
    names = _labels(
        [r for r, _ in pts], ranks=_top3([r for r, _ in pts], "mem_peak_mib")
    )
    vals = [v for _, v in pts]
    colors = [row_color(r) for r, _ in pts]
    _lollipop(
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
    ordered = sorted(rows, key=lambda x: x.get("rank", 999))
    ranks = {id(r): r["rank"] for r in ordered if r.get("rank") and r["rank"] <= 3}
    labels = _labels(ordered, ranks=ranks)
    raw_vals = [[_matrix._metric(r, k) for k in metric_keys] for r in ordered]
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
    # stt's 8 columns made the figure over-wide (the README displays the PNG
    # at ~1/2 scale), so its cells are compact with bigger fonts; tts keeps
    # the original wide layout
    geom = (
        dict(w=1.5, cw=0.95, h=1.1, ch=0.62, tick=10, ylab=10.5, cell=9.5, cb=8)
        if task == "stt"
        else dict(w=1.9, cw=1.55, h=1.2, ch=0.55, tick=8.5, ylab=8.5, cell=7.5, cb=7)
    )
    fig, ax = plt.subplots(
        figsize=(
            geom["w"] + geom["cw"] * len(metric_keys),
            geom["h"] + geom["ch"] * len(labels),
        )
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
        fontsize=geom["tick"],
    )
    ax.set_yticks(range(len(labels)), labels, fontsize=geom["ylab"])
    _style_medals(ax, labels)
    for i in range(len(labels)):
        for j in range(len(metric_keys)):
            raw = raw_vals[i][j]
            txt = (
                f"{raw:.3f}" if (isinstance(raw, float) and raw < 10) else f"{raw:,.0f}"
            )
            # pick the text color from the cell's own luminance — white on the
            # pale yellow/green mid tones was unreadable
            rgba = im.cmap(im.norm(g[i][j]))  # g: configs × metrics (transposed)
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=geom["cell"],
                color=mode.bg if lum > 0.55 else mode.text,
            )
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(colors=mode.text_dim, labelsize=geom["cb"])
    cb.set_label("normalized score (1 = best)", color=mode.text_dim, fontsize=8)
    _title(
        ax,
        "Speech "
        + ("recognition" if task == "stt" else "synthesis")
        + " — normalized metric heatmap",
        mode,
    )
    _footer_common(
        fig, rows, extra=" · green = better (direction-aware) · ❶❷❸ = top-3 composite"
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


TASK_CHARTS = {
    "stt": {
        "stt-ranking": stt_ranking,
        "stt-edge": stt_edge,
        "stt-wer-vs-rtf": stt_wer_vs_rtf,
        "stt-latency-box": stt_latency_box,
        "stt-memory": stt_memory,
        "stt-heatmap": stt_heatmap,
        "stt-power": stt_power,
    },
    "tts": {
        "tts-ranking": tts_ranking,
        "tts-edge": tts_edge,
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
        fig.savefig(
            p,
            dpi=DPI,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            pad_inches=0.2,
        )
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
