"""Chart generation from the benchmark history.

``render_all`` builds every chart that has data into
``assets/benchmarks/<task>/<name>.png`` (dark theme, 2x) and writes
``manifest.json``; ``readme.py`` resolves ``{{ chart("name") }}`` from it.
Charts needing per-utterance data read the raw run dirs under
``benchmarks/raw/`` (fallback: results/).

Label discipline (the 28-config roster made these non-negotiable):
- one engine can hold several rows (cuda + cpu twins) — every chart gets
  UNIQUE per-row labels via ``_labels``; ``ax.barh(<names>)`` treats names
  as a categorical axis and silently MERGES duplicates, overdrawing bars
  and stranding the value annotations of later rows below the axis;
- dense scatter clusters (wer-vs-rtf, power) label via ``_scatter_labels``
  (greedy bbox repel + leader lines) instead of fixed offsets, which
  turned into an unreadable pile once the cluster grew past ~8 engines;
- log-scale bar charts became lollipops: a bar on a log axis starts at an
  arbitrary axis minimum and reads as length anyway — stems + dots leave
  the label lane clear and don't lie about magnitude;
- figure heights scale with the row count so every label stays legible.
"""

from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

from . import matrix as _matrix
from . import names as _names
from . import theme
from .theme import DARK, Mode, apply, footer, row_color

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


def _labels(rows: list[dict], star_id=None) -> list[str]:
    """Unique per-chart series labels from the original engine + model.

    Rows sharing a display name (one engine, several backends) get an
    explicit "(GPU)"/"(CPU)" suffix — matching the ``paradee 8M int8
    (GPU)/(CPU)`` convention in ``names.display`` — so categorical axes
    keep every row distinct. ``star`` marks the best row FOR THE CHART
    BEING DRAWN — the metric shown decides (lower is better everywhere),
    never the composite rank.
    """
    base = [_names.display(r) for r in rows]
    counts = Counter(base)
    out: list[str] = []
    for r, b in zip(rows, base):
        if counts[b] > 1:
            b = f"{b} ({(r.get('requested_backend') or '?').upper()})"
        while b in out:  # same name AND backend — never expected; still unique
            b = f"{b} · {r.get('engine_id')}"
        out.append(b)
    if star_id is not None:
        out = [("★ " if id(r) == star_id else "") + b for r, b in zip(rows, out)]
    return out


def _star(labels: list[str], idx: int) -> list[str]:
    out = list(labels)
    out[idx] = "★ " + out[idx]
    return out


def _bold_stars(ax) -> None:
    """Bold the y-tick label of the starred (best) row."""
    for t in ax.get_yticklabels():
        if t.get_text().startswith("★"):
            t.set_fontweight("bold")


def _best_id(rows, key, lower=True):
    """id() of the row with the best (direction-aware) value of ``key``."""
    rows = [r for r in rows if _matrix._metric(r, key) is not None]
    if not rows:
        return None
    fn = min if lower else max
    return id(fn(rows, key=lambda r: _matrix._metric(r, key)))


def _title(ax, text: str, mode: Mode = DARK) -> None:
    """Consistent chart title: big, bold, left-aligned, light on dark."""
    ax.set_title(
        text, loc="left", fontsize=15, fontweight="bold", color=mode.text, pad=22
    )


def _backend_marker(row: dict) -> str:
    return theme.BACKEND_MARKERS.get((row.get("requested_backend") or "cuda"), "D")


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
    """Horizontal bar with in-axes value labels + footer-safe layout.

    ``names`` MUST be unique (see ``_labels``): the categorical y-axis maps
    duplicate names onto one slot, which silently drops bars.
    """
    assert len(set(names)) == len(names), "duplicate bar labels would merge rows"
    if log is None:
        log = _auto_log(values)
    ax.barh(names, values, color=colors, edgecolor=mode.grid)
    ax.invert_yaxis()
    _bold_stars(ax)
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
        _value_label(ax, x, i, fmt.format(v), mode)
    ax.set_xlabel(xlabel)


def _lollipop(
    ax,
    names,
    values,
    colors,
    mode: Mode,
    fmt: str,
    xlabel: str,
    label_x: list | None = None,
    ticks: list | None = None,
):
    """Stem + dot chart — the honest form for log-scale comparisons (a bar
    on a log axis starts at an arbitrary axis minimum; a stem shows position
    without pretending to encode length). Labels get a clear lane right of
    each dot, so nothing overlaps even on a 20-row roster."""
    assert len(set(names)) == len(names), "duplicate labels would merge rows"
    ax.set_xscale("log")
    lo = min(v for v in values if v > 0)
    hi = max(max(values), max(ticks or [0]))
    ax.set_xlim(lo * 0.7, hi * 1.55)
    y = list(range(len(names)))
    ax.hlines(y, lo * 0.7, values, color=colors, linewidth=2.4, zorder=2)
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
    pad = 0.055 * (__import__("math").log10(hi) - __import__("math").log10(lo))
    for i, v in enumerate(values):
        ref = max(v, (ticks or [0])[i] if ticks else 0)
        x = label_x[i] if label_x else ref * 10**pad
        _value_label(ax, x, i, fmt.format(v), mode)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    _bold_stars(ax)
    ax.set_xlabel(xlabel)


def _scatter_labels(fig, ax, pts, mode: Mode, fontsize: float = 8.2) -> None:
    """Greedy collision-free label placement for dense scatter clusters.

    ``pts``: [(x, y, text, color, prio, marker_size)] in data coords. Higher
    ``prio`` picks its spot first (front-runners keep the slot next to their
    marker). Candidates sweep expanding offset rings; each is scored by
    rendered-bbox overlap against the markers and already-placed labels and
    hard-rejected outside the axes. Winners far from their point get a thin
    leader line. Fixed offsets, the previous approach, produced a single
    unreadable pile once the RTF cluster grew past a handful of engines.
    """
    from math import cos, sin, pi
    from matplotlib.transforms import Bbox

    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ptp = fig.dpi / 72.0  # points -> pixels; offsets are in points, bboxes in px

    # marker obstacle bboxes and per-point pixel anchors
    anchors, obstacles = {}, []
    for i, (x, y, _t, _c, _p, s) in enumerate(pts):
        cx, cy = ax.transData.transform((x, y))
        r = (s**0.5) / 2 * ptp + 1.5
        anchors[i] = (cx, cy)
        obstacles.append(Bbox.from_bounds(cx - r, cy - r, 2 * r, 2 * r))
    axes_bb = ax.get_window_extent()

    probe = ax.text(0, 0, "", fontsize=fontsize)

    def text_wh(t: str) -> tuple[float, float]:
        probe.set_text(t)
        bb = probe.get_window_extent(rend)
        return bb.width + 5 * ptp, bb.height + 5 * ptp  # patch pad + safety

    placed: dict[int, Bbox] = {}
    anns: dict[int, object] = {}
    order = sorted(range(len(pts)), key=lambda i: -pts[i][4])

    def _overlap(b: Bbox, others) -> float:
        total = 0.0
        for ob in others:
            ix0, ix1 = max(b.x0, ob.x0), min(b.x1, ob.x1)
            iy0, iy1 = max(b.y0, ob.y0), min(b.y1, ob.y1)
            if ix1 > ix0 and iy1 > iy0:
                total += (ix1 - ix0) * (iy1 - iy0)
        return total

    def _scored(i: int, b: Bbox) -> float:
        # label-on-label is the worst failure — weight it over label-on-marker
        return 3.0 * _overlap(b, [bb for j, bb in placed.items() if j != i]) + _overlap(
            b, obstacles
        )

    def _plan(i: int, zero_only: bool = False):
        """Best (score, ring, dx, dy) for label i over the offset rings."""
        w, h = text_wh(pts[i][2])
        best = None
        for ring in (9, 17, 26, 37, 50, 65, 85, 105):
            for a in range(0, 360, 20):
                rad = a * pi / 180
                dx, dy = ring * cos(rad), ring * sin(rad)
                lx = anchors[i][0] + dx * ptp
                ly = anchors[i][1] + dy * ptp
                left = lx if dx >= 0 else lx - w
                bb = Bbox.from_bounds(left, ly - h / 2, w, h)
                if not (
                    axes_bb.x0 <= bb.x0
                    and bb.x1 <= axes_bb.x1
                    and axes_bb.y0 <= bb.y0
                    and bb.y1 <= axes_bb.y1
                ):
                    continue
                score = _scored(i, bb)
                if zero_only and score > 0:
                    continue
                if best is None or (score, ring) < best[:2]:
                    best = (score, ring, dx, dy)
        return best

    def _draw(i: int, dx: float, dy: float) -> None:
        x, y, txt, color, _prio, _s = pts[i]
        anns[i] = ax.annotate(
            txt,
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="left" if dx >= 0 else "right",
            va="center",
            fontsize=fontsize,
            color=color or mode.text,
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.18", fc=mode.bg, ec="none", alpha=0.55),
        )
        placed[i] = _planned_bbox(i, dx, dy)
        if (dx * dx + dy * dy) ** 0.5 >= 26:  # displaced far — leader line
            bb = placed[i]
            ex = bb.x0 if dx >= 0 else bb.x1
            ey = min(max(anchors[i][1], bb.y0), bb.y1)
            inv = ax.transData.inverted()
            (x0, y0), (x1, y1) = inv.transform(
                [(anchors[i][0], anchors[i][1]), (ex, ey)]
            )
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=mode.text_dim,
                linewidth=0.7,
                alpha=0.6,
                zorder=1,
                clip_on=False,
            )

    def _planned_bbox(i: int, dx: float, dy: float) -> Bbox:
        w, h = text_wh(pts[i][2])
        lx = anchors[i][0] + dx * ptp
        ly = anchors[i][1] + dy * ptp
        left = lx if dx >= 0 else lx - w
        return Bbox.from_bounds(left, ly - h / 2, w, h)

    for i in order:
        best = _plan(i)
        if best is None:  # fully boxed in — least-bad default
            best = (None, 9, 10, 12)
        _draw(i, best[2], best[3])

    # repair pass: re-place any label still touching another (lowest prio first)
    for _ in range(3):
        dirty = False
        for i in sorted(placed, key=lambda j: pts[j][4]):
            if _overlap(placed[i], [bb for j, bb in placed.items() if j != i]) <= 0:
                continue
            best = _plan(i, zero_only=True)
            if best:
                anns[i].remove()
                _draw(i, best[2], best[3])
                dirty = True
        if not dirty:
            break
    probe.remove()


# --------------------------------------------------------------------- STT
def stt_ranking(rows: list[dict], mode: Mode = DARK):
    rows = sorted(
        (r for r in rows if _matrix._metric(r, "wer") is not None),
        key=lambda r: _matrix._metric(r, "wer"),
    )
    plt = _plt()
    fig, ax = plt.subplots(figsize=_figsize(len(rows), 4.2, 0.27))
    apply(fig, ax)
    best = _best_id(rows, "wer")
    names = _labels(rows, star_id=best)
    wers = [_matrix._metric(r, "wer") * 100 for r in rows]
    colors = [row_color(r) for r in rows]
    _hbar(ax, names, wers, colors, mode, "{:.2f}%", "corpus WER % (lower is better ↓)")
    _title(ax, "Speech recognition — corpus WER")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _footer_common(fig, rows)
    return fig


def stt_wer_vs_rtf(rows: list[dict], mode: Mode = DARK):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
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
    w_lo, w_hi = min(p[1] for p in pts), max(p[1] for p in pts)
    ax.set_xscale("log")  # before any draw: transforms must match the final axes
    ax.set_xlim(x_lo * 0.5, x_hi * 3.2)
    ax.set_ylim(w_lo - (w_hi - w_lo) * 0.16 - 0.1, w_hi + (w_hi - w_lo) * 0.16 + 0.1)

    front = sorted(
        _matrix.score_cohort([dict(r) for r in rows], "stt"),
        key=lambda r: r["pareto"] is False,
    )
    pareto_ids = {r.get("engine_id") for r in front if r.get("pareto")}

    def _whisker(axes, wer, rtf, p90):
        if p90 and p90 > rtf:
            axes.plot(
                [rtf, p90],
                [wer, wer],
                color=mode.text_dim,
                linewidth=1.2,
                alpha=0.7,
                zorder=2,
            )

    for r, wer, rtf in pts:
        _whisker(ax, wer, rtf, _matrix._metric(r, "rtf_p90"))

    labels = _labels(rows)  # unique, backend-suffixed where needed
    lab = {id(r): t for r, t in zip(rows, labels)}
    scatter_pts = []
    for (r, wer, rtf), mem in zip(pts, mems):
        s = 60 + 240 * (mem / mmax)
        ax.scatter(
            rtf,
            wer,
            s=s,
            color=row_color(r),
            marker=_backend_marker(r),
            edgecolor=mode.text_dim,
            linewidth=0.6,
            zorder=3,
        )
        scatter_pts.append(
            (
                rtf,
                wer,
                lab[id(r)],
                mode.text,
                1 if r.get("engine_id") in pareto_ids else 0,
                s,
            )
        )
    _scatter_labels(fig, ax, scatter_pts, mode)

    # Pareto front (minimize WER then RTF) drawn as a step line
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
    _title(ax, "Speech recognition — accuracy vs speed")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _footer_common(
        fig, rows, extra=" · point size = memory · line = p50→p90 · □ GPU ○ CPU"
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
    best_i = min(range(len(data)), key=lambda i: sorted(data[i])[len(data[i]) // 2])
    labels = _star(_labels(rows_used), best_i)
    bp = ax.boxplot(
        data, tick_labels=labels, vert=False, patch_artist=True, showfliers=False
    )
    _bold_stars(ax)
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
    _footer_common(fig, rows, extra=" · green tick = p90")
    return fig


def stt_memory(rows: list[dict], mode: Mode = DARK):
    return _memory_bars(rows, mode)


def stt_heatmap(rows: list[dict], mode: Mode = DARK):
    return _heatmap(rows, "stt", mode)


def stt_power(rows: list[dict], mode: Mode = DARK):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    apply(fig, ax)
    pts = [
        (r, _matrix._metric(r, "gpu_power_w_mean"), _matrix._metric(r, "wer"))
        for r in rows
        if _matrix._metric(r, "gpu_power_w_mean")
    ]
    if len(pts) < 2:
        return None
    w_lo, w_hi = min(p[1] for p in pts), max(p[1] for p in pts)
    wers = [(p[2] or 0) * 100 for p in pts]
    ax.set_xlim(w_lo * 0.8, w_hi * 1.3)
    ax.set_ylim(min(wers) - 0.5, max(wers) + 0.5)
    labels = _labels(rows)
    lab = {id(r): t for r, t in zip(rows, labels)}
    spts = []
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
        spts.append(
            (w, (wer or 0) * 100, lab[id(r)], mode.text, 0 if is_cpu else 1, 90.0)
        )
    _scatter_labels(fig, ax, spts, mode)
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
    rows = sorted(
        (r for r in rows if _matrix._metric(r, "rtf") is not None),
        key=lambda r: _matrix._metric(r, "rtf"),
    )
    plt = _plt()
    fig, ax = plt.subplots(figsize=_figsize(len(rows), 4.4, 0.30))
    apply(fig, ax)
    best = _best_id(rows, "rtf")
    names = _labels(rows, star_id=best)
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
    best = _best_id([r for r, _ in pts], "ttfa_s")
    names = _labels([r for r, _ in pts], star_id=best)
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
    best_i = min(range(len(data)), key=lambda i: sorted(data[i])[len(data[i]) // 2])
    labels = _star(_labels(rows_used), best_i)
    bp = ax.boxplot(
        data, tick_labels=labels, vert=False, patch_artist=True, showfliers=False
    )
    _bold_stars(ax)
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
    _footer_common(fig, rows, extra=" · green tick = p90")
    return fig


def tts_memory(rows: list[dict], mode: Mode = DARK):
    return _memory_bars(rows, mode)


def tts_heatmap(rows: list[dict], mode: Mode = DARK):
    return _heatmap(rows, "tts", mode)


# -------------------------------------------------------------- shared
def _memory_bars(rows: list[dict], mode: Mode):
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
    best = _best_id([r for r, _ in pts], "mem_peak_mib")
    names = _labels([r for r, _ in pts], star_id=best)
    vals = [v for _, v in pts]
    colors = [row_color(r) for r, _ in pts]
    if _auto_log(vals):
        _lollipop(
            ax,
            names,
            vals,
            colors,
            mode,
            "{:,.0f} MiB",
            "peak container memory (MiB, lower is better ↓)",
        )
    else:
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
    ordered = sorted(rows, key=lambda x: x.get("rank", 999))
    best = next((r for r in ordered if r.get("rank") == 1), None)
    labels = _labels(ordered, star_id=id(best) if best else None)
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
    _bold_stars(ax)
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
