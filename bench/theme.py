"""Chart design system — one visual language, dark mode (mirrors mjolnir's).

Static charts render the dark palette into ``assets/benchmarks/`` as 2x PNG.
Color = identity, assigned per ENGINE FAMILY (never by rank): all parakeet.cpp
models share one hue, marker shape encodes backend (circle = CPU, square =
CUDA), variant by direct label. Okabe-Ito hues — colorblind-safe on the dark
surface.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Mode:
    name: str
    bg: str
    panel: str
    text: str
    text_dim: str
    grid: str
    green: str  # better
    red: str  # worse


DARK = Mode(
    name="dark",
    bg="#0d1117",
    panel="#161b22",
    text="#e6edf3",
    text_dim="#8b949e",
    grid="#30363d",
    green="#3fb950",
    red="#f85149",
)

# fixed identity hues per engine family (Okabe-Ito)
ENGINE_COLORS = {
    "parakeet_cpp": "#56B4E9",  # sky blue — the parakeet.cpp family
    "faster_whisper": "#E69F00",  # orange — the CT2 baseline
    "kokoro": "#009E73",  # green
    "ptts": "#CC79A7",  # pink
    "paradee": "#F0E442",  # yellow
    "piper": "#D55E00",  # vermillion
}
FALLBACK_COLOR = "#0072B2"  # blue — unknown engines

BACKEND_MARKERS = {"cuda": "s", "cpu": "o"}


def engine_color(engine: str) -> str:
    return ENGINE_COLORS.get(engine, FALLBACK_COLOR)


def row_color(row: dict) -> str:
    """Consistent identity color: per engine FAMILY (brand-aware), same in
    every chart; variant distinguished by marker shape + label."""
    from . import names as _names

    fam = _names.family(row)
    return _names.FAMILY_COLORS.get(fam) or ENGINE_COLORS.get(
        row.get("engine", ""), FALLBACK_COLOR
    )


def _plt():
    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib missing — install bench requirements (pip install -r bench/requirements.txt)"
        ) from e
    plt.rcParams.update(
        {"font.size": 11, "font.family": "DejaVu Sans", "savefig.dpi": 200}
    )
    return plt


def apply(fig, ax, mode: Mode = DARK) -> None:
    fig.patch.set_facecolor(mode.bg)
    ax.set_facecolor(mode.panel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(mode.grid)
    ax.tick_params(colors=mode.text_dim, labelcolor=mode.text)
    ax.xaxis.label.set_color(mode.text)
    ax.yaxis.label.set_color(mode.text)
    # titles are usually set AFTER apply() — color the Text artist unconditionally
    # (set_title() changes text/position but not color, so this persists)
    ax.title.set_color(mode.text)
    ax.grid(False)


def footer(fig, mode: Mode = DARK, left: str = "", right: str = "") -> None:
    fig.text(0.01, 0.012, left, fontsize=7.5, color=mode.text_dim, ha="left")
    fig.text(0.99, 0.012, right, fontsize=7.5, color=mode.text_dim, ha="right")


def log_y_if_spanning(ax, values, min_span: float = 10.0) -> bool:
    """Log scale when the range spans > min_span (spec 10.3). Returns applied."""
    vals = [v for v in values if v and v > 0]
    if not vals:
        return False
    lo, hi = min(vals), max(vals)
    if hi / max(lo, 1e-12) >= min_span:
        ax.set_yscale("log")
        return True
    return False
