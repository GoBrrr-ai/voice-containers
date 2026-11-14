"""Chart design system — one visual language, adaptive surfaces.

Every chart is a SINGLE self-adaptive SVG: theme-dependent colors are
emitted as CSS custom properties (``var(--token)``); the file carries both
palettes — ``:root`` light, an embedded ``prefers-color-scheme`` query
dark — so the same asset renders correctly on GitHub's light and dark
modes with no picture pair. Colors = identity, per ENGINE FAMILY (never
by rank) via ``TOKENS``; theme-constant hues (Okabe-Ito) stay literal.
Marker shape encodes backend (circle = CPU, square = CUDA).
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

# GitHub primer light — the README's light-mode surface
LIGHT = Mode(
    name="light",
    bg="#ffffff",
    panel="#f6f8fa",
    text="#1f2328",
    text_dim="#57606a",
    grid="#d0d7de",
    green="#1a7f37",
    red="#cf222e",
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


def row_color(row: dict) -> str:
    """Consistent identity color: per MODEL (brand-aware), same in every
    chart; the same model in two variants keeps one hue (marker + label)."""
    from . import names as _names

    return _names.color(row)


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
        {
            "font.size": 11,
            "font.family": "DejaVu Sans",
            "savefig.dpi": 200,
            # text as <text>, not glyph paths: ~5x smaller files, selectable
            # and searchable in the browser; viewers without DejaVu Sans
            # substitute their default sans (positions are baked, so only
            # glyph shapes vary by a hair)
            "svg.fonttype": "none",
            # pin the element-id salt: ids become sha(salt+content) instead
            # of per-render uuid4 -> byte-stable re-renders, quiet git diffs
            "svg.hashsalt": "voice-containers",
        }
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


def span_wants_log(values, min_span: float = 10.0) -> bool:
    """True when the positive range spans >= min_span (spec 10.3) — the single
    log-axis policy; callers apply it to whichever axis they own."""
    vals = [v for v in values if v and v > 0]
    if not vals:
        return False
    lo, hi = min(vals), max(vals)
    return hi / max(lo, 1e-12) >= min_span


# color tokens for the self-adaptive svg: (light, dark) per role. Every
# theme-derived hex the renderer emits maps to ``var(--token)``; the svg
# carries :root = dark — the default, and what theme-less viewers (plain
# readers, RSS) get — with a prefers-color-scheme:light override. Theme
# constants: engine identity hues stay literal.
TOKENS = {
    "bg": ("#ffffff", "#0d1117"),
    "panel": ("#f6f8fa", "#161b22"),
    "text": ("#1f2328", "#e6edf3"),
    "dim": ("#57606a", "#8b949e"),
    "grid": ("#d0d7de", "#30363d"),
    "green": ("#1a7f37", "#3fb950"),
    "red": ("#cf222e", "#f85149"),
    "m1": ("#b8860b", "#ffd700"),  # medals, darkened for the light surface
    "m2": ("#6e7781", "#c0c0c0"),
    "m3": ("#a0522d", "#cd7f32"),
    "lens": ("#dce3ea", "#202a36"),  # magnifier inset, contrast step vs panel
}


def token_map() -> dict[str, str]:
    """hex -> css var, both palettes (the baked render may be either)."""
    out: dict[str, str] = {}
    for name, (light, dark) in TOKENS.items():
        out[light.lower()] = f"var(--{name})"
        out[dark.lower()] = f"var(--{name})"
    return out


def token_css() -> str:
    """The <style> block injected into every chart svg. Dark is the default
    (:root); light applies via an explicit prefers-color-scheme query."""
    dark = ";".join(f"--{n}:{d}" for n, (_l, d) in TOKENS.items())
    light = ";".join(f"--{n}:{lt}" for n, (lt, _d) in TOKENS.items())
    return f":root{{{dark}}}@media (prefers-color-scheme:light){{:root{{" + light + "}}"
