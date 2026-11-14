"""Full-render smoke: synthetic bench tree -> every registered chart + README.

Exercises the entire pure path — history parse, cohort selection, scoring,
all chart builders, SVG render, manifest, README render — with zero
GPU/docker/network. This is the regression net for "README.md is GENERATED".
"""

from __future__ import annotations

import json
import os
from pathlib import Path


from bench import charts
from bench.charts import TASK_CHARTS

os.environ.setdefault("MPLBACKEND", "Agg")


def test_render_all_produces_every_chart(bench_env):
    manifest = charts.render_all(bench_env.root, "baseline")
    expected = {name for task in TASK_CHARTS for name in TASK_CHARTS[task]}
    for name in expected:
        entry = manifest[name]
        assert set(entry["variants"]) == {"dark", "light"}, name
        for rel in entry["variants"].values():
            p = bench_env.root / rel
            assert p.exists() and p.stat().st_size > 1000, name


def test_render_all_include_filter_can_exclude_everything(bench_env):
    # no run matches this substring -> empty manifest, no crash
    manifest = charts.render_all(bench_env.root, "no-such-run")
    assert manifest == {}


def test_chart_rebuild_matches_render_all_rows(bench_env):
    """The per-chart builders consume scored rows exactly as render_all feeds them."""
    from bench import matrix as _m

    rows = _m.select_rows(bench_env.root, "baseline")
    stt = _m.score_cohort([dict(r) for r in rows if r["task"] == "stt"], "stt")
    # superseded incomplete run must be gone; cpu twin survives as its own row
    ids = [r["engine_id"] for r in stt]
    assert ids.count("parakeet_cpp_0") == 1
    assert "parakeet_cpp_1" in ids
    fig = charts.stt_ranking(stt)
    assert fig is not None


def test_light_mode_renders(bench_env):
    from bench import theme as _theme
    from bench import matrix as _m

    light = _theme.LIGHT

    rows = _m.select_rows(bench_env.root, "baseline")
    stt = _m.score_cohort([dict(r) for r in rows if r["task"] == "stt"], "stt")
    tts = _m.score_cohort([dict(r) for r in rows if r["task"] == "tts"], "tts")
    for build, rows_ in (
        (charts.stt_edge, stt),
        (charts.tts_ranking, tts),
        (charts.stt_latency_box, stt),
    ):
        fig = (
            build(bench_env.root, rows_, mode=light)
            if "bench_root" in build.__code__.co_varnames[: build.__code__.co_argcount]
            else build(rows_, mode=light)
        )
        assert fig is not None


def test_empty_rows_return_none(bench_env):
    # box charts guard empty data; ranking/edge rely on render_all's empty
    # task skip (covered by the include-filter test)
    from bench import matrix as _m

    rows = _m.select_rows(bench_env.root, "baseline")
    stt = _m.score_cohort([dict(r) for r in rows if r["task"] == "stt"], "stt")
    assert charts.stt_latency_box(bench_env.root, stt) is not None
    # no dist stats anywhere -> boxes vanish -> None
    (bench_env.root / "benchmarks" / "raw" / "run_a" / "diststats.json").unlink()
    assert charts.stt_latency_box(bench_env.root, stt) is None


def test_render_readme_from_synthetic_tree(bench_env, tmp_path: Path):
    """README renders from the tmp tree with the real template."""
    template = Path(__file__).resolve().parents[1] / "README.md.j2"
    charts.render_all(bench_env.root, "baseline")
    out = bench_env.root / "README.md"
    from bench import readme as _readme

    p = _readme.render_readme(bench_env.root, template=template, out=out)
    text = p.read_text()
    assert "## 🐋 Features" in text
    assert "parakeet-tdt-cpp-stt" in text  # packages table links
    assert "stt-ranking.dark.svg" in text  # chart embeds (picture pair)
    assert "prefers-color-scheme" in text
    # jinja StrictUndefined: any missing ctx key would raise above


def test_readme_json_ctx_payload(bench_env):
    """The CI html+json contract: ctx values are plain JSON-able scalars/str."""
    from bench import readme as _readme

    ctx = _readme.collect_context(bench_env.root, "baseline")
    json.dumps(ctx)  # raises on non-serializable ctx members
