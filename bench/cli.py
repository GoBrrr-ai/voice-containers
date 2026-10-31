"""CLI: prepare | selfcheck | run | report | compare (PRD 9.3)."""

from __future__ import annotations
import json
import sys
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, help="edge-containers benchmark harness")
BENCH = Path(__file__).resolve().parent.parent  # repo root


@app.command()
def prepare(
    dataset: str = "librispeech",
    split: str = "test-clean",
    lang: str = "en",
    n: int = 500,
    seed: int = 42,
    task: str = "stt",
    cache_dir: str = str(BENCH / "datasets" / "cache"),
    hf_config: str = "en_us",
    folder: str = "",
):
    """Download, resample-cache and freeze a manifest (PRD 3.4)."""
    from datasets import prepare

    out = prepare(
        dataset,
        split,
        lang,
        n,
        seed,
        cache_dir,
        str(BENCH / "manifests"),
        task,
        hf_config=hf_config,
        folder=folder,
    )
    typer.echo(f"manifest: {out}")


@app.command()
def selfcheck():
    """Device / CUDA / probes / engines availability (PRD 7A.5, 9.3)."""
    from .env import selfcheck

    r = selfcheck(BENCH)
    typer.echo(json.dumps(r, indent=2))


@app.command()
def run(config: str):
    """Execute one config through the PRD 9.1 protocol."""
    from .runner import run_config

    rid = run_config(config)
    typer.echo(f"run_id: {rid}")


@app.command()
def report(run_dir: str):
    """Recompute summary.json/csv for a finished run."""
    from .runner import write_summary

    s = write_summary(
        Path(run_dir), json.loads((Path(run_dir) / "summary.json").read_text())["task"]
    )
    typer.echo(json.dumps(s, indent=2))


@app.command()
def compare(run_dirs: list[str], out: str = "compare.md"):
    """Side-by-side headline table (full matrix is PRD §10, fast-follow)."""
    from .runner import compare

    typer.echo(Path(compare(run_dirs, out)).read_text())


if __name__ == "__main__":
    app()
