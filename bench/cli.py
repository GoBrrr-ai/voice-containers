"""CLI: prepare | selfcheck | run | report | compare (spec 9.3)."""

from __future__ import annotations
import json
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
    """Download, resample-cache and freeze a manifest (spec 3.4)."""
    from bench.datasets import prepare

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
    """Device / CUDA / probes / engines availability (spec 7A.5, 9.3)."""
    from .env import selfcheck

    r = selfcheck(BENCH)
    typer.echo(json.dumps(r, indent=2))


@app.command()
def run(config: str):
    """Execute one config through the spec 9.1 protocol."""
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


@app.command("grant-permissions")
def grant_permissions(yes: bool = typer.Option(False, "--yes", "-y")):
    """Install the NOPASSWD sudo grant for the page-cache clear (one prompt).

    Idempotent: a present grant (even one installed by another checkout) is
    detected and left untouched. Adds ONLY
    `sysctl -w vm.drop_caches=3` — never full sudo."""
    from . import cachectl

    raise typer.Exit(cachectl.grant_drop_caches(assume_yes=yes))


@app.command()
def prune(
    keep: int = typer.Option(1, "--keep", "-k", help="raw run dirs to keep per task"),
    dry_run: bool = typer.Option(True, "--dry-run/--yes"),
):
    """Delete superseded benchmarks/raw/<run_id> dirs (latest --keep per task).

    Retention policy: history.jsonl stays append-only (it is tiny); raw dirs
    are the bulk. Protected: every run_id referenced by a matrix row (latest
    complete per engine — e.g. the pocket row living in an older run dir),
    plus the newest `keep` dirs per task; everything else is superseded.
    """
    import shutil

    from .matrix import select_rows

    protected = {r.get("run_id") for r in select_rows(BENCH)}
    root = BENCH / "benchmarks" / "raw"
    if not root.exists():
        raise typer.Exit(0)
    by_task: dict[str, list[Path]] = {}
    for d in sorted(root.iterdir()):
        if d.is_dir() and "_" in d.name:
            task = d.name.rsplit("_", 1)[-1]
            by_task.setdefault(task, []).append(d)
    removed = 0
    for task, dirs in sorted(by_task.items()):
        newest = set(dirs[-keep:]) if keep else set()
        for d in dirs:
            if d.name in protected or d in newest:
                continue
            rel = d.relative_to(BENCH)
            if dry_run:
                typer.echo(f"would remove {rel} ({sum(1 for _ in d.rglob('*'))} files)")
            else:
                shutil.rmtree(d)
                typer.echo(f"removed {rel}")
            removed += 1
    typer.echo(
        f"{removed} dir(s) {'would be ' if dry_run else ''}removed (dry-run: rerun with --yes)"
    )


@app.command("drop-caches")
def drop_caches():
    """Clear the host page cache (sysctl -w vm.drop_caches=3) — never prompts."""
    from . import cachectl

    if not cachectl.drop_caches_granted():
        typer.secho(
            "drop-caches grant missing — run: bench grant-permissions",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(4)
    rc = cachectl.drop_caches()
    if rc == 0:
        typer.secho("page cache cleared (vm.drop_caches=3)", fg=typer.colors.GREEN)
    else:
        typer.secho(f"drop_caches failed (exit {rc})", fg=typer.colors.RED, err=True)
    raise typer.Exit(rc)


@app.command()
def plot(readme: bool = True, include: str = "baseline,gpu_variants"):
    """Render all charts (+ README.md with --readme) from benchmarks/history.jsonl.

    ``include``: comma-separated run_name substrings — the published matrix
    covers the baseline cohorts plus focused variant runs. Charts render
    dark-theme 2x PNGs into assets/benchmarks/ with a manifest; the README
    is rendered from README.md.j2 (StrictUndefined). Never re-runs inference.
    """
    from . import charts
    from . import readme as _readme

    charts.render_all(BENCH, include)
    if readme:
        p = _readme.render_readme(BENCH, include=include)
        typer.echo(f"README rendered: {p}")


def _write_html(path: Path, tables: dict, task: str) -> None:
    """Sortable self-contained HTML matrix (spec 10.1) with direction-aware coloring."""
    import html as _html

    parts = [
        "<!doctype html><meta charset=utf-8><title>edge-containers matrix — "
        f"{_html.escape(task)}</title>",
        "<style>body{font:14px/1.5 system-ui;background:#0d1117;color:#e6edf3}"
        "table{border-collapse:collapse}td,th{padding:4px 10px;border:1px solid #30363d}"
        "th{background:#161b22;cursor:pointer}.g{color:#3fb950}.r{color:#f85149}"
        "</style><h1>Results matrix — " + _html.escape(task) + "</h1>",
    ]
    for cohort, t in sorted(tables.items()):
        parts.append(
            f"<h3>{_html.escape(cohort[1])} · {cohort[2]} power · {_html.escape(cohort[3])}</h3>"
        )
        parts.append("<table id=t>")
        parts.append(
            "<tr>"
            + "".join(
                f"<th onclick=sortTable(this)>{_html.escape(h)}</th>"
                for h in t["heads"]
            )
            + "</tr>"
        )
        for row in t["body"]:
            cells = []
            for c in row:
                cls = ""
                if isinstance(c, str) and c.startswith("$\\color"):
                    cls = "g" if "#3fb950" in c else ("r" if "#f85149" in c else "")
                    c = c
                cells.append(f'<td class="{cls}">{_html.escape(str(c))}</td>')
            parts.append("<tr>" + "".join(cells) + "</tr>")
        parts.append("</table>")
    parts.append(
        "<script>function sortTable(th){const t=th.closest('table'),i=[...th.parentNode.children].indexOf(th),"
        "asc=th.asc=!th.asc;[...t.rows].slice(1).sort((a,b)=>{const x=a.cells[i].innerText,y=b.cells[i].innerText;"
        "const nx=parseFloat(x),ny=parseFloat(y);const c=(isNaN(nx)||isNaN(ny))?x.localeCompare(y):nx-ny;"
        "return asc?c:-c;}).forEach(r=>t.appendChild(r));}</script>"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))


@app.command()
def compare(run_dirs: list[str], out: str = "compare.md"):
    """Side-by-side headline table (full matrix is spec §10, fast-follow)."""
    from .runner import compare as run_compare

    typer.echo(Path(run_compare(run_dirs, out)).read_text())


if __name__ == "__main__":
    app()
