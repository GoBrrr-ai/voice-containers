"""Probe-artifact parsers (spec 7A): tegrastats + docker stats → aggregates.

Used by the matrix builder to enrich each engine row with measured resource
data without re-running anything.
"""

from __future__ import annotations
import re
import time
from pathlib import Path

_TS_TS = re.compile(r"^(\d{2})-(\d{2})-(\d{4}) (\d{2}):(\d{2}):(\d{2})")
_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
_VDD = re.compile(r"VDD_GPU (\d+)mW")
_GPU_UTIL = re.compile(r"GR3D (\d+)%")
_MEM_STR = {
    "KiB": 1 / 1024,
    "MiB": 1.0,
    "GiB": 1024.0,
    "kB": 1 / 1024,
    "MB": 1 / 1024 * 1000 / 1024,
    "GB": 1000 / 1024,
}


def _parse_mem_mib(s: str) -> float:
    """'1.23GiB' / '1234MiB' / '512MiB' -> MiB."""
    m = re.match(r"([0-9.]+)([A-Za-z]+)", s.strip())
    if not m:
        return 0.0
    return float(m.group(1)) * _MEM_STR.get(m.group(2), 1.0)


def parse_tegrastats(path: Path) -> list[dict]:
    """[(epoch_s, ram_used_mib, gpu_mw, gpu_util_pct), ...] — tolerant of absence."""
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text(errors="ignore").splitlines():
        m = _TS_TS.match(line)
        if not m:
            continue
        mon, day, year, hh, mm, ss = map(int, m.groups())
        try:
            epoch = time.mktime((year, mon, day, hh, mm, ss, 0, 0, -1))
        except (ValueError, OverflowError):
            continue
        ram = _RAM.search(line)
        vdd = _VDD.search(line)
        util = _GPU_UTIL.search(line)
        rows.append(
            {
                "epoch": epoch,
                "ram_mib": int(ram.group(1)) if ram else None,
                "gpu_mw": int(vdd.group(1)) if vdd else None,
                "gpu_util_pct": int(util.group(1)) if util else None,
            }
        )
    return rows


def window_aggregate(
    ts_rows: list[dict], start_epoch: float, dur_s: float
) -> dict | None:
    """Aggregate tegrastats rows inside [start, start+dur]; None if no samples."""
    pts = [
        r
        for r in ts_rows
        if start_epoch <= r["epoch"] <= start_epoch + max(dur_s, 1) + 2
    ]
    if not pts:
        return None

    def agg(key):
        vals = [r[key] for r in pts if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "n_samples": len(pts),
        "gpu_power_w_mean": (agg("gpu_mw") / 1000.0)
        if agg("gpu_mw") is not None
        else None,
        "gpu_util_pct_mean": agg("gpu_util_pct"),
        "ram_used_mib_mean": agg("ram_mib"),
    }


def parse_docker_stats(path: Path) -> dict | None:
    """Peak/p50 container memory MiB from a docker_stats_<eid>.csv; None if empty."""
    p = Path(path)
    if not p.exists():
        return None
    mems, anons = [], []
    for line in p.read_text().splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 3 and parts[2]:
            mems.append(_parse_mem_mib(parts[2]))
        if len(parts) >= 5 and parts[4]:
            anons.append(_parse_mem_mib(parts[4]))
    if not mems:
        return None
    mems.sort()
    n = len(mems)
    out = {
        "n_samples": n,
        "mem_peak_mib": mems[-1],
        "mem_p50_mib": mems[n // 2],
    }
    if anons:
        anons.sort()
        out["anon_peak_mib"] = anons[-1]
        out["anon_p50_mib"] = anons[len(anons) // 2]
    return out


def docker_image_size_gb(image: str) -> float | None:
    """Model/image size on disk via `docker images` (None if unknown)."""
    import subprocess

    try:
        out = subprocess.run(
            [
                "docker",
                "images",
                "--format",
                "{{.Repository}}:{{.Tag}} {{.Size}}",
                image,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except Exception:
        return None
    if not out:
        return None
    m = re.search(r"([0-9.]+)(GB|MB|kB|B)", out.splitlines()[0])
    if not m:
        return None
    v, unit = float(m.group(1)), m.group(2)
    return v / 1000.0 if unit == "MB" else v if unit == "GB" else v / 1000.0
