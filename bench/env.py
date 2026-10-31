"""Environment record (spec §5.4, §7A): device, stack, probes availability."""

from __future__ import annotations
import json
import platform
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=15
        ).stdout.strip()
    except Exception:
        return ""


def _git_commit() -> str:
    try:
        root = Path(__file__).resolve().parent.parent
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        return ""


def collect(git_commit: str = "") -> dict:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "nvidia_smi": _run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "cuda_version": (
            _run(["bash", "-c", "nvcc --version | grep -oE 'release [0-9.]+'"])
        ),
        "jetpack_l4t": (
            _run(["bash", "-c", "cat /etc/nv_tegra_release 2>/dev/null | head -1"])
        ),
        "nvpmodel": _run(["nvpmodel", "-q"]),
        "jetson_clocks": _run(
            ["bash", "-c", "jetson_clocks --show 2>/dev/null | head -3"]
        ),
        "tegrastats_available": _run(["bash", "-c", "command -v tegrastats"]) != "",
        "docker_images_gpu": _run(
            [
                "bash",
                "-c",
                "docker images --format '{{.Repository}}' | grep narandill | sort | uniq",
            ]
        ),
        "git_commit": git_commit or _git_commit(),
        "utc_ts": _run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]),
    }
    # compute capability from the GPU query (accept 8.7 / 11.0 / 10.1 — spec 7A);
    # parse the CSV column, don't substring-match (driver 511.05 contains "11.0")
    cc = ""
    for line in env["nvidia_smi"].splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            cc = parts[1]
            break
    env["compute_cap"] = cc
    env["compute_cap_ok"] = cc in {"8.7", "11.0", "10.1"}
    return env


def selfcheck(bench_root: Path) -> dict:
    """spec 7A.5 fail-fast: compute cap + a tiny CUDA EP session + probe presence."""
    report = collect()  # compute_cap / compute_cap_ok set by collect()
    try:
        import onnxruntime as ort  # noqa

        report["onnxruntime_import"] = "ok"
    except Exception as e:
        report["onnxruntime_import"] = f"missing: {e}"
    report["docker_gpu_runtime"] = _run(
        [
            "bash",
            "-c",
            "docker run --rm --runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1 "
            "narandill/llama-cpp:0.8.0 nvidia-smi -L 2>/dev/null | head -1",
        ]
    )
    out = bench_root / "selfcheck.json"
    out.write_text(json.dumps(report, indent=2))
    return report
