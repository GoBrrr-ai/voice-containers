"""Runner — orchestrates per-(engine, backend, dataset) container jobs (PRD §2, §5, §9).

Each job runs in a FRESH container (process isolation); the parent hosts the
background probes (tegrastats + docker stats) and writes env.json, run.json,
probes.parquet and the aggregated summary.
"""

from __future__ import annotations
import json
import subprocess
import threading
import time
from pathlib import Path

from .common import jsonl_read
from .metrics.wer import corpus_wer_cer, dist_stats


class TegraStats:
    """tegrastats --interval 100 parser; on non-Jetson hosts it's just absent."""

    def __init__(self, out_path: Path):
        self.out = open(out_path, "w")
        self.proc = None
        self.stop = False

    def start(self):
        try:
            self.proc = subprocess.Popen(
                ["tegrastats", "--interval", "100"],
                stdout=self.out,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            self.proc = None

    def stop(self):
        if self.proc:
            self.proc.terminate()
        self.out.close()


def docker_stats_sampler(
    container: str, out_path: Path, stop_evt: threading.Event, interval: float = 1.0
):
    """Sample `docker stats --no-stream` for the child container (RSS, CPU%)."""
    import csv, io

    with open(out_path, "w") as f:
        f.write("t_ns,cpu_perc,mem_usage,mem_limit\n")
        t0 = time.perf_counter_ns()
        while not stop_evt.is_set():
            try:
                r = subprocess.run(
                    [
                        "docker",
                        "stats",
                        "--no-stream",
                        "--format",
                        "{{.CPUPerc}},{{.MemUsage}}",
                        container,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
                if "," in line:
                    cpu, mem = line.split(",", 1)
                    mem_used = mem.split("/")[0].strip()
                    f.write(f"{time.perf_counter_ns() - t0},{cpu},{mem_used},\n")
                    f.flush()
            except Exception:
                pass
            stop_evt.wait(interval)


def run_container_job(
    engine_img: str,
    job: dict,
    run_dir: Path,
    bench_dir: Path,
    cache_dir: str,
    gpu: bool = True,
) -> dict:
    """Spawn the engine container with the client mounted; wait; return meta."""
    name = f"bench-{job['task']}-{job['engine']}-{int(time.time())}"
    # rewrite host cache paths to their container mount (/data/bench-cache);
    # work on a copy — run_config reuses the same job dict for every engine
    import copy as _copy
    job = _copy.deepcopy(job)
    cache_root = Path(cache_dir).resolve()
    for s in job["samples"]:
        if s.get("audio_path"):
            s["audio_path"] = "/data/bench-cache/" + str(
                Path(s["audio_path"]).resolve().relative_to(cache_root)
            )
    job_path = run_dir / f"job_{job['engine']}.json"
    job_path.write_text(
        json.dumps({**job, "out_dir": "/bench/out"}, ensure_ascii=False)
    )
    cmd = ["docker", "run", "--rm", "--name", name]
    if gpu:
        cmd += ["--runtime", "nvidia", "-e", "NVIDIA_DISABLE_REQUIRE=1"]
    cmd += [
        "-v",
        f"{run_dir}:/bench/out",
        "-v",
        f"{str(bench_dir)}:/bench/code:ro",
        "-v",
        f"{cache_root}:/data/bench-cache:ro",
        "--entrypoint",
        "python3",
        engine_img,
        "/bench/code/client.py",
        f"/bench/out/job_{job['engine']}.json",
    ]
    t0 = time.perf_counter_ns()
    proc = subprocess.Popen(cmd)
    stop_evt = threading.Event()
    stats_t = threading.Thread(
        target=docker_stats_sampler,
        args=(name, run_dir / f"docker_stats_{job['engine']}.csv", stop_evt),
    )
    stats_t.start()
    proc.wait()
    stop_evt.set()
    stats_t.join()
    return {
        "engine": job["engine"],
        "container": name,
        "exit": proc.returncode,
        "wall_ns": time.perf_counter_ns() - t0,
    }


def run_config(config_path: str) -> str:
    """Full PRD §9.1 protocol for one config. Returns run_id."""
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text())
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{cfg.get('device_label', 'device')}_{cfg['task']}"
    out = Path(cfg.get("output_dir", "results")) / run_id
    out.mkdir(parents=True, exist_ok=True)
    Path(out / "audio").mkdir(exist_ok=True)

    # 1) idle baseline is handled by the probes thread (tegrastats runs throughout)
    teg = TegraStats(out / "tegrastats.log")
    teg.start()

    manifest = list(jsonl_read(cfg["dataset"]["manifest"]))
    manifest = manifest[: int(cfg["dataset"].get("n", len(manifest)))]
    task = cfg["task"]
    for eng in cfg["engines"]:
        job = {
            "task": task,
            "engine": eng["name"],
            "opts": eng.get("opts", {}) | {"device": eng.get("backends", ["cuda"])[0]},
            "samples": manifest,
            "out_dir": str(out / "audio"),
            "streaming": True,
        }
        img = eng["image"]  # engine image for this adapter
        meta = run_container_job(
            img,
            job,
            out,
            Path(__file__).parent / "engines",
            cfg.get("cache_dir", "bench/cache"),
        )
        (out / f"container_{eng['name']}.json").write_text(json.dumps(meta))
    teg.stop()

    # aggregate
    write_summary(out, task)
    return run_id


def write_summary(run_dir: Path, task: str) -> dict:
    """Aggregate utterances.jsonl → summary.json + summary.csv (PRD §8.3)."""
    rows = list(jsonl_read(run_dir / "utterances.jsonl"))
    ok = [r for r in rows if r.get("status") == "ok"]
    summary: dict = {"task": task, "n": len(rows), "n_ok": len(ok)}
    if task == "stt" and ok:
        s = corpus_wer_cer(
            (r["ref_norm"] for r in ok),
            ((r.get("hyp_norm") or r.get("hyp_raw", "")).lower().strip() for r in ok),
        )
        summary.update(s)
        summary.update(
            dist_stats(
                [r["proc_ns"] / 1e9 / max(r.get("audio_s") or 1, 1e-9) for r in ok],
                "rtf",
            )
        )
        summary.update(
            dist_stats([r["proc_ns"] / 1e9 for r in ok], "offline_latency_s")
        )
    if task == "tts" and ok:
        if all(r.get("ttfa_ns") for r in ok):
            summary.update(dist_stats([r["ttfa_ns"] / 1e9 for r in ok], "ttfa_s"))
        summary.update(
            dist_stats(
                [
                    r["synth_ns"]
                    / 1e9
                    / max(r.get("audio_s") or (r.get("num_chars", 1) / 14), 1e-9)
                    for r in ok
                ],
                "rtf",
            )
        )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    # csv: one row per utterance (flat)
    import csv

    keys = sorted({k for r in rows for k in r})
    with open(run_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    return summary


def compare(run_dirs: list[str], out: str) -> str:
    """Side-by-side headline table (PRD 8.4; the full §10 matrix is fast-follow)."""
    lines = ["| run | task | n_ok | headline |", "|---|---|---|---|"]
    for rd in run_dirs:
        p = Path(rd) / "summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        head = (
            f"WER={s['wer']:.3f}"
            if s.get("wer") is not None
            else f"RTF_p50={s.get('rtf')}"
            if s.get("rtf")
            else "-"
        )
        lines.append(f"| {Path(rd).name} | {s['task']} | {s['n_ok']} | {head} |")
    Path(out).write_text("\n".join(lines) + "\n")
    return out
