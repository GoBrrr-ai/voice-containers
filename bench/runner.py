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
    eid = job["engine_id"]
    name = f"bench-{job['task']}-{eid}-{int(time.time())}"
    # container-side paths; work on a copy — run_config reuses the same job dict
    import copy as _copy

    job = _copy.deepcopy(job)
    cache_root = Path(cache_dir).resolve()
    run_dir = Path(run_dir).resolve()
    bench_dir = Path(bench_dir).resolve()
    for s in job["samples"]:
        if s.get("audio_path"):
            s["audio_path"] = "/data/bench-cache/" + str(
                Path(s["audio_path"]).resolve().relative_to(cache_root)
            )
    job_path = run_dir / f"job_{eid}.json"
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
        f"{bench_dir}:/bench/code:ro",
        "-v",
        f"{cache_root}:/data/bench-cache:ro",
        # writable HF cache so model downloads persist across fresh containers
        "-v",
        f"{cache_root / 'hf'}:/data/hf-cache",
        "--entrypoint",
        "python3",
        engine_img,
        "/bench/code/client.py",
        f"/bench/out/job_{eid}.json",
    ]
    t0 = time.perf_counter_ns()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    stop_evt = threading.Event()
    stats_t = threading.Thread(
        target=docker_stats_sampler,
        args=(name, run_dir / f"docker_stats_{eid}.csv", stop_evt),
    )
    stats_t.start()
    out_log = proc.stdout.read() if proc.stdout else ""
    proc.wait()
    stop_evt.set()
    stats_t.join()
    # keep the container's combined output next to the run for post-mortem
    (run_dir / f"logs_{eid}.txt").write_text(out_log)
    meta = {
        "engine": job["engine"],
        "engine_id": eid,
        "container": name,
        "exit": proc.returncode,
        "wall_ns": time.perf_counter_ns() - t0,
    }
    if proc.returncode != 0:
        meta["tail"] = out_log.strip().splitlines()[-5:]
        print(
            f"[runner] WARNING: engine {eid} container exited {proc.returncode}; "
            f"see logs_{eid}.txt in the run dir"
        )
    return meta


def run_config(config_path: str) -> str:
    """Full PRD §9.1 protocol for one config. Returns run_id."""
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text())
    root = (
        Path(config_path).resolve().parent.parent
    )  # bench/configs/*.yaml -> repo root
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{cfg.get('device_label', 'device')}_{cfg['task']}"
    out = Path(root / cfg.get("output_dir", "results")) / run_id
    out.mkdir(parents=True, exist_ok=True)
    Path(out / "audio").mkdir(exist_ok=True)

    # 1) idle baseline is handled by the probes thread (tegrastats runs throughout)
    teg = TegraStats(out / "tegrastats.log")
    teg.start()

    manifest = list(jsonl_read(root / cfg["dataset"]["manifest"]))
    manifest = manifest[: int(cfg["dataset"].get("n", len(manifest)))]
    task = cfg["task"]
    for i, eng in enumerate(cfg["engines"]):
        # unique id per engine entry (the same adapter may appear with different models)
        eid = eng.get("id") or f"{eng['name']}_{i}"
        job = {
            "task": task,
            "engine": eng["name"],
            "engine_id": eid,
            "opts": eng.get("opts", {}) | {"device": eng.get("backends", ["cuda"])[0]},
            "samples": manifest,
            "streaming": bool(cfg.get("streaming", {}).get("realtime_pass", False)),
        }
        img = eng["image"]  # engine image for this adapter
        meta = run_container_job(
            img,
            job,
            out,
            Path(__file__).parent / "engines",
            root / cfg.get("cache_dir", "bench/cache"),
        )
        (out / f"container_{eid}.json").write_text(json.dumps(meta))
    teg.stop()

    # aggregate
    write_summary(out, task)
    return run_id


def _engine_summary(rows: list[dict], task: str) -> dict:
    """Aggregate one engine's utterance rows (PRD 8.3). Mutates rows with hyp_norm."""
    ok = [r for r in rows if r.get("status") == "ok"]
    s: dict = {"n": len(rows), "n_ok": len(ok)}
    if task == "stt" and ok:
        from .metrics.normalize import normalize

        for r in ok:  # host-side normalization (containers lack the normalizer)
            r["hyp_norm"] = normalize(r.get("hyp_raw", ""), r.get("language", "en"))
        s.update(
            corpus_wer_cer((r["ref_norm"] for r in ok), (r["hyp_norm"] for r in ok))
        )
        s.update(
            dist_stats(
                [r["proc_ns"] / 1e9 / max(r.get("audio_s") or 1, 1e-9) for r in ok],
                "rtf",
            )
        )
        s.update(dist_stats([r["proc_ns"] / 1e9 for r in ok], "offline_latency_s"))
    if task == "tts" and ok:
        if all(r.get("ttfa_ns") for r in ok):
            s.update(dist_stats([r["ttfa_ns"] / 1e9 for r in ok], "ttfa_s"))
        # RTF against the produced audio's duration (PRD 6.2)
        s.update(
            dist_stats(
                [
                    r["synth_ns"] / 1e9 / max(r.get("out_audio_s") or 1e-9, 1e-9)
                    for r in ok
                ],
                "rtf",
            )
        )
    return s


def write_summary(run_dir: Path, task: str) -> dict:
    """Aggregate utterances.jsonl → summary.json + summary.csv (PRD §8.3).

    Headline metrics are computed PER ENGINE (per_engine.engine_id); the top
    level is corpus-wide and only meaningful for single-engine runs.
    """
    rows = list(jsonl_read(run_dir / "utterances.jsonl"))
    ok = [r for r in rows if r.get("status") == "ok"]
    summary: dict = {"task": task, "n": len(rows), "n_ok": len(ok)}
    summary["per_engine"] = {
        eid: _engine_summary(
            [r for r in rows if r.get("engine_id", r.get("engine")) == eid], task
        )
        for eid in sorted({str(r.get("engine_id", r.get("engine", "?"))) for r in rows})
    }
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
