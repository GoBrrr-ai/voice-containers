"""Runner — orchestrates per-(engine, backend, dataset) container jobs (spec §2, §5, §9).

Each job runs in a FRESH container (process isolation); the parent hosts the
background probes (tegrastats + docker stats) and writes env.json (device info,
power profile + the full resolved config), per-engine container metas + logs,
and the aggregated summary.json / summary.csv.
"""

from __future__ import annotations
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from .common import jsonl_read
from .metrics.wer import corpus_wer_cer, dist_stats

# Config keys understood by this runner. Unknown keys are REJECTED loudly so a
# methodology knob can never be a silent no-op (spec 9.2); unsupported values
# are rejected with an explicit message instead of being ignored.
KNOWN_KEYS = {
    "run_name",
    "task",
    "device_label",
    "power_profile",
    "dataset",
    "normalizer",
    "cache_dir",
    "engines",
    "streaming",
    "warmup_n",
    "repeats",
    "cooldown_s",
    "soak_minutes",
    "probes",
    "publish_raw",
    "drop_caches",
    "output_dir",
    "gpu_free_threshold_mib",
    "gpu_wait_minutes",
    "gpu_util_threshold_pct",
    "gpu_confirm_samples",
}


def _fail(msg: str) -> None:
    raise SystemExit(f"config error: {msg}")


def _gpu_used_mib() -> int:
    """Total memory held by compute apps (0 when nvidia-smi is absent)."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
        return sum(int(x) for x in out.split() if x.strip().isdigit())
    except Exception:
        return 0


def _gpu_busy_pct() -> int:
    """GPU utilization percent (0 when nvidia-smi is absent).

    Memory alone is not enough: a co-tenant holding GBs of KV cache but
    idling passes a memory-only gate, then bursts with continuous batching
    and corrupts RTF/TTFA mid-run. Utilization catches the bursts.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
        return max(int(x) for x in out.split() if x.strip().isdigit())
    except Exception:
        return 0


def _wait_for_free_gpu(
    threshold_mib: int,
    wait_minutes: float,
    confirm: int = 6,
    poll_s: float = 2.0,
    util_threshold_pct: int = 10,
) -> None:
    """Co-tenancy policy (clean-window): never measure on a busy GPU.

    A sample is clean only when BOTH the GPU memory held by compute apps is
    below ``threshold_mib`` AND utilization is below ``util_threshold_pct``.
    The window counts as clean only after ``confirm`` CONSECUTIVE clean
    samples (a single lucky sample between co-tenant bursts doesn't). If it
    never opens within ``wait_minutes``, DEFER (hard error) rather than
    produce contaminated timings. Other processes are never killed.
    """

    def _sample() -> tuple[int, int]:
        return _gpu_used_mib(), _gpu_busy_pct()

    used, util = _sample()
    if used < threshold_mib and util < util_threshold_pct:
        print(
            f"[runner] GPU clean ({used} MiB held, {util}% util — thresholds "
            f"{threshold_mib} MiB / {util_threshold_pct}%)"
        )
        return
    print(
        f"[runner] GPU busy ({used} MiB held, {util}% util) — waiting up to "
        f"{wait_minutes:.0f} min for a clean window "
        f"({confirm} consecutive samples @ {poll_s:.0f}s)"
    )
    deadline = time.monotonic() + wait_minutes * 60
    streak = 0
    while time.monotonic() < deadline:
        time.sleep(poll_s)
        used, util = _sample()
        if used < threshold_mib and util < util_threshold_pct:
            streak += 1
            if streak >= confirm:
                print(
                    f"[runner] clean window confirmed ({used} MiB, {util}% util, "
                    f"{streak} consecutive samples) — proceeding"
                )
                return
        else:
            streak = 0
    _fail(
        f"GPU still busy ({used} MiB held, {util}% util) after "
        f"{wait_minutes:.0f} min — deferred. Re-run when the GPU is free."
    )


class TegraStats:
    """tegrastats --interval <ms> sampler; absent on non-Jetson hosts."""

    def __init__(self, out_path: Path, interval_ms: int = 100):
        self.out = open(out_path, "w")
        self.proc = None
        self.interval_ms = interval_ms

    def start(self):
        try:
            self.proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=self.out,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            self.proc = None

    def stop(self):
        """Terminate and reap the sampler (never leaks a child)."""
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.out.close()


def _container_anon_mib(container: str) -> str:
    """Anon (process) memory of the container from its own cgroup memory.stat
    ('' if unavailable). Handles cgroup v2 (anon) and v1 (total_anon)."""
    try:
        r = subprocess.run(
            [
                "docker",
                "exec",
                container,
                "sh",
                "-c",
                "cat /sys/fs/cgroup/memory.stat 2>/dev/null || "
                "cat /sys/fs/cgroup/memory/memory.stat 2>/dev/null",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in r.stdout.splitlines():
            parts = line.split()  # cgroup memory.stat is space-separated
            if len(parts) == 2 and parts[0] in ("anon", "total_anon"):
                return f"{int(parts[1]) / 1024 / 1024:.0f}MiB"  # bytes -> MiB
    except Exception:
        pass
    return ""


def docker_stats_sampler(
    container: str, out_path: Path, stop_evt: threading.Event, interval: float = 1.0
):
    """Sample `docker stats --no-stream` for the child container (RSS, CPU%)."""
    with open(out_path, "w") as f:
        f.write("t_ns,cpu_perc,mem_usage,mem_limit,anon\n")
        t0 = time.perf_counter_ns()
        failures = 0
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
                    # anon = true process memory; MemUsage also counts the
                    # container's ACTIVE file cache (the model just read),
                    # which is exactly the ~1 GB of noise we're excluding
                    anon = _container_anon_mib(container)
                    f.write(f"{time.perf_counter_ns() - t0},{cpu},{mem_used},,{anon}\n")
                    f.flush()
                    failures = 0
                else:
                    failures += 1
            except Exception:
                failures += 1
            if failures and failures % 30 == 1:
                print(
                    f"[runner] docker stats unavailable for {container} "
                    f"({failures} failed samples so far)"
                )
            stop_evt.wait(interval)


def _free_port() -> int:
    """Grab an ephemeral loopback port for the server container's -p mapping."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_served(transport: str, host_port: int, timeout_s: float = 300.0) -> bool:
    """Poll until the voice-server answers: HTTP /health (openai) or a
    wyoming Describe->Info handshake (TCP).

    The wyoming probe must be protocol-level: docker's userland proxy ACCEPTS
    the host-side TCP connect before the container server is listening, so a
    bare connect() succeeds during model load and every early request gets
    connection-reset (seen live: 500 instant resets racing a whisper load).
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if transport == "wyoming":
            try:
                import asyncio

                from wyoming.client import AsyncTcpClient
                from wyoming.info import Describe, Info

                async def _probe() -> bool:
                    async with AsyncTcpClient("127.0.0.1", host_port) as client:
                        await client.write_event(Describe().event())
                        while True:
                            ev = await asyncio.wait_for(client.read_event(), 10)
                            if ev is None:
                                return False
                            if Info.is_type(ev.type):
                                return True

                if asyncio.run(asyncio.wait_for(_probe(), timeout=15)):
                    return True
            except Exception:
                pass
        else:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{host_port}/health", timeout=2
                ) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
        time.sleep(1.0)
    return False


def _run_served(
    engine_img: str,
    job: dict,
    run_dir: Path,
    bench_dir: Path,
    gpu: bool,
    name: str,
    serve: dict,
) -> dict:
    """Serve-mode job (docs/voice-api.md; server-api-audit plan §Runner).

    The image runs its own voice-server (`docker run -d`, CMD serves);
    client.py drives it from the HOST over loopback — no code mounts, host
    paths stay host paths. Samplers wrap the server container, matching the
    container-mode measurement window (tegrastats is run-level).
    """
    eid = job["engine_id"]
    run_dir = Path(run_dir).resolve()
    bench_dir = Path(bench_dir).resolve()
    t0 = time.perf_counter_ns()
    cid = ""
    try:
        host_port = _free_port()
        cmd = ["docker", "run", "-d", "--name", name]
        if gpu:
            cmd += ["--runtime", "nvidia", "-e", "NVIDIA_DISABLE_REQUIRE=1"]
        # per-engine server env (e.g. DEVICE=cpu / PARAKEET_DEVICE=cpu rows);
        # "{backend}" in a value expands to the job's requested_backend so one
        # entry can serve both device rows (paradee/pocket [cuda, cpu])
        for k, v in (serve.get("env") or {}).items():
            cmd += [
                "-e",
                f"{k}={str(v).replace('{backend}', str(job.get('requested_backend') or ''))}",
            ]
        for vol in job.pop("volumes", []):
            cmd += ["-v", vol]
        cmd += ["-p", f"127.0.0.1:{host_port}:{int(serve['port'])}", engine_img]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return {
                "engine": job["engine"],
                "engine_id": eid,
                "image": engine_img,
                "container": name,
                "exit": r.returncode,
                "served": True,
                "tail": (r.stderr or r.stdout).strip().splitlines()[-5:],
            }
        cid = (r.stdout or "").strip().splitlines()[-1]
        # job for the HOST client: untouched host audio paths + endpoint info
        job = {
            **job,
            "serve": {**serve, "host": "127.0.0.1", "host_port": host_port},
            "out_dir": str(run_dir),
        }
        job_path = run_dir / f"job_{eid}.json"
        job_path.write_text(json.dumps(job, ensure_ascii=False))
        if not _wait_served(serve["transport"], host_port):
            logs = subprocess.run(
                ["docker", "logs", name], capture_output=True, text=True
            ).stdout
            return {
                "engine": job["engine"],
                "engine_id": eid,
                "image": engine_img,
                "container": name,
                "exit": 1,
                "served": True,
                "tail": ("server never became healthy: " + logs)
                .strip()
                .splitlines()[-8:],
            }
        stop_evt = threading.Event()
        stats_t = threading.Thread(
            target=docker_stats_sampler,
            args=(name, run_dir / f"docker_stats_{eid}.csv", stop_evt),
            daemon=True,
        )
        stats_t.start()
        try:
            proc = subprocess.run(
                [sys.executable, str(bench_dir / "client.py"), str(job_path)],
                capture_output=True,
                text=True,
            )
        finally:
            stop_evt.set()
            stats_t.join(timeout=10)
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        (run_dir / f"logs_{eid}.txt").write_text(
            (proc.stdout or "") + (proc.stderr or "")
        )
        meta = {
            "engine": job["engine"],
            "engine_id": eid,
            "image": engine_img,
            "requested_backend": job.get("requested_backend"),
            "container": name,
            "exit": proc.returncode,
            "wall_ns": time.perf_counter_ns() - t0,
            "served": True,
            "host_port": host_port,
        }
        if proc.returncode != 0:
            meta["tail"] = (
                ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()[-5:]
            )
            print(
                f"[runner] WARNING: served engine {eid} client exited "
                f"{proc.returncode}; see logs_{eid}.txt in the run dir"
            )
        return meta
    finally:
        if cid:
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True)


def _container_audio_path(host_path: str, cache_root: Path) -> str:
    """Map a manifest audio_path to its container mount (/data/bench-cache/...).

    Manifests store cache-relative paths; absolute paths under cache_root are
    tolerated (legacy manifests) — anything else is a hard error.
    """
    p = Path(host_path)
    if p.is_absolute():
        try:
            rel = p.resolve().relative_to(cache_root)
        except ValueError as e:
            raise ValueError(
                f"manifest audio_path {host_path!r} is outside cache_dir "
                f"{cache_root} — regenerate the manifest (prepare now stores "
                "cache-relative paths)"
            ) from e
    else:
        rel = p
    return "/data/bench-cache/" + str(rel)


def run_container_job(
    engine_img: str,
    job: dict,
    run_dir: Path,
    bench_dir: Path,
    cache_dir: str,
    gpu: bool = True,
) -> dict:
    """Spawn the engine container with the client mounted; wait; return meta.

    Serve mode (job["serve"] = {port, transport}): the image runs its own
    voice-server and the client drives it from the host — see _run_served.
    """
    eid = job["engine_id"]
    name = f"bench-{job['task']}-{eid}-{int(time.time())}"
    # container-side paths; work on a copy — run_config reuses the same job dict
    import copy as _copy

    job = _copy.deepcopy(job)
    cache_root = Path(cache_dir).resolve()
    if serve := job.pop("serve", None):
        # host client: manifests store cache-RELATIVE audio paths — resolve
        # against the real cache dir (container mode maps to /data/bench-cache)
        for s in job["samples"]:
            if s.get("audio_path"):
                p = Path(s["audio_path"])
                if not p.is_absolute():
                    s["audio_path"] = str(cache_root / p)
        return _run_served(engine_img, job, run_dir, bench_dir, gpu, name, serve)
    run_dir = Path(run_dir).resolve()
    bench_dir = Path(bench_dir).resolve()
    for s in job["samples"]:
        if s.get("audio_path"):
            s["audio_path"] = _container_audio_path(s["audio_path"], cache_root)
    job_path = run_dir / f"job_{eid}.json"
    job_path.write_text(
        json.dumps({**job, "out_dir": "/bench/out"}, ensure_ascii=False)
    )
    cmd = ["docker", "run", "--rm", "--name", name]
    if gpu:
        cmd += ["--runtime", "nvidia", "-e", "NVIDIA_DISABLE_REQUIRE=1"]
    # per-engine extra mounts (e.g. host model volumes not baked in the image)
    for vol in job.pop("volumes", []):
        cmd += ["-v", vol]
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
        "image": engine_img,
        "requested_backend": job.get("requested_backend"),
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
    """Execute the spec §9.1 protocol for one config. Returns run_id."""
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text())
    unknown = set(cfg) - KNOWN_KEYS
    if unknown:
        _fail(f"unknown keys {sorted(unknown)} (typo or unsupported knob)")
    # value-level validation for keys that exist but are easy to get wrong
    if (norm := cfg.get("normalizer")) and norm != "whisper_english":
        _fail(
            f"normalizer {norm!r} unsupported (only the fixed whisper_english "
            "normalizer exists — spec 3.5 forbids per-engine normalizers)"
        )
    teg_cfg = cfg.get("probes", {}) or {}
    if teg_cfg.get("tegrastats") not in (None, "auto", True, False):
        _fail("probes.tegrastats must be auto/true/false")
    if int(cfg.get("repeats", 1)) != 1:
        _fail("repeats > 1 is not implemented yet — set repeats: 1")
    # serve mode validation (docs/voice-api.md): per-engine voice-server flag
    for eng in cfg["engines"]:
        if serve := eng.get("serve"):
            if (
                not isinstance(serve, dict)
                or not serve.get("port")
                or serve.get("transport") not in ("openai", "wyoming")
                or not isinstance(serve.get("env", {}), dict)
            ):
                _fail(
                    "engines[].serve must be {port: <container port>, "
                    "transport: openai|wyoming, env?: {K: V}} — the runner "
                    "docker-runs the image's own server and drives it from "
                    "the host"
                )
    # co-tenancy gate BEFORE touching anything: if the GPU is busy, wait for a
    # clean window or defer (never measure contaminated, never kill tenants)
    gpu_backends = [
        b for eng in cfg["engines"] for b in eng.get("backends", ["cuda"]) if b != "cpu"
    ]
    if gpu_backends:
        _wait_for_free_gpu(
            int(cfg.get("gpu_free_threshold_mib", 100000)),
            float(cfg.get("gpu_wait_minutes", 30)),
            confirm=int(cfg.get("gpu_confirm_samples", 6)),
            util_threshold_pct=int(cfg.get("gpu_util_threshold_pct", 10)),
        )
    root = Path(config_path).resolve().parents[2]  # bench/configs/x.yaml -> repo root
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{cfg.get('device_label', 'device')}_{cfg['task']}"
    out = Path(root / cfg.get("output_dir", "results")) / run_id
    out.mkdir(parents=True, exist_ok=True)
    Path(out / "audio").mkdir(exist_ok=True)

    interval_ms = int((cfg.get("probes", {}) or {}).get("interval_ms", 100))
    warmup_n = int(cfg.get("warmup_n", 0))
    cooldown_s = float(cfg.get("cooldown_s", 0))
    soak_s = float(cfg.get("soak_minutes", 0)) * 60
    realtime_pass = bool((cfg.get("streaming", {}) or {}).get("realtime_pass", False))

    # 1) idle baseline is covered by the probes running throughout the run
    teg = TegraStats(out / "tegrastats.log", interval_ms)
    teg.start()
    try:
        # 2) environment record per run (spec 9.1 / 5.4): device info + config
        from .env import collect

        env_rec = collect()
        env_rec["power_profile"] = cfg.get("power_profile", "default")
        env_rec["device_label"] = cfg.get("device_label", "")
        env_rec["config"] = cfg  # full resolved config, for provenance
        (out / "env.json").write_text(json.dumps(env_rec, indent=2))

        # page-cache drop right before measuring (cachectl): a warm
        # file cache inflates the container memory numbers. Automatic — runs
        # passwordlessly with the grant, silently skips without it.
        from . import cachectl

        cachectl.prepare_bench()

        manifest = list(jsonl_read(root / cfg["dataset"]["manifest"]))
        manifest = manifest[: int(cfg["dataset"].get("n", len(manifest)))]
        task = cfg["task"]
        first_job = True
        for i, eng in enumerate(cfg["engines"]):
            backends = eng.get("backends", ["cuda"])
            for bi, backend in enumerate(backends):
                # H2: one job per (engine, backend); id stays stable for the
                # single-backend case, gains the backend suffix otherwise
                if eng.get("id"):
                    eid = eng["id"] if len(backends) == 1 else f"{eng['id']}_{backend}"
                else:
                    eid = (
                        f"{eng['name']}_{i}"
                        if len(backends) == 1
                        else f"{eng['name']}_{i}_{backend}"
                    )
                if cooldown_s and not first_job:
                    time.sleep(cooldown_s)
                first_job = False
                # cold page cache per measurement — without this the previous
                # engine's model files sit in cache and inflate the next one's
                # cgroup memory reading (cachectl; silent sans grant)
                from . import cachectl

                cachectl.prepare_bench(quiet=i > 0 or bi > 0)
                job = {
                    "task": task,
                    "engine": eng["name"],
                    "engine_id": eid,
                    "opts": eng.get("opts", {}) | {"device": backend},
                    "requested_backend": backend,
                    "samples": manifest,
                    "warmup_n": warmup_n,
                    "streaming": realtime_pass,
                    "volumes": eng.get("volumes", []),
                    "serve": eng.get("serve"),
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
        if soak_s:
            time.sleep(soak_s)
    finally:
        teg.stop()
        # release the run's page cache back to the host
        try:
            from . import cachectl

            cachectl.prepare_bench(quiet=True)
        except Exception:
            pass

    # aggregate
    write_summary(out, task)

    # benchmarks/history.jsonl (+ committed raws for baseline-class runs)
    from . import history as _history

    _history.append_run(out, root)
    if cfg.get("publish_raw", str(cfg.get("run_name", "")).find("baseline") >= 0):
        raw = _history.publish_raw(out, root)
        print(f"[runner] raw artifacts -> {raw}")
    return run_id


def _engine_summary(rows: list[dict], task: str) -> dict:
    """Aggregate one engine's utterance rows (spec 8.3). Mutates rows with hyp_norm."""
    ok = [r for r in rows if r.get("status") == "ok"]
    s: dict = {"n": len(rows), "n_ok": len(ok)}
    if skipped := [r for r in rows if r.get("status") == "skipped"]:
        s["n_skipped"] = len(skipped)
        s["skip_reason"] = skipped[0].get("reason", "")
    if task == "stt" and ok:
        from .metrics.normalize import normalize

        for r in ok:  # host-side normalization (containers lack the normalizer)
            r["hyp_norm"] = normalize(r.get("hyp_raw", ""), r.get("language", "en"))
        s.update(
            corpus_wer_cer((r["ref_norm"] for r in ok), (r["hyp_norm"] for r in ok))
        )
        timed = [r for r in ok if r.get("audio_s")]
        if timed:
            s.update(
                dist_stats(
                    [r["proc_ns"] / 1e9 / max(r["audio_s"], 1e-9) for r in timed],
                    "rtf",
                )
            )
        s.update(dist_stats([r["proc_ns"] / 1e9 for r in ok], "offline_latency_s"))
    if task == "tts" and ok:
        s.update(dist_stats([r["synth_ns"] / 1e9 for r in ok], "synth_latency_s"))
        if all(r.get("ttfa_ns") for r in ok):
            s.update(dist_stats([r["ttfa_ns"] / 1e9 for r in ok], "ttfa_s"))
        # RTF against the produced audio's duration (spec 6.2)
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
    """Aggregate utterances.jsonl → summary.json + summary.csv (spec §8.3).

    Headline metrics are computed PER ENGINE (per_engine.engine_id); the top
    level is corpus-wide and only meaningful for single-engine runs.
    """
    utt = Path(run_dir) / "utterances.jsonl"
    rows = list(jsonl_read(utt)) if utt.exists() else []
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
    """Side-by-side headline table (spec 8.4; the full §10 matrix is fast-follow)."""
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
