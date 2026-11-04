"""S8 chain probe — the assistant metric: stop-speaking -> first TTS audio.

Drives the LIVE deployment shape over loopback, sequentially per utterance:

    t0  POST wav    -> STT final text          (stt_ns)
    t1  chat stream -> first token             (llm_ttft_ns)
        accumulate -> first sentence ".!?"      (llm_1sent_ns; the text the
                                                 deployment sends to TTS first)
    t2  pcm stream  -> first audio chunk       (tts_ttfa_ns)

chain_ns = tts_first - t0. The LLM is REAL (llama.cpp GPU, deployment-default
Qwen3-4B-Instruct-2507 Q4_K_M) — no stub; the same three containers an
assistant would run, co-resident on the GPU like production.

Not a bench task: no WER / history rows — STT quality lives in the engine
rows. This probe owns the latency decomposition only.
Results: results/chain_<ts>/chain.jsonl + summary.json; table to stdout.

Usage:
    bench/.venv/bin/python -m bench.chain_probe [--n 100] [--engines a,b]
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
from pathlib import Path

CACHE_DIR = Path("datasets/cache").resolve()
MANIFEST = Path("manifests/stt_librispeech_test-clean_en_500_seed42.jsonl").resolve()
LLM_IMAGE = "narandill/llama-cpp:0.8.0"
LLM_MODEL = Path("task/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf").resolve()

TTS_IMAGE = "narandill/kokoro-tts:0.1.0"

# The duplex-relevant STTs: the latency anchor (parakeet-cpp) and the two
# streaming engines (native partials; time-to-final is the honest turn-based
# cost — the overlap optimization is future work, this is its baseline).
STT_ENGINES = {
    "parakeet_q4k": {
        "image": "narandill/parakeet-tdt-cpp-stt:0.1.0",
        "env": {},
        "engine": "parakeet_cpp",
    },
    "nemostream_cuda": {
        "image": "narandill/nemotron-streaming-stt:0.2.0",
        "env": {"NEMOSTREAM_PROVIDER": "cuda"},
        "engine": "nemotron_streaming",
    },
    "moonshine_stream": {
        "image": "narandill/moonshine-streaming-stt:0.2.0",
        "env": {"MOONSHINE_ENGINE": "streaming", "DEVICE": "cpu"},
        "engine": "moonshine",
    },
}

SYSTEM_PROMPT = (
    "You are a concise voice assistant. Reply with ONE short spoken sentence."
)

RUNTIME = ["--runtime", "nvidia", "-e", "NVIDIA_DISABLE_REQUIRE=1"]


def _sh(*args):
    return subprocess.run(args, capture_output=True, text=True)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_http(url: str, timeout_s: float = 180) -> bool:
    import urllib.request

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:  # nosec B310 - loopback
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


class Container:
    def __init__(self, name, image, container_port, env=None, cmd=None, volumes=()):
        self.name = name
        self.host_port = _free_port()
        c = ["docker", "run", "-d", "--rm", "--name", name, *RUNTIME]
        for k, v in (env or {}).items():
            c += ["-e", f"{k}={v}"]
        for v in volumes:
            c += ["-v", v]
        c += ["-p", f"127.0.0.1:{self.host_port}:{container_port}"]
        c.append(image)
        if cmd:
            c += cmd
        r = _sh(*c)
        if r.returncode != 0:
            raise RuntimeError(f"container {name}: {r.stderr[-300:]}")

    def down(self):
        _sh("docker", "rm", "-f", self.name)


def _now_ns() -> int:
    return time.perf_counter_ns()


def _chat_stream(llm, model: str, user: str, max_tokens: int = 48):
    return llm.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
        stream=True,
    )


def _stream_first_sentence(stream):
    """-> (ttft_ns, sent_ns, sentence, done); sentence cut at the first .!? —
    the text the deployment would send to TFS first. No draining here: the
    TTS stage starts at sent_ns (the remainder streams past us uncaptured)."""
    ttft = sent_t = None
    buf = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if not delta:
            continue
        now = _now_ns()
        if ttft is None:
            ttft = now
        buf += delta
        if sent_t is None:
            for i, ch in enumerate(buf):
                if ch in ".!?":
                    return ttft, now, buf[: i + 1], False
    return ttft, _now_ns(), buf.strip(), True


def _run_once(eng, llm, tts, s, model: str):
    """One chain pass; s=None -> warmup against the first manifest utterance."""
    import wave

    if s is None:
        s = dict(json.loads(MANIFEST.read_text().splitlines()[0]), utt_id="warmup")
    wav = CACHE_DIR / s["audio_path"]
    t0 = _now_ns()
    with wave.open(str(wav), "rb") as w:
        pcm = w.readframes(w.getnframes())
        sr = w.getframerate()
    hyp = eng.transcribe(pcm, sr, wav_path=str(wav))
    t1 = _now_ns()
    if not hyp.strip():
        return None
    stream = _chat_stream(llm, model, hyp)
    ttft, sent_t, first_sentence, _done = _stream_first_sentence(stream)
    if not first_sentence.strip():
        return None
    t2 = _now_ns()
    first_audio = None
    chunks = 0
    for _pcm, _sr in tts.synthesize_stream(first_sentence):
        chunks += 1
        if first_audio is None:
            first_audio = _now_ns()
    t3 = _now_ns()
    if not chunks:
        return None
    return {
        "utt_id": s["utt_id"],
        "audio_s": s.get("duration_s"),
        "hyp": hyp,
        "first_sentence": first_sentence,
        "stt_ns": t1 - t0,
        "llm_ttft_ns": (ttft or t2) - t1,
        "llm_1sent_ns": sent_t - t1,
        "tts_ttfa_ns": (first_audio or t3) - t2,
        "chain_ns": (first_audio or t3) - t0,
    }


def probe_engine(key: str, cfg: dict, llm, tts, model: str, rows: list, n: int):
    from bench.engines.remote import OpenAISTTRemote

    stt = Container(f"chain-stt-{key}", cfg["image"], 8080, env=cfg["env"] or None)
    try:
        time.sleep(3)
        if not _wait_http(f"http://127.0.0.1:{stt.host_port}/health", 180):
            print(f"[chain] {key}: server never came up", flush=True)
            return
        eng = OpenAISTTRemote(cfg["engine"], "127.0.0.1", stt.host_port, {})
        eng.load({})
        _run_once(eng, llm, tts, None, model)  # engine + server-cache warmup
        manifest = [
            json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()
        ][:n]
        n_ok = 0
        for s in manifest:
            rec = _run_once(eng, llm, tts, s, model)
            if rec:
                rec["stt_engine"] = key
                rec["run_position"] = len([r for r in rows if r["stt_engine"] == key])
                rows.append(rec)
                n_ok += 1
        print(f"[chain] {key}: {n_ok}/{n} ok", flush=True)
    finally:
        stt.down()


def main() -> int:
    import argparse

    from openai import OpenAI

    from bench.engines.remote import OpenAITTSRemote

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--engines", default=",".join(STT_ENGINES))
    args = ap.parse_args()

    assert LLM_MODEL.exists(), f"missing {LLM_MODEL}"
    llm_c = Container(
        "chain-llm",
        LLM_IMAGE,
        8080,
        volumes=[f"{LLM_MODEL.parent}:/models:ro"],
        cmd=[
            "llama-server",
            "-m",
            f"/models/{LLM_MODEL.name}",
            "-ngl",
            "999",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
        ],
    )
    tts_c = Container("chain-tts", TTS_IMAGE, 8880)
    try:
        assert _wait_http(f"http://127.0.0.1:{llm_c.host_port}/health", 300), "llm up"
        assert _wait_http(f"http://127.0.0.1:{tts_c.host_port}/health", 180), "tss up"
        llm_model_id = json.loads(
            _sh("curl", "-s", f"http://127.0.0.1:{llm_c.host_port}/v1/models").stdout
        )["data"][0]["id"]
        llm = OpenAI(
            base_url=f"http://127.0.0.1:{llm_c.host_port}/v1", api_key="bench"
        ).chat.completions
        tts = OpenAITTSRemote("kokoro", "127.0.0.1", tts_c.host_port, {})
        tts.load({})
        # pipeline warmup: prompt cache + token stream + tts engine
        for chunk in _chat_stream(llm, llm_model_id, "Hello."):
            pass
        for _ in tts.synthesize_stream("Warmup sentence."):
            pass

        run_dir = Path("results").resolve() / f"chain_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        for key in [k for k in args.engines.split(",") if k]:
            probe_engine(key, STT_ENGINES[key], llm, tts, llm_model_id, rows, args.n)

        from bench.metrics.wer import dist_stats

        summary = {
            "llm_model": f"{LLM_MODEL.name} (llama-cpp GPU -ngl 999)",
            "tts": TTS_IMAGE,
            "n_requested": args.n,
            "per_engine": {},
        }
        for key in sorted({r["stt_engine"] for r in rows}):
            rs = [r for r in rows if r["stt_engine"] == key]
            if not rs:
                continue
            stages: dict = {}
            for ns_key, name in (
                ("stt_ns", "stt_s"),
                ("llm_ttft_ns", "llm_ttft_s"),
                ("llm_1sent_ns", "llm_1sent_s"),
                ("tts_ttfa_ns", "tts_ttfa_s"),
                ("chain_ns", "chain_s"),
            ):
                stages.update(dist_stats([r[ns_key] / 1e9 for r in rs], name))
            summary["per_engine"][key] = {"n_ok": len(rs), **stages}

        (run_dir / "chain.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        print(
            f"\n== S8 chain — stop-speaking -> first audio "
            f"(n={args.names if False else args.n}/engine, llm {llm_model_id}, "
            f"tts kokoro)",
        )
        print(
            "| stt | chain p50 | p90 | stt p50 | llm ttft | 1st-sent | tts ttfa | n |"
        )
        print("|---|---|---|---|---|---|---|---|")
        for key, m in summary["per_engine"].items():
            ms = lambda k: f"{m[k] * 1000:.0f}" if m.get(k) else "-"  # noqa: E731
            print(
                f"| {key} | {ms('chain_s')} | {ms('chain_s_p90')} | {ms('stt_s')} "
                f"| {ms('llm_ttft_s')} | {ms('llm_1sent_s')} "
                f"| {ms('tts_ttfa_s')} | {m['n_ok']} |"
            )
        print(f"results -> {run_dir.relative_to(Path.cwd())}")
        return 0
    finally:
        llm_c.down()
        tts_c.down()


if __name__ == "__main__":
    raise SystemExit(main())
