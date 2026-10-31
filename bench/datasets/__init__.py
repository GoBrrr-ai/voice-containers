"""Dataset adapters (PRD §3). Each yields common Sample records; the harness
handles resampling, normalization and manifest creation — adapters never do."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

from ..common import Sample

# Resampled-cache layout: <cache_dir>/<dataset>/<utt_id>.wav (16 kHz mono PCM16)
RESAMPLE_RATE = 16000


def cache_key(*parts: str) -> str:
    return hashlib.sha256(":".join(parts).encode()).hexdigest()[:16]


def wav_cache_path(cache_dir: str, dataset: str, utt_id: str) -> Path:
    p = Path(cache_dir) / dataset
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{utt_id}.wav"


def resample_to_cache(src: str | Path, dst: Path) -> float:
    """soxr-resample to 16 kHz mono PCM16; returns duration_s (outside timers)."""
    import soundfile as sf
    import soxr

    data, sr = sf.read(str(src), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != RESAMPLE_RATE:
        mono = soxr.resample(mono, sr, RESAMPLE_RATE)
    sf.write(str(dst), mono, RESAMPLE_RATE, subtype="PCM_16")
    return len(mono) / RESAMPLE_RATE


class BaseAdapter:
    name = "base"
    supports = set()

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir

    def download(self) -> None:
        raise NotImplementedError

    def iter_samples(self, split: str, language: str):
        raise NotImplementedError

    def dataset_version(self) -> str:
        raise NotImplementedError


class LibriSpeechAdapter(BaseAdapter):
    """HF openslr/librispeech_asr, config 'clean' (PRD 3.3)."""

    name = "librispeech"
    supports = {"stt"}
    _rev = "main"

    def download(self) -> None:
        import datasets  # noqa: F401  (downloads into HF cache)

        self._ds = datasets.load_dataset(
            "openslr/librispeech_asr",
            "clean",
            split="test",
            revision=self._rev,
            trust_remote_code=False,
        )

    def dataset_version(self) -> str:
        return f"openslr/librispeech_asr@{self._rev}"

    def iter_samples(self, split: str, language: str):
        import numpy as np
        import soundfile as sf

        for row in self._ds:
            utt = row["id"]
            dst = wav_cache_path(self.cache_dir, self.name, utt)
            # Always decode via the embedded audio (torchcodec) — extraction
            # paths recorded in the datasets cache can be stale on other hosts.
            dec = row["audio"].get_all_samples()
            data = dec.data.numpy()
            mono = data.mean(axis=0) if data.ndim > 1 else data
            if not dst.exists():
                sf.write(str(dst), mono, dec.sample_rate, subtype="PCM_16")
            dur = len(mono) / dec.sample_rate
            yield Sample(
                dataset=self.name,
                dataset_version=self.dataset_version(),
                split=split,
                utt_id=utt,
                language=language,
                speaker_id=str(row["speaker_id"]),
                gender=row.get("gender"),
                text_raw=row["text"],
                audio_path=str(dst),
                sample_rate=RESAMPLE_RATE,
                duration_s=dur,
            )


class LibriTTSRAdapter(BaseAdapter):
    """HF mythicinfinity/libritts_r, config clean, split test.clean — TEXT only
    for TTS (audio not required by the harness, PRD 7)."""

    name = "libritts_r"
    supports = {"tts"}
    _rev = "main"

    def download(self) -> None:
        import datasets

        self._ds = datasets.load_dataset(
            "mythicinfinity/libritts_r", "clean", split="test.clean", revision=self._rev
        )

    def dataset_version(self) -> str:
        return f"mythicinfinity/libritts_r@{self._rev}"

    def iter_samples(self, split: str, language: str):
        for i, row in enumerate(self._ds):
            utt = f"lttsr-{row['speaker_ID']}-{i:05d}"  # text-only: never touch the audio column
            text = " ".join((row["text_normalized"] or "").split())
            if not text:
                continue
            yield Sample(
                dataset=self.name,
                dataset_version=self.dataset_version(),
                split=split,
                utt_id=utt,
                language=language,
                speaker_id=str(row["speaker_ID"]),
                gender=None,
                text_raw=text,
                audio_path=None,
                sample_rate=None,
                duration_s=None,
            )


class FleursAdapter(BaseAdapter):
    """HF google/fleurs, per-language config (PRD 3.3)."""

    name = "fleurs"
    supports = {"stt", "tts"}
    _LANG = {
        "en_us": "en",
        "pl_pl": "pl",
        "de_de": "de",
        "fr_fr": "fr",
        "es_419": "es",
        "it_it": "it",
        "pt_br": "pt",
        "nl_nl": "nl",
    }

    def __init__(self, cache_dir: str, hf_config: str = "en_us"):
        super().__init__(cache_dir)
        self.hf_config = hf_config

    def download(self) -> None:
        import datasets

        self._ds = datasets.load_dataset(
            "google/fleurs", self.hf_config, split="test", trust_remote_code=False
        )

    def dataset_version(self) -> str:
        return f"google/fleurs@{self.hf_config}"

    def iter_samples(self, split: str, language: str):
        lang = self._LANG.get(self.hf_config, language)
        for i, row in enumerate(self._ds):
            utt = f"fleurs-{self.hf_config}-{i:05d}"
            dst = wav_cache_path(self.cache_dir, self.name, utt)
            if not dst.exists():
                resample_to_cache(row["audio"]["path"], dst)
            yield Sample(
                dataset=self.name,
                dataset_version=self.dataset_version(),
                split=split,
                utt_id=utt,
                language=lang,
                speaker_id=None,
                gender=row.get("gender"),
                text_raw=row.get("transcription", ""),
                audio_path=str(dst),
                sample_rate=RESAMPLE_RATE,
                duration_s=row["audio"]["array"].shape[0]
                / row["audio"]["sampling_rate"],
            )


class CustomDirAdapter(BaseAdapter):
    """Folder of WAVs + transcripts.tsv (`utt_id<TAB>text`) — PRD 3.3."""

    name = "custom"
    supports = {"stt", "tts"}

    def __init__(self, cache_dir: str, folder: str = ""):
        super().__init__(cache_dir)
        self.folder = folder

    def download(self) -> None:
        pass

    def dataset_version(self) -> str:
        h = hashlib.sha256()
        for p in sorted(Path(self.folder).iterdir()):
            h.update(str(p.name).encode())
        return f"custom_dir:{h.hexdigest()[:12]}"

    def iter_samples(self, split: str, language: str):
        import soundfile as sf

        tsv = Path(self.folder) / "transcripts.tsv"
        for line in tsv.read_text().splitlines():
            utt, text = line.split("\t", 1)
            wav = Path(self.folder) / f"{utt}.wav"
            dur = float(sf.info(str(wav)).duration)
            yield Sample(
                dataset=self.name,
                dataset_version=self.dataset_version(),
                split=split,
                utt_id=utt,
                language=language,
                text_raw=text,
                audio_path=str(wav),
                sample_rate=sf.info(str(wav)).samplerate,
                duration_s=dur,
            )


def get_adapter(cache_dir: str, name: str, **opts) -> BaseAdapter:
    if name == "librispeech":
        return LibriSpeechAdapter(cache_dir)
    if name == "libritts_r":
        return LibriTTSRAdapter(cache_dir)
    if name == "fleurs":
        return FleursAdapter(cache_dir, opts.get("hf_config", "en_us"))
    if name == "custom_dir":
        return CustomDirAdapter(cache_dir, opts.get("folder", ""))
    raise ValueError(f"unknown dataset {name}")


def build_manifest(samples, n: int, seed: int, task: str) -> list[dict]:
    """Frozen, seeded subset (PRD 3.4): STT stratified by speaker; TTS stratified
    by char-length bucket + the 20 longest sentences appended explicitly."""
    import random

    rows = list(samples)  # plain dicts from prepare()
    rng = random.Random(seed)
    if task == "stt":
        by_spk: dict[str, list[dict]] = {}
        for r in rows:
            by_spk.setdefault(r["speaker_id"] or "?", []).append(r)
        picked: list[dict] = []
        spks = sorted(by_spk)
        i = 0
        while len(picked) < n and any(by_spk[s] for s in spks):
            s = spks[i % len(spks)]
            if by_spk[s]:
                picked.append(by_spk[s].pop(rng.randrange(len(by_spk[s]))))
            i += 1
    else:

        def bucket(ln: int) -> str:
            return (
                "<=50"
                if ln <= 50
                else "51-150"
                if ln <= 150
                else "151-300"
                if ln <= 300
                else ">300"
            )

        by_b: dict[str, list[dict]] = {}
        for r in rows:
            r["length_bucket"] = bucket(len(r["text_raw"]))
            by_b.setdefault(r["length_bucket"], []).append(r)
        per = max(1, n // 4)
        picked = []
        for b in ("<=50", "51-150", "151-300", ">300"):
            pool = by_b.get(b, [])
            rng.shuffle(pool)
            picked += pool[:per]
        longest = sorted(rows, key=lambda r: len(r["text_raw"]), reverse=True)[:20]
        seen = {p["utt_id"] for p in picked}
        picked += [r for r in longest if r["utt_id"] not in seen]
    rng.shuffle(picked)
    return picked[: n + 20] if task != "stt" else picked[:n]


def prepare(
    dataset: str,
    split: str,
    language: str,
    n: int,
    seed: int,
    cache_dir: str,
    out_dir: str,
    task: str,
    **opts,
) -> str:
    """Download → resample cache → normalized manifest. Returns manifest path."""
    from ..metrics.normalize import normalize

    ad = get_adapter(cache_dir, dataset, **opts)
    ad.download()
    rows = []
    for s in ad.iter_samples(split, language):
        d = s.to_dict()
        d["text_norm"] = normalize(d["text_raw"], language)
        d["num_chars"] = len(d["text_raw"])
        d["num_words"] = len(d["text_norm"].split())
        rows.append(d)
    picked = build_manifest(rows, n, seed, task)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    name = f"{task}_{dataset}_{split.replace('.', '_')}_{language}_{n}_seed{seed}"
    out = Path(out_dir) / f"{name}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(out)
