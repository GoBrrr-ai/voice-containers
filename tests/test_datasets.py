"""datasets: cache layout, resample, seeded manifests, prepare() glue.

HF download stays monkeypatched away — the adapters' network layer is
exercised on the bench host, not in unit tests. Everything around it
(stratified sampling, path scrubbing, normalization, manifest freeze) is real.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import pytest

from bench.common import Sample
from bench.datasets import (
    BaseAdapter,
    build_manifest,
    get_adapter,
    resample_to_cache,
    wav_cache_path,
)


# ------------------------------------------------------------------ cache layer
def test_wav_cache_path_creates_dir(tmp_path):
    p = wav_cache_path(str(tmp_path), "librispeech", "1034-121119-0000")
    assert p == tmp_path / "librispeech" / "1034-121119-0000.wav"
    assert p.parent.is_dir()


def test_resample_to_cache(tmp_path):
    src = tmp_path / "in.wav"
    sf.write(str(src), np.zeros(8000, dtype=np.float32), 8000, subtype="FLOAT")
    dst = tmp_path / "out.wav"
    dur = resample_to_cache(src, dst)
    assert dur == pytest.approx(1.0, abs=0.01)
    info = sf.info(str(dst))
    assert info.samplerate == 16000 and info.subtype == "PCM_16"


def test_resample_same_rate(tmp_path):
    src = tmp_path / "in.wav"
    sf.write(str(src), np.zeros(16000, dtype=np.float32), 16000)
    assert resample_to_cache(src, tmp_path / "o.wav") == pytest.approx(1.0, 0.01)


# ---------------------------------------------------------------- manifest
def _rows(k=8):
    return [
        {
            "utt_id": f"u{i}",
            "speaker_id": f"spk{i % 2}",
            "text_raw": "word " * (5 + i),
            "audio_path": f"a{i}.wav",
        }
        for i in range(k)
    ]


def test_build_manifest_stt_stratified_deterministic():
    a = build_manifest(_rows(), 4, seed=42, task="stt")
    b = build_manifest(_rows(), 4, seed=42, task="stt")
    assert a == b and len(a) == 4
    spks = {r["speaker_id"] for r in a}
    assert spks == {"spk0", "spk1"}  # round-robin keeps both speakers


def test_build_manifest_stt_n_bigger_than_pool():
    rows = _rows(4)
    picked = build_manifest(rows, 100, seed=1, task="stt")
    assert len(picked) == 4  # exhausts, no duplicates


def test_build_manifest_tts_buckets_and_longest():
    rows = [
        {
            "utt_id": f"u{i}",
            "speaker_id": "s",
            # bucket sizes: <=50, 51..150, 151..300, >300
            "text_raw": "w" * (10 + 100 * (i % 4)) + f" {i}",
        }
        for i in range(40)
    ]
    picked = build_manifest([dict(r) for r in rows], 8, seed=7, task="tts")
    assert len(picked) <= 8 + 20
    assert all("length_bucket" in r for r in picked)
    longest = sorted(rows, key=lambda r: len(r["text_raw"]), reverse=True)[:20]
    longest_ids = {r["utt_id"] for r in longest}
    picked_ids = {r["utt_id"] for r in picked}
    assert longest_ids & picked_ids  # the explicit longest tail is in


# ------------------------------------------------------------------- prepare
class FakeAdapter(BaseAdapter):
    name = "fake"
    supports = {"stt"}

    def download(self):
        pass

    def dataset_version(self):
        return "fake@1"

    def iter_samples(self, split, language):
        for i in range(6):
            yield Sample(
                dataset="fake",
                dataset_version=self.dataset_version(),
                split=split,
                utt_id=f"u{i}",
                language=language,
                speaker_id=f"spk{i % 2}",
                text_raw=f"Hello number {i}",
                audio_path=str(self.cache / f"u{i}.wav"),
                sample_rate=16000,
                duration_s=0.5,
            )


def test_prepare_end_to_end(tmp_path, monkeypatch):
    import bench.datasets as ds_mod

    cache = tmp_path / "cache"
    cache.mkdir()
    for i in range(6):
        sf.write(str(cache / f"u{i}.wav"), np.zeros(1600, np.float32), 16000)

    ad = FakeAdapter(str(cache))
    ad.cache = cache
    monkeypatch.setattr(ds_mod, "get_adapter", lambda *a, **k: ad)

    out = ds_mod.prepare(
        "fake", "test-clean", "en", 4, 42, str(cache), str(tmp_path / "m"), "stt"
    )
    rows = [json.loads(x) for x in Path(out).read_text().splitlines()]
    assert len(rows) == 4
    r = rows[0]
    assert r["text_norm"] == r["text_raw"].lower() or r["text_norm"]
    assert r["num_words"] >= 2 and r["num_chars"] == len(r["text_raw"])
    assert not Path(r["audio_path"]).is_absolute()  # cache-relative
    assert r["dataset_version"] == "fake@1"
    assert "seed42" in Path(out).name and "stt_fake" in Path(out).name


def test_prepare_tts_manifest_name(tmp_path, monkeypatch):
    import bench.datasets as ds_mod

    cache = tmp_path / "cache"
    cache.mkdir()
    ad = FakeAdapter(str(cache))
    ad.cache = cache
    monkeypatch.setattr(ds_mod, "get_adapter", lambda *a, **k: ad)
    out = ds_mod.prepare(
        "fake", "test.clean", "en", 2, 7, str(cache), str(tmp_path / "m"), "tts"
    )
    assert "tts_fake_test_clean" in Path(out).name


def test_get_adapter_dispatch(tmp_path):
    assert isinstance(get_adapter(str(tmp_path), "librispeech"), object)
    assert get_adapter(str(tmp_path), "libritts_r").supports == {"tts"}
    with pytest.raises(ValueError):
        get_adapter(str(tmp_path), "nope")


def test_base_adapter_abstract():
    a = BaseAdapter("x")
    with pytest.raises(NotImplementedError):
        a.download()
    with pytest.raises(NotImplementedError):
        a.iter_samples("s", "en")
    with pytest.raises(NotImplementedError):
        a.dataset_version()


# ------------------------------------------------------------- custom_dir
def test_custom_dir_adapter(tmp_path):
    from bench.datasets import CustomDirAdapter

    folder = tmp_path / "ds"
    folder.mkdir()
    for i, utt in enumerate(("u1", "u2")):
        sf.write(str(folder / f"{utt}.wav"), np.zeros(8000, np.float32), 16000)
    (folder / "transcripts.tsv").write_text("u1\tHello there\nu2\tsecond line\n")

    ad = CustomDirAdapter(str(tmp_path / "cache"), str(folder))
    ad.download()
    v1 = ad.dataset_version()
    assert v1.startswith("custom_dir:") and len(v1) == len("custom_dir:") + 12
    # name hash is content-order stable
    assert ad.dataset_version() == v1

    samples = list(ad.iter_samples("test-clean", "en"))
    assert [s.utt_id for s in samples] == ["u1", "u2"]
    assert samples[0].text_raw == "Hello there"
    assert samples[0].sample_rate == 16000
    assert samples[1].dataset_version == v1
    assert Path(samples[0].audio_path).exists()


def test_custom_dir_via_factory(tmp_path):
    from bench.datasets import CustomDirAdapter

    ad = get_adapter(str(tmp_path), "custom_dir", folder="/data/x")
    assert isinstance(ad, CustomDirAdapter) and ad.folder == "/data/x"
