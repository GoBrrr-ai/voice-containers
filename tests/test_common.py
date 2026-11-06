"""common.Sample — the shared record contract (spec 3.1)."""

from dataclasses import FrozenInstanceError

import pytest

from bench.common import Sample


def _sample(**kw):
    base = dict(
        dataset="librispeech",
        dataset_version="0123abcd",
        split="test-clean",
        utt_id="1034-121119-0000",
        language="en",
    )
    base.update(kw)
    return Sample(**base)


def test_defaults_are_the_published_contract():
    s = _sample(text_raw="Hi there", text_norm="hi there")
    assert s.audio_path is None
    assert s.duration_s is None
    assert s.speaker_id is None
    assert s.num_chars == 0
    assert s.num_words == 0


def test_to_dict_roundtrips_every_field():
    s = _sample()
    d = s.to_dict()
    assert d["utt_id"] == "1034-121119-0000"
    assert Sample(**d) == s


def test_records_are_frozen():
    with pytest.raises(FrozenInstanceError):
        _sample().utt_id = "other"
