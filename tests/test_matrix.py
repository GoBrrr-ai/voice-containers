"""matrix — normalization direction + cohort identity (the pure parts)."""

from bench.matrix import _norm, cohort_of


def test_norm_respects_direction():
    assert _norm([1.0, 2.0, 3.0], lower_better=True) == [1.0, 0.5, 0.0]
    assert _norm([1.0, 2.0, 3.0], lower_better=False) == [0.0, 0.5, 1.0]


def test_norm_constant_row_is_neutral():
    assert _norm([5.0, 5.0, 5.0], lower_better=True) == [0.5, 0.5, 0.5]


def test_cohort_defaults_for_sparse_rows():
    # no task + no metrics -> tts side, unknown device/manifest placeholders
    assert cohort_of({}) == ("tts", "?", "default", "?")


def test_cohort_keys_on_the_four_never_mix_axes():
    row = {
        "task": "stt",
        "device_label": "jetson-agx-thor-128gb",
        "power_profile": "MAXN",
        "manifest": "manifests/stt.jsonl",
    }
    assert cohort_of(row) == (
        "stt",
        "jetson-agx-thor-128gb",
        "MAXN",
        "manifests/stt.jsonl",
    )
