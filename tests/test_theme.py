"""theme.span_wants_log — the single log-axis policy (spec 10.3)."""

from bench.theme import span_wants_log


def test_empty_or_nonpositive_never_wants_log():
    assert span_wants_log([]) is False
    assert span_wants_log([0, 0, 0]) is False
    assert span_wants_log([0, 5.0]) is False  # single positive value = no span


def test_flat_span_stays_linear():
    assert span_wants_log([4.0, 5.0, 6.0]) is False


def test_span_at_threshold_is_inclusive():
    assert span_wants_log([1.0, 10.0]) is True  # 10 / 1 == min_span


def test_just_below_default_threshold():
    assert span_wants_log([1.0, 9.999]) is False


def test_real_latency_spans_go_log():
    assert span_wants_log([0.16, 2.08]) is True  # the tts-tail reality, 13x


def test_min_span_override():
    assert span_wants_log([1.0, 50.0], min_span=100) is False
    assert span_wants_log([1.0, 100.0], min_span=100) is True
