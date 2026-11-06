"""probes parsers — pure functions over the recorded telemetry artifacts."""

from bench.probes import _parse_mem_mib, parse_docker_stats, window_aggregate


def _ts_rows(e0, n=10):
    return [
        {
            "epoch": e0 + i,
            "gpu_mw": 12000 + i,
            "gpu_util_pct": 30.0,
            "ram_mib": 2048.0,
        }
        for i in range(n)
    ]


def test_window_aggregate_bounds_and_units():
    win = window_aggregate(_ts_rows(100.0), 102.0, 3.0)  # [102, 107] incl tail
    assert win["n_samples"] == 6
    mean_w = (sum(12000 + i for i in range(2, 8)) / 6) / 1000
    assert abs(win["gpu_power_w_mean"] - mean_w) < 1e-9
    assert win["gpu_util_pct_mean"] == 30.0
    assert win["ram_used_mib_mean"] == 2048.0


def test_window_aggregate_none_when_window_empty():
    assert window_aggregate(_ts_rows(100.0), 0.0, 1.0) is None


def test_window_aggregate_tolerates_none_fields():
    rows = [
        {"epoch": 1.0, "gpu_mw": None, "gpu_util_pct": None, "ram_mib": None},
        {"epoch": 2.0, "gpu_mw": None, "gpu_util_pct": 30.0, "ram_mib": None},
    ]
    win = window_aggregate(rows, 0.0, 5.0)
    assert win["n_samples"] == 2
    assert win["gpu_power_w_mean"] is None
    assert win["gpu_util_pct_mean"] == 30.0


def test_parse_mem_mib_units():
    assert _parse_mem_mib("512MiB") == 512.0
    assert _parse_mem_mib("1GiB") == 1024.0
    assert _parse_mem_mib("1.5GiB") == 1536.0
    assert _parse_mem_mib("nonsense") == 0.0


def test_parse_docker_stats(tmp_path):
    p = tmp_path / "docker_stats_e0.csv"
    p.write_text(
        "t_ns,cpu_perc,mem_usage,mem_limit,anon\n"
        "0,1.0,512MiB,8GiB,100MiB\n"
        "1,2.0,1GiB,8GiB,300MiB\n"
    )
    out = parse_docker_stats(p)
    assert out["n_samples"] == 2
    assert out["mem_peak_mib"] == 1024.0
    assert out["anon_peak_mib"] == 300.0
    assert "anon_p50_mib" in out
    assert "mem_p50_mib" in out


def test_parse_docker_stats_missing_or_header_only(tmp_path):
    assert parse_docker_stats(tmp_path / "nope.csv") is None
    p = tmp_path / "empty.csv"
    p.write_text("t_ns,cpu_perc,mem_usage,mem_limit,anon\n")
    assert parse_docker_stats(p) is None
