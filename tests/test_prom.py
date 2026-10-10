import math
import time

import pytest

from sglang_tui.prom import Differ, History, build_snapshot, gauge, is_nan, parse

SNIPPET = """
# HELP x A counter
# TYPE x counter
x_total{mode="a"} 100.0
x_total{mode="b"} 50.0
g{m="1"} 2.0
nanv 1.0
nanv NaN
h_bucket{le="0.1"} 0.0
h_bucket{le="1.0"} 10.0
h_bucket{le="10.0"} 12.0
h_bucket{le="+Inf"} 12.0
"""


def test_is_nan_handles_none_finite_and_special_float_values():
    assert is_nan(None)
    assert is_nan(float("nan"))
    assert not is_nan(0.0)
    assert not is_nan(float("inf"))
    assert not is_nan(float("-inf"))


@pytest.mark.parametrize("value", ["NaN", object()])
def test_is_nan_rejects_non_numeric_values(value):
    with pytest.raises(TypeError):
        is_nan(value)


def test_parse_basic():
    s = parse(SNIPPET)
    assert gauge(s, "x_total") == 150.0
    assert gauge(s, "x_total", {"mode": "a"}) == 100.0
    assert gauge(s, "g") == 2.0


def test_parse_skips_nan_values_in_gauge():
    s = parse(SNIPPET)
    assert gauge(s, "nanv") == 1.0  # NaN series skipped, finite summed


def test_differ_rate_and_restart():
    s1 = parse(SNIPPET)
    d = Differ(rate_window=10.0)
    t = 1000.0
    d.update(s1, t)
    assert d.rate("x_total") in (None, 0.0)  # first poll: no baseline
    s2 = parse(SNIPPET.replace("100.0", "200.0").replace("50.0", "100.0"))
    d.update(s2, t + 10)
    assert abs(d.rate("x_total") - 15.0) < 1e-9  # (300-150)/10


def test_differ_rate_sliding_baseline_smooths_bursts():
    # sglang flushes counters in bursts (~30s apart). The sliding baseline keeps
    # the pre-burst point alive so the rate reads the burst as a smooth /s value
    # instead of a spike-then-zero sawtooth.
    d = Differ(rate_window=45.0)
    d.update(parse("x_total 1000.0\n"), 1000.0)
    burst = parse("x_total 1200.0\n")
    d.update(burst, 1030.0)
    assert abs(d.rate("x_total") - 200.0 / 30.0) < 1e-9
    d.update(burst, 1035.0)  # no growth; baseline (t=1000) still inside 45s window
    assert abs(d.rate("x_total") - 200.0 / 35.0) < 1e-9


def test_differ_counter_regression():
    d = Differ()
    d.update(parse(SNIPPET), 1000.0)
    d.update(parse(SNIPPET), 1010.0)
    assert d.rate("x_total") == 0.0
    # counter drops (server restart) -> series cleared, no negative rate
    d.update(parse(SNIPPET.replace('100.0', '1.0').replace('50.0', '1.0')), 1020.0)
    assert d.rate("x_total") in (None, 0.0)


def test_differ_quantile():
    d = Differ()
    d.update(parse(SNIPPET), 1000.0)
    # no growth -> no observations in window -> None
    d.update(parse(SNIPPET), 1001.0)
    assert d.quantile("h", 0.5) is None
    # grow buckets: 10 more in (0.1,1.0], 2 more in (1.0,10]
    s4 = parse(
        """
h_bucket{le="0.1"} 0.0
h_bucket{le="1.0"} 20.0
h_bucket{le="10.0"} 22.0
h_bucket{le="+Inf"} 22.0
"""
    )
    d.update(s4, 1002.0)
    # window deltas: (0.1,1.0]=10, (1.0,10]=2 => median inside (0.1,1.0] at le=1.0
    med = d.quantile("h", 0.5)
    assert med is not None and 0.1 < med <= 1.0


def test_build_snapshot_real_dump():
    from pathlib import Path

    text = Path(__file__).parent.joinpath("fixtures/metrics_sample.txt").read_text()
    samples = parse(text)
    d = Differ()
    now = time.time()
    d.update(samples, now - 1.0)
    d.update(parse(text), now)
    s = build_snapshot(samples, d, "http://test", "Qwen3.8-Flash-Next")
    assert s.running == 2.0
    assert s.queue == 3.0
    assert s.kv_max == 693824.0
    assert abs(s.gen_tps - 297.78808501358833) < 1e-6
    assert abs(s.spec_accept_rate - 0.5541666666666667) < 1e-9
    assert s.mem_weights_gb == 81.490234375
    assert s.fwd_occupancy is None  # NaN series skipped
    assert s.ttft_p50 is None  # single poll: no window yet


def test_history_nan_to_zero():
    h = History(5)
    h.push(None)
    h.push(float("nan"))
    h.push(-3.0)
    h.push(4.0)
    assert h.series() == [0.0, 0.0, 0.0, 4.0]
