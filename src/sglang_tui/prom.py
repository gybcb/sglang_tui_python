"""Prometheus text-format parsing and derived-metric computation for sglang."""

from __future__ import annotations

import math
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

Sample = Tuple[Dict[str, str], float]

_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[^\s]+)"
    r"(?:\s+(?P<ts>[^\s]+))?\s*$"
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse(text: str) -> Dict[str, List[Sample]]:
    """Parse a Prometheus exposition into {metric_name: [(labels, value)]}."""
    out: Dict[str, List[Sample]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if m is None:
            continue
        raw_labels = m.group("labels")
        labels = (
            {k: v.replace('\\"', '"').replace("\\\\", "\\") for k, v in _LABEL_RE.findall(raw_labels)}
            if raw_labels
            else {}
        )
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        out.setdefault(m.group("name"), []).append((labels, value))
    return out


def _matches(labels: Dict[str, str], label_filter: Optional[Dict[str, str]]) -> bool:
    if not label_filter:
        return True
    return all(labels.get(k) == v for k, v in label_filter.items())


def gauge(samples: Dict[str, List[Sample]], name: str, label_filter: Optional[Dict[str, str]] = None) -> Optional[float]:
    """Sum of a gauge's matching series; NaN series are skipped."""
    vals = [v for labels, v in samples.get(name, []) if _matches(labels, label_filter) and not math.isnan(v)]
    return sum(vals) if vals else None


def is_nan(value: Optional[float]) -> bool:
    return value is None or math.isnan(value)


def _cumulative_to_intervals(sorted_le_deltas: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Bucket deltas stay cumulative over le; convert to per-interval counts."""
    out: List[Tuple[float, float]] = []
    prev = 0.0
    for le, d in sorted_le_deltas:
        out.append((le, max(0.0, d - prev)))
        prev = d
    return out


def delta_of(prev_map: Dict, key, value: float) -> float:
    old = prev_map.get(key)
    return 0.0 if old is None else max(0.0, value - old)


class Differ:
    """Turn cumulative counters/histograms into rates and window quantiles.

    SGLang flushes counters in bursts on a log interval, so rates use a sliding
    ~rate_window-second baseline instead of per-poll deltas. Call update()
    once per poll.
    """

    def __init__(self, rate_window: float = 30.0, hist_window: float = 30.0) -> None:
        self._prev: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
        self._rate_window = rate_window
        self._hist_window = hist_window
        # name -> label-dim -> deque[(ts, summed value across matching series)]
        self._counter_series: Dict[str, Dict[Tuple, Deque[Tuple[float, float]]]] = {}
        # base histogram -> {labels_wo_le: deque[(ts, {le: interval_count})]}
        self._hist_series: Dict[str, Dict[Tuple, Deque[Tuple[float, Dict[float, float]]]]] = {}
        self._last_now: Optional[float] = None

    def update(self, samples: Dict[str, List[Sample]], now: float) -> None:
        new_prev: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
        # name -> labels_without_model/instance_dim -> summed value; keyed by model_name+extra for rate()
        counter_sums: Dict[Tuple[str, Tuple], float] = {}
        hist_interval: Dict[str, Dict[Tuple, Dict[float, float]]] = {}

        for name, series in samples.items():
            is_counter = name.endswith(("_total", "_count", "_sum"))
            is_bucket = name.endswith("_bucket")
            if not (is_counter or is_bucket):
                continue
            for labels, value in series:
                lkey = tuple(sorted(labels.items()))
                new_prev[(name, lkey)] = value
                if is_counter and not is_bucket:
                    ckey = tuple(sorted((k, v) for k, v in labels.items() if k in ("mode", "cache_source", "is_streaming", "endpoint", "method")))
                    counter_sums[(name, ckey)] = counter_sums.get((name, ckey), 0.0) + value
                if is_bucket:
                    base = name[: -len("_bucket")]
                    le_raw = labels.get("le", "+Inf")
                    try:
                        le = math.inf if le_raw == "+Inf" else float(le_raw)
                    except ValueError:
                        continue
                    key_wo_le = tuple(sorted((k, v) for k, v in labels.items() if k != "le"))
                    hist_interval.setdefault(base, {}).setdefault(key_wo_le, {})[le] = delta_of(self._prev, (name, lkey), value)

        for (name, ckey), total in counter_sums.items():
            dq = self._counter_series.setdefault(name, {}).setdefault(ckey, deque())
            if dq and total < dq[-1][1]:  # counter reset (server restart)
                dq.clear()
            dq.append((now, total))
            while len(dq) > 1 and now - dq[0][0] > self._rate_window:
                dq.popleft()

        for base, groups in hist_interval.items():
            for key, cum_by_le in groups.items():
                intervals = _cumulative_to_intervals(sorted(cum_by_le.items()))
                if not any(c > 0 for _, c in intervals):
                    continue
                dq = self._hist_series.setdefault(base, {}).setdefault(key, deque())
                dq.append((now, dict(intervals)))
                while len(dq) > 1 and now - dq[0][0] > self._hist_window:
                    dq.popleft()

        self._prev = new_prev
        self._last_now = now

    def rate(self, name: str, label_filter: Optional[Dict[str, str]] = None) -> Optional[float]:
        groups = self._counter_series.get(name)
        if not groups:
            return None
        total_rate = 0.0
        seen = False
        for ckey, dq in groups.items():
            labels = dict(ckey)
            if label_filter and any(labels.get(k) != v for k, v in label_filter.items()):
                continue
            if len(dq) < 2:
                continue
            (t0, v0), (t1, v1) = dq[0], dq[-1]
            dt = t1 - t0
            if dt <= 0:
                continue
            total_rate += max(0.0, (v1 - v0) / dt)
            seen = True
        return total_rate if seen else None

    def quantile(self, name: str, q: float, label_filter: Optional[Dict[str, str]] = None) -> Optional[float]:
        groups = self._hist_series.get(name)
        if not groups:
            return None
        for key, dq in groups.items():
            if not _matches(dict(key), label_filter):
                continue
            merged: Dict[float, float] = {}
            for _, by_le in dq:
                for le, c in by_le.items():
                    merged[le] = merged.get(le, 0.0) + c
            buckets = sorted(merged.items())
            total = sum(c for _, c in buckets)
            if total <= 0:
                return None
            target = q * total
            cum = 0.0
            prev_le = 0.0
            for le, c in buckets:
                if cum + c >= target:
                    if math.isinf(le):
                        return prev_le
                    frac = 0.0 if c <= 0 else (target - cum) / c
                    return prev_le + (le - prev_le) * frac
                cum += c
                prev_le = le
            return None
        return None


@dataclass
class Snapshot:
    connected: bool = True
    timestamp: float = 0.0
    model: str = ""
    url: str = ""
    running: Optional[float] = None
    queue: Optional[float] = None
    grammar_queue: Optional[float] = None
    retracted: Optional[float] = None
    aborted_rate: Optional[float] = None
    req_rate: Optional[float] = None
    http_inflight: Optional[float] = None
    gen_tps: Optional[float] = None
    decode_tps: Optional[float] = None
    prefill_tps: Optional[float] = None
    ttft_p50: Optional[float] = None
    ttft_p99: Optional[float] = None
    e2e_p50: Optional[float] = None
    e2e_p99: Optional[float] = None
    itl_p50: Optional[float] = None
    queue_p50: Optional[float] = None
    token_usage: Optional[float] = None
    kv_used: Optional[float] = None
    kv_evictable: Optional[float] = None
    kv_available: Optional[float] = None
    kv_max: Optional[float] = None
    hit_rate: Optional[float] = None
    hicache_used: Optional[float] = None
    hicache_total: Optional[float] = None
    mem_weights_gb: Optional[float] = None
    mem_kv_gb: Optional[float] = None
    mem_graph_gb: Optional[float] = None
    utilization: Optional[float] = None
    fwd_occupancy: Optional[float] = None
    is_cuda_graph: Optional[float] = None
    spec_accept_rate: Optional[float] = None
    spec_accept_length: Optional[float] = None
    spec_steps: Optional[float] = None
    context_len: Optional[float] = None


def build_snapshot(
    samples: Dict[str, List[Sample]],
    differ: Differ,
    url: str,
    model: str,
    connected: bool = True,
) -> Snapshot:
    now = time.time()
    s = Snapshot(connected=connected, timestamp=now, url=url, model=model)
    s.running = gauge(samples, "sglang:num_running_reqs")
    s.queue = gauge(samples, "sglang:num_queue_reqs")
    s.grammar_queue = gauge(samples, "sglang:num_grammar_queue_reqs")
    s.retracted = gauge(samples, "sglang:num_retracted_reqs")
    s.aborted_rate = differ.rate("sglang:num_aborted_requests_total")
    s.req_rate = differ.rate("sglang:num_requests_total")
    s.http_inflight = gauge(samples, "sglang:http_requests_active")
    s.gen_tps = gauge(samples, "sglang:gen_throughput")
    s.decode_tps = differ.rate("sglang:realtime_tokens_total", {"mode": "decode"})
    if s.decode_tps is None:
        s.decode_tps = differ.rate("sglang:generation_tokens_total")
    s.prefill_tps = differ.rate("sglang:realtime_tokens_total", {"mode": "prefill_compute"})
    if s.prefill_tps is None:
        s.prefill_tps = differ.rate("sglang:prompt_tokens_total")
    s.ttft_p50 = differ.quantile("sglang:time_to_first_token_seconds", 0.5)
    s.ttft_p99 = differ.quantile("sglang:time_to_first_token_seconds", 0.99)
    s.e2e_p50 = differ.quantile("sglang:e2e_request_latency_seconds", 0.5)
    s.e2e_p99 = differ.quantile("sglang:e2e_request_latency_seconds", 0.99)
    s.itl_p50 = differ.quantile("sglang:inter_token_latency_seconds", 0.5)
    s.queue_p50 = differ.quantile("sglang:queue_time_seconds", 0.5)
    s.token_usage = gauge(samples, "sglang:token_usage")
    s.kv_used = gauge(samples, "sglang:kv_used_tokens")
    s.kv_evictable = gauge(samples, "sglang:kv_evictable_tokens")
    s.kv_available = gauge(samples, "sglang:kv_available_tokens")
    s.kv_max = gauge(samples, "sglang:max_total_num_tokens")
    s.hit_rate = gauge(samples, "sglang:cache_hit_rate")
    s.hicache_used = gauge(samples, "sglang:hicache_host_used_tokens")
    s.hicache_total = gauge(samples, "sglang:hicache_host_total_tokens")
    s.mem_weights_gb = gauge(samples, "sglang:weight_memory_usage_gb")
    s.mem_kv_gb = gauge(samples, "sglang:kv_cache_memory_usage_gb")
    s.mem_graph_gb = gauge(samples, "sglang:graph_memory_usage_gb")
    s.utilization = gauge(samples, "sglang:utilization")
    s.fwd_occupancy = gauge(samples, "sglang:fwd_occupancy")
    s.is_cuda_graph = gauge(samples, "sglang:is_cuda_graph")
    s.spec_accept_rate = gauge(samples, "sglang:spec_accept_rate")
    s.spec_accept_length = gauge(samples, "sglang:spec_accept_length")
    s.spec_steps = gauge(samples, "sglang:spec_num_steps")
    s.context_len = gauge(samples, "sglang:context_len")
    return s


class History:
    """Fixed-length rolling series for one derived value."""

    def __init__(self, maxlen: int = 90) -> None:
        self.values: Deque[float] = deque(maxlen=maxlen)

    def push(self, value: Optional[float]) -> None:
        self.values.append(0.0 if value is None or math.isnan(value) else max(0.0, value))

    def series(self) -> List[float]:
        return list(self.values)
