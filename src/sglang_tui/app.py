"""sgl-tui: a btop-style terminal dashboard for SGLang servers."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import List, Optional

import httpx
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from .prom import Differ, History, Snapshot, build_snapshot, parse
from .widgets import (
    BLUE,
    CYAN,
    DIM,
    GREEN,
    MAGENTA,
    RED,
    YELLOW,
    ChartRow,
    Gauge,
    StatRow,
    StackedGauge,
    fmt_num,
    fmt_pct,
    fmt_secs,
    heat_color,
)


class HelpPanel(ModalScreen):
    CSS = """
    HelpPanel { align: center middle; }
    #help-box { width: 56; height: auto; border: round #00d7ff; background: #0d1117; padding: 1 2; }
    """
    BINDINGS = [("q", "dismiss", "Close"), ("escape", "dismiss", "Close")]

    def compose(self) -> ComposeResult:
        with Vertical(id="help-box"):
            yield Static(
                "[b #00d7ff]sgl-tui keys[/]\n\n"
                "  [b]q[/b] / [b]ctrl+c[/b]  quit\n"
                "  [b]+[/b] [b]-[/b]        refresh interval\n"
                "  [b]h[/b]                history window\n"
                "  [b]?[/b] / [b]f1[/b]    this help\n"
            )

    def on_key(self, event) -> None:
        # Belt-and-braces: some Textual versions don't wire ModalScreen's
        # "dismiss" action to plain key bindings, so pop the screen directly.
        if event.key in ("escape", "q"):
            event.stop()
            event.prevent_default()
            self.app.pop_screen()


class SglangTui(App):
    TITLE = "sgl-tui"
    CSS_PATH = Path(__file__).parent / "tui.tcss"

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
        ("plus", "faster", "Faster"),
        ("minus", "slower", "Slower"),
        ("question_mark", "help", "Help"),
        ("f1", "help", "Help"),
        ("h", "window", "Window"),
    ]

    def __init__(self, url: str, refresh: float = 1.0) -> None:
        super().__init__()
        self.last_error: Optional[str] = None
        self._prefer_curl = False
        self.base_url = url.rstrip("/")
        if self.base_url.endswith("/metrics"):
            self.base_url = self.base_url[: -len("/metrics")]
        self.base_url = self.base_url.rstrip("/")
        self.refresh_interval = refresh
        self.history_window = 90
        self.differ = Differ()
        self.last_snapshot: Optional[Snapshot] = None
        self._server_model = ""
        self.histories = {k: History(90) for k in ("gen", "decode", "prefill", "ttft", "e2e", "itl", "queue", "run", "qcnt", "hit")}

    def compose(self) -> ComposeResult:
        yield Static("sgl-tui", id="header")
        with Vertical(id="main"):
            with Vertical(classes="panel", id="p-requests"):
                yield ChartRow("RUNNING", id="c-run", color=GREEN)
                yield ChartRow("Q CNT", id="c-que", color=YELLOW)
                yield StatRow("GRAMMAR", id="s-grammar")
                yield StatRow("RETRACT", id="s-retract")
                yield StatRow("REQ /s", id="s-rate")
            with Vertical(classes="panel", id="p-throughput"):
                yield ChartRow("GEN", id="c-gen")
                yield ChartRow("DECODE", id="c-decode")
                yield ChartRow("PREFILL", id="c-prefill")
                yield StatRow("HTTP LIVE", id="s-http")
            with Vertical(classes="panel", id="p-kv"):
                yield Gauge("TOKENS", id="g-token")
                yield StackedGauge("KV POOL", id="k-kv")
                yield ChartRow("PREFIX HIT", id="c-hit", color=BLUE)
                yield Gauge("HICACHE", id="g-hicache")
            with Vertical(classes="panel", id="p-latency"):
                yield ChartRow("TTFT", id="c-ttft")
                yield ChartRow("E2E", id="c-e2e")
                yield ChartRow("ITL", id="c-itl")
                yield ChartRow("QUEUE", id="c-queue")
            with Vertical(classes="panel", id="p-memory"):
                yield StatRow("WEIGHTS", id="s-mw")
                yield StatRow("KV GB", id="s-mkv")
                yield StatRow("GRAPH GB", id="s-mg")
                yield StatRow("GPU UTIL", id="s-util")
            with Vertical(classes="panel", id="p-spec"):
                yield Gauge("ACCEPT", id="g-spec")
                yield StatRow("ACCEPT LEN", id="s-slen")
                yield StatRow("STEPS", id="s-steps")
        yield Static("", id="footer")

    async def on_mount(self) -> None:
        for pid, title in [
            ("p-requests", "REQUESTS"),
            ("p-throughput", "THROUGHPUT tok/s"),
            ("p-kv", "KV CACHE"),
            ("p-latency", "LATENCY p50/p99"),
            ("p-memory", "MEMORY"),
            ("p-spec", "SPEC DECODE"),
        ]:
            self.query_one(f"#{pid}").border_title = title
        self._update_footer()
        self.set_interval(1.0, self._tick_clock)
        self.run_worker(self._poll_loop(), exclusive=True, group="poll")

    async def _fetch_metrics(self, client: httpx.AsyncClient) -> str:
        """Fetch /metrics via httpx; fall back to Apple-signed system curl when the
        macOS Local Network privacy gate blocks our Python's LAN sockets."""
        url = f"{self.base_url}/metrics"
        if not self._prefer_curl:
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.text
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
                self._prefer_curl = True  # don't pay httpx timeout each poll
        proc = await asyncio.create_subprocess_exec(
            "curl", "-s", "--fail", "--max-time", "6", url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            self._prefer_curl = False  # curl failed too — retry httpx next poll
            raise RuntimeError(f"curl exit {proc.returncode}: {stderr.decode(errors='trim')[:120]}")
        return stdout.decode(errors="trim")

    async def _poll_loop(self) -> None:
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                t0 = time.time()
                try:
                    text = await self._fetch_metrics(client)
                    samples = parse(text)
                    self.differ.update(samples, t0)
                    model = ""
                    for labels, _v in samples.get("sglang:num_running_reqs", []):
                        model = labels.get("model_name", "")
                        break
                    if model:
                        self._server_model = model
                    snapshot = build_snapshot(samples, self.differ, self.base_url, self._server_model, connected=True)
                    self.last_error = None
                except Exception as exc:
                    msg = f"{type(exc).__name__}: {exc}"
                    if "No route to host" in str(exc):
                        msg += " (LAN blocked by macOS Local Network permission — see README)"
                    self.last_error = msg
                    snapshot = Snapshot(connected=False, timestamp=time.time(), url=self.base_url, model=self._server_model)

                self.last_snapshot = snapshot
                for key, val in [
                    ("gen", snapshot.gen_tps),
                    ("decode", snapshot.decode_tps),
                    ("prefill", snapshot.prefill_tps),
                    ("ttft", snapshot.ttft_p50),
                    ("e2e", snapshot.e2e_p50),
                    ("itl", snapshot.itl_p50),
                    ("queue", snapshot.queue_p50),
                    ("run", snapshot.running),
                    ("qcnt", snapshot.queue),
                    ("hit", snapshot.hit_rate),
                ]:
                    self.histories[key].push(val)
                self._apply(snapshot)
                await asyncio.sleep(self.refresh_interval)

    def _pts(self, key: str) -> List[float]:
        return self.histories[key].series()[-self.history_window :]

    def _apply(self, s: Snapshot) -> None:
        q = self.query_one
        state = "[b #5faf00]● ONLINE[/]" if s.connected else "[b #ff5f5f]● OFFLINE[/]"
        err = "" if s.connected or self.last_error is None else f" [dim]{self.last_error[:60]}[/]"
        q("#header").update(
            f"[b #00d7ff]sgl-tui[/] ─ [b]{s.model or '?'}[/] ─ {s.url} ─ {state}{err} ─ {time.strftime('%H:%M:%S')}"
        )
        if not s.connected or self.last_snapshot is None:
            return

        run_pts = self._pts("run")
        que_pts = self._pts("qcnt")
        q("#c-run").set(f"{fmt_num(s.running, 0)}/{fmt_num(max(run_pts, default=0.0), 0)}", run_pts)
        q("#c-que").set(f"{fmt_num(s.queue, 0)}/{fmt_num(max(que_pts, default=0.0), 0)}", que_pts)
        q("#s-grammar").set(fmt_num(s.grammar_queue, 0))
        q("#s-retract").set(fmt_num(s.retracted, 0), "", RED if (s.retracted or 0) > 0 else GREEN)
        q("#s-rate").set(f"{s.req_rate or 0:.2f}/s", f"abort {s.aborted_rate or 0:.2f}/s")
        q("#s-http").set(fmt_num(s.http_inflight, 0), f"ctx {fmt_num(s.context_len)}")

        gen_pts = self._pts("gen")
        decode_pts = self._pts("decode")
        prefill_pts = self._pts("prefill")
        q("#c-gen").set(f"{fmt_num(s.gen_tps)}/{fmt_num(max(gen_pts, default=0.0))}", gen_pts)
        q("#c-decode").set(f"{fmt_num(s.decode_tps)}/{fmt_num(max(decode_pts, default=0.0))}", decode_pts)
        q("#c-prefill").set(f"{fmt_num(s.prefill_tps)}/{fmt_num(max(prefill_pts, default=0.0))}", prefill_pts)

        q("#g-token").set(s.token_usage, f"{fmt_num(s.kv_used)}/{fmt_num(s.kv_max)}")
        q("#k-kv").set(
            [(s.kv_used, MAGENTA), (s.kv_evictable, BLUE), (s.kv_available, "#30363d")],
            fmt_num((s.kv_used or 0) + (s.kv_evictable or 0) + (s.kv_available or 0)),
        )
        hit_pts = self._pts("hit")
        q("#c-hit").set(f"{fmt_pct(s.hit_rate)}/{fmt_pct(max(hit_pts, default=0.0))}", hit_pts)
        if s.hicache_total:
            q("#g-hicache").set((s.hicache_used or 0) / s.hicache_total, f"{fmt_num(s.hicache_used)}/{fmt_num(s.hicache_total)}")
        else:
            q("#g-hicache").set(None, "off")

        q("#c-ttft").set(f"{fmt_secs(s.ttft_p50)}/{fmt_secs(s.ttft_p99)}", self._pts("ttft"))
        q("#c-e2e").set(f"{fmt_secs(s.e2e_p50)}/{fmt_secs(s.e2e_p99)}", self._pts("e2e"), color=MAGENTA)
        q("#c-itl").set(fmt_secs(s.itl_p50), self._pts("itl"), color=YELLOW)
        q("#c-queue").set(fmt_secs(s.queue_p50), self._pts("queue"), color=GREEN)

        q("#s-mw").set(f"{s.mem_weights_gb:.1f} GB" if s.mem_weights_gb is not None else "--")
        q("#s-mkv").set(f"{s.mem_kv_gb:.1f} GB" if s.mem_kv_gb is not None else "--")
        q("#s-mg").set(f"{s.mem_graph_gb:.2f} GB" if s.mem_graph_gb is not None else "--")
        q("#s-util").set(fmt_pct(s.utilization), "cg:" + ("on" if (s.is_cuda_graph or 0) > 0 else "off"))

        q("#g-spec").set(s.spec_accept_rate, fmt_pct(s.spec_accept_rate))
        q("#s-slen").set(fmt_num(s.spec_accept_length, 2))
        q("#s-steps").set(fmt_num(s.spec_steps, 0), f"≈{(s.spec_steps or 0) + 1:.0f}x" if s.spec_steps else "")

    def _tick_clock(self) -> None:
        if self.last_snapshot is not None:
            self._apply(self.last_snapshot)

    def _update_footer(self) -> None:
        self.query_one("#footer").update(
            f"[b #00d7ff]^Q[/] quit   [b #00d7ff]+/-[/] refresh {self.refresh_interval:.1f}s   "
            f"[b #00d7ff]h[/] window {self.history_window}s   [b #00d7ff]?[/] help"
        )

    def action_faster(self) -> None:
        self.refresh_interval = max(0.5, round(self.refresh_interval - 0.5, 1))
        self._update_footer()

    def action_slower(self) -> None:
        self.refresh_interval = min(5.0, round(self.refresh_interval + 0.5, 1))
        self._update_footer()

    def action_window(self) -> None:
        self.history_window = 30 if self.history_window == 90 else 90
        self._update_footer()

    def action_help(self) -> None:
        self.push_screen(HelpPanel())
