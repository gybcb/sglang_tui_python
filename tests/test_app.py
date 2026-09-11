import asyncio

import httpx
import pytest

from sglang_tui.app import HelpPanel, SglangTui
from sglang_tui.prom import Differ, build_snapshot, parse
from sglang_tui.widgets import ChartRow


@pytest.mark.asyncio
async def test_running_queue_rendered_as_charts():
    fixture = open("tests/fixtures/metrics_sample.txt").read()
    app = SglangTui(url="http://127.0.0.1:1")
    async with app.run_test(size=(100, 30)) as pilot:
        samples = parse(fixture)
        differ = Differ()
        differ.update(samples, 0.0)
        snap = build_snapshot(samples, differ, app.base_url, "test-model", connected=True)
        app.histories["run"].push(snap.running)
        app.histories["qcnt"].push(snap.queue)
        app.last_snapshot = snap
        app._apply(snap)
        await pilot.pause()
        run = app.query_one("#c-run")
        que = app.query_one("#c-que")
        assert isinstance(run, ChartRow) and isinstance(que, ChartRow)
        assert run.points == [2.0] and que.points == [3.0]
        assert run.value == "2/2" and que.value == "3/3"


@pytest.mark.asyncio
async def test_running_value_shows_current_over_history_max():
    app = SglangTui(url="http://127.0.0.1:1")
    async with app.run_test(size=(100, 30)) as pilot:
        app.histories["run"].push(8.0)
        snap = build_snapshot({}, Differ(), app.base_url, "test-model", connected=True)
        snap.running = 2.0
        snap.queue = 0.0
        app.histories["run"].push(snap.running)
        app.histories["qcnt"].push(snap.queue)
        app.last_snapshot = snap
        app._apply(snap)
        await pilot.pause()
        assert app.query_one("#c-run").value == "2/8"
        assert app.query_one("#c-que").value == "0/0"


@pytest.mark.asyncio
async def test_help_panel_opens_and_closes():
    app = SglangTui(url="http://127.0.0.1:1")
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert len(app.screen_stack) == 1
        await pilot.press("question_mark")
        await pilot.pause()
        assert isinstance(app.screen, HelpPanel)
        assert len(app.screen_stack) == 2
        await pilot.press("escape")
        await pilot.pause()
        assert len(app.screen_stack) == 1
        # reopen and close with q
        await pilot.press("question_mark")
        await pilot.pause()
        assert len(app.screen_stack) == 2
        await pilot.press("q")
        await pilot.pause()
        assert len(app.screen_stack) == 1


@pytest.mark.asyncio
async def test_fetch_falls_back_to_curl(monkeypatch):
    fixture = open("tests/fixtures/metrics_sample.txt").read()
    app = SglangTui(url="http://127.0.0.1:1")

    class DeadClient:
        async def get(self, url):
            raise httpx.ConnectError("[Errno 65] No route to host")

    class FakeProc:
        returncode = 0

        async def communicate(self):
            return fixture.encode(), b""

    async def fake_exec(*args, **kwargs):
        assert args[0] == "curl"
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    text = await app._fetch_metrics(DeadClient())
    assert app._prefer_curl is True
    assert "sglang:num_running_reqs" in text


@pytest.mark.asyncio
async def test_fetch_both_channels_fail(monkeypatch):
    app = SglangTui(url="http://127.0.0.1:1")

    class DeadClient:
        async def get(self, url):
            raise httpx.ConnectError("nope")

    class FailProc:
        returncode = 7

        async def communicate(self):
            return b"", b"connection refused"

    async def fake_exec(*args, **kwargs):
        return FailProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    with pytest.raises(RuntimeError, match="curl exit 7"):
        await app._fetch_metrics(DeadClient())
    # both failed: reset preference so httpx gets retried next poll
    assert app._prefer_curl is False
