"""btop-style widgets: segmented gauges, stacked bars, labeled stat/chart rows."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from rich.text import Text
from textual.widget import Widget

FILL = "█"
EMPTY = "░"

GREEN = "#5faf00"
YELLOW = "#ffd75f"
RED = "#ff5f5f"
CYAN = "#00d7ff"
MAGENTA = "#d75f87"
BLUE = "#5f87ff"
DIM = "#5f8787"
FG = "#d7d7d7"


def fmt_num(v: Optional[float], dec: int = 1, suffix: str = "") -> str:
    if v is None or math.isnan(v):
        return "--"
    if v >= 1e9:
        return f"{v / 1e9:.2f}G{suffix}"
    if v >= 1e6:
        return f"{v / 1e6:.2f}M{suffix}"
    if v >= 1e4:
        return f"{v / 1e3:.1f}k{suffix}"
    if v >= 100:
        return f"{v:,.0f}{suffix}"
    return f"{v:.{dec}f}{suffix}"


def fmt_pct(v: Optional[float]) -> str:
    if v is None or math.isnan(v):
        return "--%"
    return f"{v * 100:.1f}%"


def fmt_secs(v: Optional[float]) -> str:
    if v is None or math.isnan(v):
        return "--"
    if v >= 1.0:
        return f"{v:.2f}s"
    return f"{v * 1000:.0f}ms"


def heat_color(ratio: float) -> str:
    if ratio < 0.5:
        return GREEN
    if ratio < 0.75:
        return YELLOW
    return RED


class Gauge(Widget):
    """btop-style segmented bar: LABEL ████████░░░░ right-text"""

    def __init__(self, label: str, **kw) -> None:
        super().__init__(**kw)
        self.label = label
        self.ratio: Optional[float] = None
        self.text_right: str = "--"
        self.fixed_color: Optional[str] = None

    def set(self, ratio: Optional[float], text_right: str = "--") -> None:
        if ratio is not None and math.isnan(ratio):
            ratio = None
        self.ratio = None if ratio is None else min(1.0, max(0.0, ratio))
        self.text_right = text_right
        self.refresh()

    def render(self) -> Text:
        w = max(self.size.width, 14)
        label = self.label[:10].ljust(11)
        right_w = max(len(self.text_right), 5)
        bar_w = max(w - len(label) - right_w - 1, 5)
        if self.ratio is None:
            return Text.assemble((label, f"bold {CYAN}"), (f"{'--':<{bar_w}} {self.text_right:>{right_w}}", DIM))
        filled = int(round(self.ratio * bar_w))
        color = self.fixed_color or heat_color(self.ratio)
        return Text.assemble(
            (label, f"bold {CYAN}"),
            (FILL * filled, f"bold {color}"),
            (EMPTY * (bar_w - filled), DIM),
            (f" {self.text_right:>{right_w}}", FG),
        )


class StackedGauge(Widget):
    """Multi-segment bar, e.g. KV pool: active | evictable | free."""

    def __init__(self, label: str, **kw) -> None:
        super().__init__(**kw)
        self.label = label
        self.parts: List[Tuple[float, str]] = []
        self.text_right: str = "--"

    def set(self, parts: List[Tuple[Optional[float], str]], text_right: str = "--") -> None:
        vals = [0.0 if p is None or math.isnan(p) else max(0.0, p) for p, _ in parts]
        total = sum(vals) or 1.0
        self.parts = [(v / total, c) for v, (_, c) in zip(vals, parts)]
        self.text_right = text_right
        self.refresh()

    def render(self) -> Text:
        w = max(self.size.width, 12)
        label = self.label[:10].ljust(11)
        right_w = max(len(self.text_right), 5)
        bar_w = max(w - len(label) - right_w - 1, 5)
        counts = [int(round(r * bar_w)) for r, _ in self.parts]
        if counts and sum(counts) < bar_w:
            counts[-1] += bar_w - sum(counts)
        text = Text()
        text.append(label, style=f"bold {CYAN}")
        for count, (_, color) in zip(counts, self.parts):
            text.append(FILL * count, style=f"bold {color}")
        text.append(f" {self.text_right:>{right_w}}", style=FG)
        return text


class StatRow(Widget):
    """LABEL  value  [secondary]"""

    def __init__(self, label: str, **kw) -> None:
        super().__init__(**kw)
        self.label = label
        self.value: str = "--"
        self.value2: str = ""
        self.value_color: str = FG

    def set(self, value: str, value2: str = "", color: str = FG) -> None:
        self.value = value
        self.value2 = value2
        self.value_color = color
        self.refresh()

    def render(self) -> Text:
        w = max(self.size.width, 12)
        label = self.label[:10].ljust(11)
        right = f"  {self.value2}" if self.value2 else ""
        val_w = max(w - len(label) - len(right), 6)
        return Text.assemble(
            (label, FG),
            (self.value[: val_w - 1].rjust(val_w - 1), f"bold {self.value_color}"),
            (right, DIM),
        )


class ChartRow(Widget):
    """Single row: 'LABEL  ▅▃▄▆▅▂  value' — btop inline CPU-graph style."""

    BLOCKS = "▁▂▃▄▅▆▇█"

    def __init__(self, label: str, color: str = CYAN, **kw) -> None:
        super().__init__(**kw)
        self.label = label
        self.value: str = "--"
        self.value_color: str = FG
        self.points: List[float] = []
        self.chart_color = color

    def set(self, value: str, points: List[float], color: Optional[str] = None, value_color: str = FG) -> None:
        self.value = value
        self.points = points
        self.value_color = value_color
        if color:
            self.chart_color = color
        self.refresh()

    def render(self) -> Text:
        w = max(self.size.width, 16)
        label = self.label[:10].ljust(11)
        val_w = max(len(self.value) + 1, 10)
        cols = max(w - len(label) - val_w, 8)
        text = Text()
        text.append(label, style=f"bold {CYAN}")
        text.append(self._sparkline(cols), style=f"bold {self.chart_color}")
        text.append(" " + self.value.rjust(val_w - 1), style=f"bold {self.value_color}")
        return text

    def _sparkline(self, cols: int) -> str:
        pts = list(self.points) or [0.0]
        n = len(pts)
        if n > cols:  # resample down
            idx = [round(i * (n - 1) / (cols - 1)) for i in range(cols)]
            pts = [pts[j] for j in idx]
        elif n < cols:
            pts = [0.0] * (cols - n) + pts
        hi = max(pts)
        if hi <= 0:
            return self.BLOCKS[0] * cols
        return "".join(self.BLOCKS[min(7, int(p / hi * 7.999))] for p in pts)
