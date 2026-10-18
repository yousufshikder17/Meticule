"""Headless Matplotlib renderer for Meticule reports (Phase 5).

It draws one of five chart primitives (lines, bars, heatmap, scatter, histogram) from numbers and short labels that the
platform computed. Nothing in a request is executed or interpreted as markup: mathtext parsing is off, text is stripped
of control characters and truncated, every number must be finite and every size is bounded. Output is PNG or SVG with
fixed metadata, so the same request renders the same bytes in the same environment.
"""
import base64
import io
import math
import re
import sys

import ml_runtime_common as common
from ml_runtime_common import RunnerError

MAX_POINTS, MAX_SERIES, MAX_CATEGORIES, MAX_CELLS, MAX_BYTES = 20_000, 12, 50, 2_500, 2 * 1024 * 1024
CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def bad(message):
    raise RunnerError("invalid_report_data", message)


def text(value, limit=80):
    if not isinstance(value, str):
        bad("Labels must be strings")
    return CONTROL.sub(" ", value)[:limit]


def numbers(values, limit=MAX_POINTS):
    if not isinstance(values, list) or len(values) > limit or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        bad("Values must be lists of finite numbers within the size limit")
    return [float(v) for v in values]


def series_list(items):
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_SERIES:
        bad("Between 1 and 12 series are required")
    return items


def draw(ax, chart):
    kind = chart.get("type")
    if kind == "lines":
        total = 0
        for s in series_list(chart["series"]):
            x, y = numbers(s["x"]), numbers(s["y"])
            total += len(x)
            if len(x) != len(y) or total > MAX_POINTS:
                bad("Series x and y must have equal, bounded lengths")
            style = s.get("style", "line")
            if style == "points":
                ax.plot(x, y, "o", markersize=3, label=text(s["label"]))
            elif style == "step":
                ax.step(x, y, where="post", label=text(s["label"]))
            elif style == "dashed":
                ax.plot(x, y, "--", label=text(s["label"]))
            else:
                ax.plot(x, y, label=text(s["label"]))
        if chart.get("diagonal"):
            ax.plot([0, 1], [0, 1], ":", color="grey", linewidth=1)
    elif kind == "bars":
        categories = [text(c, 40) for c in chart["categories"]]
        series = series_list(chart["series"])
        if not 1 <= len(categories) <= MAX_CATEGORIES:
            bad("Between 1 and 50 categories are required")
        width = 0.8 / len(series)
        for i, s in enumerate(series):
            values = numbers(s["values"])
            if len(values) != len(categories):
                bad("Each bar series needs one value per category")
            ax.bar([j + i * width - 0.4 + width / 2 for j in range(len(categories))], values, width, label=text(s["label"]))
        ax.set_xticks(range(len(categories)), categories, rotation=30 if len(categories) > 6 else 0, ha="right" if len(categories) > 6 else "center")
    elif kind == "heatmap":
        matrix = chart["matrix"]
        if not isinstance(matrix, list) or not matrix or len(matrix) * len(matrix[0]) > MAX_CELLS:
            bad("Heatmap must be a bounded, non-empty matrix")
        rows = [numbers(r) for r in matrix]
        if any(len(r) != len(rows[0]) for r in rows):
            bad("Heatmap rows must have equal lengths")
        x_ticks, y_ticks = [text(t, 30) for t in chart["xTicks"]], [text(t, 30) for t in chart["yTicks"]]
        if len(x_ticks) != len(rows[0]) or len(y_ticks) != len(rows):
            bad("Heatmap ticks must match its shape")
        image = ax.imshow(rows, cmap="Blues", aspect="auto")
        ax.figure.colorbar(image, ax=ax)
        ax.set_xticks(range(len(x_ticks)), x_ticks)
        ax.set_yticks(range(len(y_ticks)), y_ticks)
        if chart.get("annotate") and len(rows) * len(rows[0]) <= 400:
            peak = max(max(r) for r in rows) or 1
            for i, r in enumerate(rows):
                for j, v in enumerate(r):
                    ax.text(j, i, f"{v:g}", ha="center", va="center", color="white" if v > peak / 2 else "black", fontsize=8)
    elif kind == "scatter":
        x, y = numbers(chart["x"]), numbers(chart["y"])
        if len(x) != len(y):
            bad("Scatter x and y must have equal lengths")
        ax.scatter(x, y, s=8)
        if "hline" in chart:
            ax.axhline(numbers([chart["hline"]])[0], color="grey", linestyle=":", linewidth=1)
    elif kind == "histogram":
        bins = chart.get("bins", 30)
        if not isinstance(bins, int) or not 2 <= bins <= 100:
            bad("Histogram bins must be 2..100")
        total = 0
        for s in series_list(chart["series"]):
            values = numbers(s["values"])
            total += len(values)
            if total > MAX_POINTS:
                bad("Histogram data exceeds the size limit")
            if values:
                ax.hist(values, bins=bins, alpha=0.6, label=text(s["label"]))
    else:
        bad("Unknown chart type")


def render(request):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"text.parse_math": False, "text.usetex": False, "svg.hashsalt": "meticule", "svg.fonttype": "path", "font.family": "DejaVu Sans"})
    fmt, width, height = request["format"], request["width"], request["height"]
    if fmt not in ("png", "svg") or not (isinstance(width, int) and 320 <= width <= 1600 and isinstance(height, int) and 240 <= height <= 1200):
        bad("Unsupported format or size")
    chart = request["chart"]
    figure, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    try:
        draw(ax, chart)
        ax.set_title(text(request["title"], 100))
        ax.set_xlabel(text(chart.get("xLabel", "")))
        ax.set_ylabel(text(chart.get("yLabel", "")))
        if chart.get("logY"):
            ax.set_yscale("log")
        if chart["type"] in ("lines", "bars", "histogram") and len(chart.get("series", [])) > 1:
            ax.legend(fontsize=8)
        ax.grid(chart["type"] != "heatmap", alpha=0.3)
        figure.tight_layout()
        buffer = io.BytesIO()
        metadata = {"Software": None} if fmt == "png" else {"Date": None, "Creator": None}
        figure.savefig(buffer, format=fmt, dpi=100, metadata=metadata)
    finally:
        plt.close(figure)
    data = buffer.getvalue()
    if len(data) > MAX_BYTES:
        bad("Rendered report exceeds the size limit")
    return {"image": base64.b64encode(data).decode(), "matplotlib": matplotlib.__version__, "bytes": len(data)}


def execute(request):
    if request["command"] == "render":
        try:
            return render(request)
        except (KeyError, TypeError, AttributeError):
            bad("Malformed chart")
    raise ValueError("Unknown report command")


if __name__ == "__main__":
    sys.exit(common.serve(execute))
