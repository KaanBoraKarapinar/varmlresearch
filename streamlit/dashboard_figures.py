"""Plotly figures of the Streamlit dashboard (data in, figure out; no Streamlit import)."""

from __future__ import annotations

import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dashboard_data  # noqa: F401  (puts the project root on sys.path)
from core.units import plot_format

TEMPLATE = "plotly_white"
COLORS = pc.qualitative.Plotly
# the stacked-area palette: the D3 categorical colours (blue, orange, green, red, purple, brown, pink, grey, olive, cyan). A series gets the slot of
# its position and never a generated colour; the series after the last slot are folded into "Other" (OTHER_COLOR)
CATEGORICAL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
OTHER_COLOR = "#d0d0d0"
# the three series of every variance chart (model, the EWMA, the squared return) keep the same color everywhere
MODEL_COLOR, EWMA_COLOR, R2_COLOR = "#1f5fbf", "#ff7f0e", "#9aa0a6"


def _layout(fig: go.Figure, title: str, y_title: str, log_y: bool = False, range_slider: bool = True) -> go.Figure:
    fig.update_layout(
        title=title, template=TEMPLATE, hovermode="x unified", yaxis_title=y_title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=480,
    )
    fig.update_yaxes(type="log" if log_y else "linear")
    if range_slider:
        fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06))
    return fig


def timeline_figure(spans: dict[str, dict]) -> go.Figure:
    """One row per run: the first train window (light) and the forecast span (dark)."""
    rows = []
    for label, s in spans.items():
        rows.append({"run": label, "part": "first train window", "start": s["first_train_start"], "end": s["first_train_end"]})
        rows.append({"run": label, "part": "forecast days", "start": s["first_forecast"], "end": s["last_forecast"]})
    fig = px.timeline(
        pd.DataFrame(rows), x_start="start", x_end="end", y="run", color="part", template=TEMPLATE,
        color_discrete_map={"first train window": "#b8c4d6", "forecast days": "#1f5fbf"},
    )
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_layout(height=120 + 38 * len(spans), title="Time covered by each run", legend_title="")
    return fig


def var_curves_figure(var_dfs: dict[str, pd.DataFrame], title: str) -> go.Figure:
    """Every run's VaR over the realized loss (grey), same days."""
    fig = go.Figure()
    first = next(iter(var_dfs.values()))
    fig.add_trace(go.Scatter(
        x=first.index, y=first["loss"], mode="lines", name="realized loss", line=dict(color="rgba(120,120,120,0.55)", width=1),
    ))
    for i, (label, v) in enumerate(var_dfs.items()):
        fig.add_trace(go.Scatter(x=v.index, y=v["varForecast"], mode="lines", name=label, line=dict(color=COLORS[i % len(COLORS)], width=1.6)))
    fig.update_yaxes(tickformat=plot_format()["tickformat"], ticksuffix=plot_format()["ticksuffix"])
    return _layout(fig, title, "VaR / loss")


def pairwise_matrix_figure(pairwise) -> go.Figure:
    """
    The pairwise tick-loss matrix (core.pairwise_tick_loss_matrix): the cell of a row run against a column run is green
    when the row is significantly better, red when significantly worse, gray when they cannot be told apart. The number is
    the mean tick-loss difference (row minus column) in 1e-4 units, negative = the row costs less.
    """
    names = list(pairwise.mean_difference.index)
    code = pairwise.verdict.replace({"better": 1.0, "worse": -1.0, "same": 0.0, "": float("nan")}).astype(float)
    text = (pairwise.mean_difference * 1e4).round(2).astype(str)
    for name in names:
        text.loc[name, name] = ""
    fig = go.Figure(go.Heatmap(
        z=code.to_numpy(), x=names, y=names, text=text.to_numpy(), texttemplate="%{text}", xgap=2, ygap=2,
        colorscale=[[0, "#ef9a9a"], [0.5, "#eceff1"], [1, "#a5d6a7"]], zmin=-1, zmax=1, showscale=False,
        customdata=pairwise.p_value.to_numpy(),
        hovertemplate="row: %{y}<br>column: %{x}<br>Holm-corrected p: %{customdata:.3f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top", tickangle=-30)
    fig.update_layout(template=TEMPLATE, height=140 + 60 * len(names), title="Row run against column run (green = row significantly better)", margin=dict(t=170))
    return fig


def variance_runs_figure(runs: pd.DataFrame, baselines: pd.DataFrame, log_y: bool, title: str) -> go.Figure:
    """Several runs' variances (one solid color each) with the chosen baselines in their fixed styles (dashed / dotted)."""
    styles = {
        "EWMA (reference run)": dict(color=EWMA_COLOR, width=1.8, dash="dash"), "EWMA variance": dict(color=EWMA_COLOR, width=1.8, dash="dash"),
        "squared return r²": dict(color=R2_COLOR, width=1), "return^2": dict(color=R2_COLOR, width=1),
        "realized variance (21-day mean of r²)": dict(color="#9467bd", width=1.8, dash="dashdot"),
    }
    palette = [c for c in COLORS if c.lower() != EWMA_COLOR]
    fig = go.Figure()
    for i, column in enumerate(runs.columns):
        fig.add_trace(go.Scatter(x=runs.index, y=runs[column], mode="lines", name=column, line=dict(color=palette[i % len(palette)], width=1.8)))
    for column in baselines.columns:
        fig.add_trace(go.Scatter(x=baselines.index, y=baselines[column], mode="lines", name=column, line=styles.get(column, dict(color="#8c564b", width=1.5, dash="dash"))))
    return _layout(fig, title, "variance", log_y)


def cumulative_tick_loss_figure(lines: dict[str, pd.Series]) -> go.Figure:
    """Running sums of the daily tick loss, one line per run (blue for the first)."""
    fig = go.Figure()
    for i, (name, series) in enumerate(lines.items()):
        fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name=name, line=dict(color=COLORS[(i + 0) % len(COLORS)] if i else MODEL_COLOR, width=2.2)))
    return _layout(fig, "Cumulative tick loss", "cumulative tick loss")


def lines_figure(frame: pd.DataFrame, title: str, y_title: str, log_y: bool = False, dash_last: str | None = None) -> go.Figure:
    """Columns as lines; the column named dash_last (the EWMA reference) is drawn dashed black."""
    fig = go.Figure()
    for i, column in enumerate(frame.columns):
        is_ref = column == dash_last
        fig.add_trace(go.Scatter(
            x=frame.index, y=frame[column], mode="lines", name=column,
            line=dict(color="black" if is_ref else COLORS[i % len(COLORS)], width=1.6, dash="dash" if is_ref else "solid"),
        ))
    return _layout(fig, title, y_title, log_y)


def variance_vs_baselines_figure(frame: pd.DataFrame, log_y: bool) -> go.Figure:
    """The model's predicted variance (blue) against the chosen baselines (dotted)."""
    styles = {   # one fixed, clearly different color per series (the same ones in every chart of the dashboard)
        "EWMA (reference run)": dict(color=EWMA_COLOR, width=1.8, dash="dash"),
        "squared return r²": dict(color=R2_COLOR, width=1),
        "target-definition variance": dict(color="#2ca02c", width=1.7, dash="dot"),
        "realized variance (21-day mean of r²)": dict(color="#9467bd", width=1.8, dash="dashdot"),
    }
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame.index, y=frame["model"], mode="lines", name="model", line=dict(color=MODEL_COLOR, width=2.2)))
    for column in (c for c in frame.columns if c != "model"):
        fig.add_trace(go.Scatter(x=frame.index, y=frame[column], mode="lines", name=column, line=styles.get(column, dict(color="#8c564b", width=1.5, dash="dash"))))
    return _layout(fig, "Predicted variance vs realized variance", "variance", log_y)


def day_variance_figure(frame: pd.DataFrame, day, log_y: bool) -> go.Figure:
    """Portfolio variance around one day: the model (blue), the EWMA (dashed black) and the squared return (grey), the day dotted."""
    styles = {
        "predicted variance": dict(color=MODEL_COLOR, width=2.2), "EWMA variance": dict(color=EWMA_COLOR, width=1.8, dash="dash"),
        "return²": dict(color=R2_COLOR, width=1),
    }
    fig = go.Figure()
    for column in ("return²", "EWMA variance", "predicted variance"):
        if column in frame:
            fig.add_trace(go.Scatter(
                x=frame.index, y=frame[column], mode="lines+markers", name=column, line=styles[column], marker=dict(size=5),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>" + column + ": %{y:.3e}<extra></extra>",
            ))
    fig.add_vline(x=pd.Timestamp(day), line_dash="dot", line_color="gray")
    fig.update_xaxes(tickformat="%b %d")
    return _layout(fig, f"Portfolio variance around {pd.Timestamp(day).date()}", "variance", log_y, range_slider=False)


def loglog_scatter_figure(variance: pd.DataFrame, reference: str) -> go.Figure:
    """Each run's variance against the reference variance, every day one point (log-log, diagonal = equal)."""
    fig = go.Figure()
    ref = variance[reference]
    for i, column in enumerate(variance.columns):
        if column == reference:
            continue
        fig.add_trace(go.Scattergl(
            x=ref, y=variance[column], mode="markers", name=column,
            marker=dict(size=4, opacity=0.45, color=COLORS[i % len(COLORS)]),
            text=variance.index.strftime("%Y-%m-%d"), hovertemplate="%{text}<br>EWMA %{x:.2e}<br>run %{y:.2e}<extra>" + column + "</extra>",
        ))
    low, high = float(variance.min().min()), float(variance.max().max())
    fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="equal", line=dict(color="black", dash="dot")))
    fig.update_layout(
        title=f"Predicted variance vs {reference}", template=TEMPLATE, height=520,
        xaxis=dict(type="log", title=f"{reference} variance"), yaxis=dict(type="log", title="run variance"),
    )
    return fig


def var_vs_loss_figure(daily: pd.DataFrame, title: str) -> go.Figure:
    """One run: VaR, loss, exceedance markers and a vertical line where every window starts."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.index, y=daily["loss"], mode="lines", name="realized loss", line=dict(color="rgba(120,120,120,0.6)", width=1)))
    fig.add_trace(go.Scatter(x=daily.index, y=daily["varForecast"], mode="lines", name="VaR", line=dict(color="#1f5fbf", width=1.8)))
    hit = daily[daily["exceeded"]]
    fig.add_trace(go.Scatter(
        x=hit.index, y=hit["loss"], mode="markers", name="exceedance", marker=dict(color="crimson", size=7, symbol="x"),
        customdata=hit[["varForecast", "excess", "window_id"]], hovertemplate="%{x|%Y-%m-%d}<br>loss %{y:.2%}<br>VaR %{customdata[0]:.2%}<br>excess %{customdata[1]:.2%}<br>window %{customdata[2]}<extra></extra>",
    ))
    starts = daily.groupby("window_id").head(1).index
    for date in starts[1:]:
        fig.add_vline(x=date, line_width=1, line_dash="dot", line_color="rgba(0,0,0,0.25)")
    fig.update_yaxes(tickformat=".1%")
    return _layout(fig, title, "VaR / loss")


def ticker_variance_figure(frames: dict[str, pd.DataFrame], log_y: bool, show_r2: bool) -> go.Figure:
    """One panel per ticker: the model's variance against the EWMA's own variance (and r^2 if asked)."""
    tickers = list(frames)
    # plotly needs the gap between two panels to be below 1 / (rows - 1); with many tickers the panels are also lower
    fig = make_subplots(rows=len(tickers), cols=1, shared_xaxes=True, subplot_titles=tickers, vertical_spacing=min(0.05, 0.8 / max(len(tickers) - 1, 1)))
    for row, ticker in enumerate(tickers, start=1):
        f = frames[ticker]
        if show_r2:
            fig.add_trace(go.Scatter(x=f.index, y=f["return^2"], mode="lines", name="return^2", legendgroup="r2", showlegend=row == 1,
                                     line=dict(color=R2_COLOR, width=1)), row=row, col=1)
        fig.add_trace(go.Scatter(x=f.index, y=f["model variance"], mode="lines", name="model variance", legendgroup="m", showlegend=row == 1,
                                 line=dict(color=MODEL_COLOR, width=2.2)), row=row, col=1)
        fig.add_trace(go.Scatter(x=f.index, y=f["EWMA variance"], mode="lines", name="EWMA variance", legendgroup="e", showlegend=row == 1,
                                 line=dict(color=EWMA_COLOR, width=1.8, dash="dash")), row=row, col=1)
    fig.update_yaxes(type="log" if log_y else "linear")
    fig.update_layout(template=TEMPLATE, height=(260 if len(tickers) <= 6 else 160) * len(tickers) + 80, hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), title="Variance per ticker")
    return fig


def heatmap_figure(frame: pd.DataFrame, title: str) -> go.Figure:
    fig = px.imshow(frame.T, aspect="auto", color_continuous_scale="Blues", labels=dict(x="window", y="", color=""), template=TEMPLATE)
    fig.update_layout(title=title, height=60 + 28 * len(frame.columns))
    return fig


def bar_figure(series: pd.Series, title: str, y_title: str) -> go.Figure:
    fig = px.bar(series, template=TEMPLATE, labels={"index": "", "value": y_title})
    fig.update_layout(title=title, showlegend=False, height=380)
    return fig
