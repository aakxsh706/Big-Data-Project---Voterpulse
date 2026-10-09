"""Plotly figures. Charts support the plain-language statements; they never replace them.

Colour roles: the daily distribution is a *diverging* scale — green (positive) and red
(negative) poles with a neutral grey midpoint — always shown with a text legend and an
optional table view. Single-series charts use one ink colour and no legend.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

POSITIVE = "#1E7B46"
NEGATIVE = "#B42318"
NEUTRAL = "#98A2B3"
INK = "#1F4E79"
TEXT = "#344054"
MUTED = "#667085"
GRID = "#EAECF0"
FONT = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def _layout(fig: go.Figure, height: int = 300, showlegend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=TEXT),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, size=13, color=TEXT), bordercolor=GRID),
        showlegend=showlegend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=TEXT)),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    return fig


def tone_trend_chart(series: pd.DataFrame) -> go.Figure:
    """Daily average compound score (-1..+1) with the neutral band marked."""
    fig = go.Figure()
    fig.add_hrect(y0=-0.05, y1=0.05, fillcolor=GRID, opacity=0.6, line_width=0, layer="below")
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    if not series.empty:
        fig.add_trace(go.Scatter(
            x=series["day"], y=series["mean_compound"], mode="lines+markers",
            line=dict(color=INK, width=2), marker=dict(size=8, color=INK, line=dict(color="white", width=2)),
            customdata=series[["headlines"]],
            hovertemplate="<b>%{x|%d %b}</b><br>Average tone: %{y:+.2f}<br>Headlines: %{customdata[0]}<extra></extra>",
            name="Average tone",
        ))
    fig.update_yaxes(range=[-1, 1], tickvals=[-1, -0.5, 0, 0.5, 1],
                     ticktext=["−1 negative", "−0.5", "0 neutral", "+0.5", "+1 positive"])
    fig.update_xaxes(tickformat="%d %b")
    return _layout(fig, 280)


def distribution_chart(series: pd.DataFrame) -> go.Figure:
    """Stacked daily counts: negative at the base, neutral middle, positive on top."""
    fig = go.Figure()
    for label, color in (("Negative", NEGATIVE), ("Neutral", NEUTRAL), ("Positive", POSITIVE)):
        if series.empty:
            continue
        fig.add_trace(go.Bar(
            x=series["day"], y=series[label], name=label, marker=dict(color=color, line=dict(color="white", width=2)),
            hovertemplate=f"<b>%{{x|%d %b}}</b><br>{label}: %{{y}} headlines<extra></extra>",
        ))
    fig.update_layout(barmode="stack", bargap=0.35)
    fig.update_xaxes(tickformat="%d %b")
    fig.update_yaxes(title=dict(text="Headlines", font=dict(color=MUTED, size=12)))
    return _layout(fig, 280, showlegend=True)


def sources_chart(sources: list[tuple[str, int]], top: int = 10) -> go.Figure:
    data = sources[:top][::-1]
    fig = go.Figure(go.Bar(
        x=[n for _, n in data], y=[s for s, _ in data], orientation="h",
        marker=dict(color=INK), text=[n for _, n in data], textposition="outside",
        textfont=dict(color=TEXT), cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x} headlines<extra></extra>",
    ))
    fig.update_xaxes(showticklabels=False, gridcolor="rgba(0,0,0,0)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=TEXT))
    return _layout(fig, max(160, 32 * len(data) + 40))


def method_breakdown_chart(methods: pd.DataFrame) -> go.Figure:
    grouped = methods.groupby("method", as_index=False)["headlines"].sum()
    fig = go.Figure(go.Bar(
        x=grouped["method"], y=grouped["headlines"], marker=dict(color=INK),
        text=grouped["headlines"], textposition="outside", textfont=dict(color=TEXT), cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>%{y} headlines<extra></extra>",
    ))
    return _layout(fig, 260)
