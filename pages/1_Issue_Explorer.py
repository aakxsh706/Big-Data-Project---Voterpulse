"""Issue details: mood explanation first, charts second, source headlines last."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from html import escape  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from voterpulse.dashboard import charts  # noqa: E402
from voterpulse.dashboard import components as ui  # noqa: E402
from voterpulse.dashboard import data_access as da  # noqa: E402
from voterpulse.sentiment import aggregator as agg  # noqa: E402

cfg = ui.page_setup("Issue Explorer")
now = ui.now_utc()
snap = ui.get_snapshot(cfg)
ui.render_demo_banner(cfg, snap)

names = cfg.issues.issue_names
default = st.session_state.get("selected_issue", names[0])
st.title("Issue Explorer")
c1, c2 = st.columns([2, 3], vertical_alignment="bottom")
with c1:
    issue_name = st.selectbox("Issue", names, index=names.index(default) if default in names else 0)
    st.session_state["selected_issue"] = issue_name
with c2:
    window = ui.period_selector(snap, now, key="explorer_period", allow_custom=True)

issue = cfg.issues.by_name(issue_name)
view = da.issue_view(cfg, snap, issue, window, now)
r = view.report

# ---- 1. Plain-language explanation (primary content)
st.markdown(
    f'<div class="vp-card" style="border-top:5px solid {r.color};min-height:0">'
    f'<div class="vp-card-issue">{escape(issue.name)}</div>{ui.mood_badge(r)}'
    f'<div class="vp-big-mood" style="color:#10243E">Mood: {escape(r.headline)}</div>'
    + "".join(f'<div class="vp-meta" style="font-size:1rem;color:#344054">{escape(s)}</div>' for s in r.explanation)
    + f'<div class="vp-meta" style="margin-top:.6rem">{escape(r.sample)}</div>'
    f'<div class="vp-meta">{escape(r.freshness)}</div></div>',
    unsafe_allow_html=True,
)
st.caption(f"What this issue covers: {issue.description}")
ui.render_disclaimer()

if view.summary.count == 0:
    st.info("There are no headlines for this issue in the selected period.", icon="ℹ️")
    st.stop()

# ---- 2. Headline counts
m1, m2, m3, m4 = st.columns(4)
m1.metric("Headlines", view.summary.count)
m2.metric("Read as negative", f"{round(view.summary.negative_share * 100)}%")
m3.metric("Read as positive", f"{round(view.summary.positive_share * 100)}%")
m4.metric("Neutral", f"{round(view.summary.neutral_share * 100)}%")

# ---- 3. Charts (secondary)
series = agg.daily_series(snap.frame, issue.name, window)
left, right = st.columns(2, gap="large")
with left:
    st.markdown("**Average tone per day**")
    st.caption("Each point is the average headline score for one day (IST). The grey band is the neutral zone.")
    st.plotly_chart(charts.tone_trend_chart(series), width="stretch", config={"displayModeBar": False})
with right:
    st.markdown("**Positive, neutral and negative headlines per day**")
    st.caption("Bar height is the number of headlines that day.")
    st.plotly_chart(charts.distribution_chart(series), width="stretch", config={"displayModeBar": False})

t = view.trend
if t.sufficient:
    st.caption(f"Trend check: last 7 complete days averaged {t.current_mean:+.2f} ({t.current_count} headlines); "
               f"the 7 days before averaged {t.previous_mean:+.2f} ({t.previous_count} headlines). "
               "Today's incomplete day is not part of the trend.")
else:
    st.caption(f"Trend check needs at least {cfg.thresholds.trend.min_headlines_each_window} headlines in each of "
               f"the two weeks compared (found {t.current_count} and {t.previous_count}).")

with st.expander("Show the numbers behind these charts"):
    table = series.rename(columns={"day": "Day (IST)", "headlines": "Headlines", "mean_compound": "Average tone"})
    st.dataframe(table, hide_index=True, width="stretch")

# ---- 4. Sources
st.markdown("**Where these headlines come from**")
src = view.summary.sources
top_share = src[0][1] / view.summary.count if src else 0
st.caption(f"{len(src)} publications. The largest source provides {round(top_share * 100)}% of the headlines"
           + (" — coverage is concentrated, so this mood may mostly reflect one outlet." if top_share >= 0.4 else ".")
           + " Google News does not cover all Indian media equally.")
st.plotly_chart(charts.sources_chart(src), width="stretch", config={"displayModeBar": False})

# ---- 5. Recent related headlines
st.markdown("**Recent headlines behind this summary**")
recent = agg.in_window(snap.frame, window)
recent = recent[recent["issue"] == issue.name].sort_values("event_time", ascending=False)
for _, row in recent.head(10).iterrows():
    link = f' — <a href="{escape(row["url"])}" target="_blank" rel="noopener">open article</a>' if isinstance(row["url"], str) else ""
    st.markdown(
        f'<div class="vp-meta" style="font-size:.97rem;color:#1B1F24;margin-bottom:.15rem">'
        f'{ui.highlight_evidence(row["headline"], row["evidence"])}</div>'
        f'<div class="vp-meta" style="margin-bottom:.7rem">{escape(row["label"])} · '
        f'{escape(str(row["source"] or "Unknown source"))} · {escape(da.format_ist(row["event_time"]))}{link}</div>',
        unsafe_allow_html=True,
    )
ui.page_link("pages/2_Headlines.py", "See all headlines with full classification details →")
