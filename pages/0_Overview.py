"""Overview page: six issue cards, alerts and the disclaimer.

The dashboard only *reads* the database. Collection runs in a separate process
(scripts/run_scheduler.py), so Streamlit reruns can never start duplicate schedulers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from voterpulse.dashboard import components as ui  # noqa: E402
from voterpulse.dashboard import data_access as da  # noqa: E402

cfg = ui.page_setup("Overview")
now = ui.now_utc()
snap = ui.get_snapshot(cfg)
ui.render_header(cfg, snap, now)

if snap.total_articles == 0:
    ui.render_empty_state(cfg, snap)

top_left, top_right = st.columns([3, 2], vertical_alignment="bottom")
with top_left:
    st.subheader("How each issue is being covered")
with top_right:
    window = ui.period_selector(snap, now, key="overview_period")

views = da.all_issue_views(cfg, snap, window, now)
for row_start in range(0, len(views), 3):
    cols = st.columns(3, gap="medium")
    for col, view in zip(cols, views[row_start:row_start + 3]):
        with col:
            ui.render_issue_card(view, key=f"card_{view.issue.key}")

st.caption(
    f"Mood describes the period selected above. Trend compares the last 7 complete days with the 7 days "
    f"before. Issues with fewer than {cfg.settings.aggregation.min_headlines} headlines show "
    "“Not enough data” in grey. Colours always come with a text label: green = mostly positive, "
    "red = mostly negative, amber = mixed or neutral, grey = not enough data."
)

st.subheader("Mood-shift alerts")
ui.render_alerts(snap)

st.divider()
st.markdown(
    "**Explore further:** open **Issue Explorer** for charts and explanations, **Headlines** to read the "
    "actual news behind every number, and **Methodology** to see exactly how this works and where it can go wrong."
)
