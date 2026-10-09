"""Source headlines: trace every aggregate back to the actual news and its classification."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from html import escape  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from voterpulse.dashboard import components as ui  # noqa: E402
from voterpulse.dashboard import data_access as da  # noqa: E402
from voterpulse.sentiment import aggregator as agg  # noqa: E402

METHOD_TEXT = {
    "llm": "Local AI model",
    "keyword_fallback": "Keyword rules (AI model was unavailable or gave an invalid answer)",
}
PAGE_SIZE = 25


@st.cache_data(ttl=60, show_spinner=False)
def headlines(db_path: str) -> pd.DataFrame:
    return da.load_headlines(ui.get_config())


cfg = ui.page_setup("Headlines")
now = ui.now_utc()
snap = ui.get_snapshot(cfg)
ui.render_demo_banner(cfg, snap)
st.title("Headlines")
st.markdown("Every number in VoterPulse comes from these headlines. Open any headline to see **why** it was "
            "given its issue — the highlighted words are the evidence the classifier pointed to.")

df = headlines(str(cfg.db_path))
if df.empty:
    ui.render_empty_state(cfg, snap)
    st.stop()

with st.container(border=True):
    f1, f2, f3 = st.columns([2, 1.3, 1.3])
    issues = f1.multiselect("Issue", cfg.issues.issue_names, default=[st.session_state.get("selected_issue")]
                            if st.session_state.get("selected_issue") else [])
    tones = f2.multiselect("Tone", ["Negative", "Neutral", "Positive"])
    methods = f3.multiselect("Classified by", list(METHOD_TEXT), format_func=lambda m: METHOD_TEXT[m].split(" (")[0])
    g1, g2 = st.columns([2, 3], vertical_alignment="bottom")
    query = g1.text_input("Search headline text", placeholder="e.g. MSP, power cut, NEET")
    window = None
    with g2:
        window = ui.period_selector(snap, now, key="headlines_period", allow_custom=True)
    show_rejected = st.toggle("Also show headlines rejected as “Not an election issue” or still waiting "
                              "(these never count towards any mood)")

view = agg.in_window(df, window)
if not show_rejected:
    view = view[view["is_relevant"] == 1]
if issues:
    view = view[view["issue"].isin(issues)]
if tones:
    view = view[view["label"].isin(tones)]
if methods:
    view = view[view["method"].isin(methods)]
if query:
    view = view[view["headline"].str.contains(query, case=False, regex=False, na=False)]

st.markdown(f"**{len(view)} headlines** match ({window.label.lower() if window.key != 'custom' else window.label}).")
pages = max(1, -(-len(view) // PAGE_SIZE))
page = st.number_input("Page", min_value=1, max_value=pages, value=1, step=1) if pages > 1 else 1
chunk = view.iloc[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]

for _, row in chunk.iterrows():
    label = row["label"] if isinstance(row["label"], str) else "Not scored"
    issue_text = row["issue"] if isinstance(row["issue"], str) else "Waiting for classification"
    with st.expander(f"{row['headline']}  ·  {issue_text}  ·  {label}"):
        st.markdown(f'<div style="font-size:1.08rem">{ui.highlight_evidence(row["headline"], row["evidence"])}</div>',
                    unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            st.markdown(f"**Publication:** {row['source'] or 'Unknown source'}  \n"
                        f"**Published:** {da.format_ist(row['published_at']) if row['published_at'] else 'Date not given by the feed'}  \n"
                        f"**Collected:** {da.format_ist(row['ingested_at'])}")
            if isinstance(row["url"], str):
                st.link_button("Open original article ↗", row["url"])
        with b:
            st.markdown(f"**Assigned issue:** {issue_text}  \n"
                        + (f"**Also mentions:** {row['secondary']} (not counted)  \n" if isinstance(row["secondary"], str) else "")
                        + f"**Evidence words:** “{row['evidence'] or '—'}”  \n"
                        f"**Tone of the headline wording:** {label}  \n"
                        f"**Classified by:** {METHOD_TEXT.get(row['method'], '—')}")
            if isinstance(row["keyword_issue"], str) and row["keyword_issue"] != row["issue"]:
                st.caption(f"The keyword cross-check suggested “{row['keyword_issue']}”, so this headline is in the review queue.")
            if row["validation_status"] == "evidence_unverified":
                st.caption("The evidence words could not be found exactly in the headline; flagged for human review.")
            if pd.notna(row["compound"]):
                st.caption(f"Developer detail — VADER compound {row['compound']:+.3f} "
                           f"(pos {row['positive']:.2f}, neu {row['neutral']:.2f}, neg {row['negative']:.2f})")

st.download_button("Download these headlines (CSV)", view.drop(columns=["event_time"]).to_csv(index=False),
                   file_name="voterpulse_headlines.csv", mime="text/csv")
