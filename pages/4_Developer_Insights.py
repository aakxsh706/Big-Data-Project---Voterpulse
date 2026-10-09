"""Developer / evaluation view: numbers, logs, model details and the review queue.

Kept separate from the voter-facing pages on purpose.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from voterpulse.classification.llm_classifier import OllamaClassifier  # noqa: E402
from voterpulse.dashboard import charts  # noqa: E402
from voterpulse.dashboard import components as ui  # noqa: E402
from voterpulse.dashboard import data_access as da  # noqa: E402

cfg = ui.page_setup("Developer Insights")
now = ui.now_utc()
snap = ui.get_snapshot(cfg)
ui.render_demo_banner(cfg, snap)
st.title("Developer insights")
st.caption("Technical view for developers and evaluators. Voter-facing pages do not show these numbers.")

tables = da.load_developer_tables(cfg)


@st.cache_data(ttl=30, show_spinner=False)
def ollama_status(host: str, model: str) -> tuple[bool, str]:
    clf = OllamaClassifier.from_config(cfg)
    try:
        return True, clf.resolve_model_id()
    except Exception as exc:  # display only
        message = str(exc)
        return False, message if len(message) <= 140 else message[:140] + "…"


ok, model_info = ollama_status(cfg.model.ollama.host, cfg.model.ollama.model)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Stored headlines", snap.total_articles)
c2.metric("Waiting / failed classification", int(tables["failed"]["unclassified"].iloc[0]))
pending = int((tables["review"]["status"] == "pending").sum()) if not tables["review"].empty else 0
c3.metric("Pending reviews", pending)
c4.metric("Ollama", "reachable" if ok else "unavailable")
st.caption(f"Configured model: `{cfg.model.ollama.model}` · expected digest `{cfg.model.ollama.expected_digest}` · "
           f"detected: `{model_info}` · database: `{snap.db_path}`")

tab_numbers, tab_class, tab_review, tab_runs, tab_alerts, tab_logs = st.tabs(
    ["Sentiment numbers", "Classification", "Review queue", "Pipeline runs", "Alert history", "Logs"])

with tab_numbers:
    rows = []
    for period in ("24h", "7d", "30d"):
        from voterpulse.sentiment import aggregator as agg
        window = agg.rolling_window(period, now, snap.coverage_start)
        for view in da.all_issue_views(cfg, snap, window, now):
            s, t = view.summary, view.trend
            rows.append({
                "period": period, "issue": s.issue, "n": s.count,
                "mean_compound": None if s.mean_compound is None else round(s.mean_compound, 4),
                "pos_%": round(s.positive_share * 100, 1), "neu_%": round(s.neutral_share * 100, 1),
                "neg_%": round(s.negative_share * 100, 1), "mood": view.report.mood.value,
                "trend_delta": None if t.delta is None else round(t.delta, 4), "trend": view.report.trend.value,
                "week_n": t.current_count, "prev_week_n": t.previous_count, "partial_window": window.is_partial,
            })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Thresholds: config/sentiment_thresholds.yaml · rationale: docs/thresholds.md")

with tab_class:
    m = tables["methods"]
    if m.empty:
        st.info("No classifications yet.")
    else:
        left, right = st.columns([1, 1.4])
        with left:
            st.markdown("**Classification method**")
            st.plotly_chart(charts.method_breakdown_chart(m), width="stretch", config={"displayModeBar": False})
        with right:
            st.markdown("**By method and validation status**")
            st.dataframe(m, hide_index=True, width="stretch")
            ag = tables["agreement"].assign(agreement_rate=lambda d: (d["agree"] / d["headlines"]).round(3))
            st.markdown("**Agreement with keyword baseline**")
            st.dataframe(ag, hide_index=True, width="stretch")
        st.markdown("**Models and prompt versions used**")
        st.dataframe(tables["models"], hide_index=True, width="stretch")

with tab_review:
    review = tables["review"]
    if review.empty:
        st.info("The review queue is empty.")
    else:
        status_filter = st.segmented_control("Show", ["pending", "all"], default="pending", key="rq_filter")
        shown = review if status_filter == "all" else review[review["status"] == "pending"]
        st.dataframe(shown.drop(columns=["raw_response"]), hide_index=True, width="stretch")
        if not shown.empty:
            with st.form("review_form"):
                rid = st.selectbox("Review item", shown["id"].tolist(),
                                   format_func=lambda i: f"#{i}: {shown.set_index('id').loc[i, 'headline'][:90]}")
                item = shown.set_index("id").loc[rid]
                st.markdown(f"LLM: **{item['llm_label'] or '—'}** · keyword: **{item['keyword_label'] or '—'}** · "
                            f"reason: `{item['reason']}` · evidence: “{item['evidence'] or ''}”")
                with st.expander("Raw model response (audit)"):
                    st.code(item["raw_response"] or "(none)", language="json")
                decision = st.radio("Decision", ["llm_correct", "keyword_correct", "both_wrong", "dismissed"],
                                    horizontal=True)
                corrected = st.selectbox("Correct issue (if both wrong)", [""] + cfg.issues.all_labels)
                notes = st.text_area("Reviewer notes")
                if st.form_submit_button("Save review"):
                    da.save_review(cfg, int(rid), decision, notes or None, corrected or None)
                    st.success("Saved. (Review decisions are recorded for evaluation; they do not change the "
                               "stored classification automatically.)")
                    st.cache_data.clear()

with tab_runs:
    runs = tables["runs"]
    if runs.empty:
        st.info("No pipeline runs recorded yet.")
    else:
        st.dataframe(runs, hide_index=True, width="stretch")
        done = runs[runs["status"].isin(["success", "partial"])]
        if not done.empty:
            st.caption(f"Median run duration {done['duration_seconds'].median():.1f}s · "
                       f"median avg LLM latency {done['avg_classification_ms'].median() if done['avg_classification_ms'].notna().any() else float('nan'):.0f} ms")

with tab_alerts:
    st.dataframe(tables["alerts"], hide_index=True, width="stretch")

with tab_logs:
    text = da.tail_log(cfg, 300)
    st.code(text or "(log file is empty)", language=None)
