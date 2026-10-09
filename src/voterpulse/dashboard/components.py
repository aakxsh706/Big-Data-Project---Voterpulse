"""Reusable Streamlit UI pieces. Presentation only — all numbers come from data_access."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from html import escape

import streamlit as st

from voterpulse import DISCLAIMER
from voterpulse.config import IST, AppConfig, get_config
from voterpulse.dashboard import data_access as da
from voterpulse.logging_utils import setup_from_config
from voterpulse.sentiment import aggregator as agg
from voterpulse.sentiment.translator import Mood, MoodReport

# Darker text variants keep >= 4.5:1 contrast on the tinted badge backgrounds.
MOOD_TEXT = {
    Mood.POSITIVE: "#18603A", Mood.NEGATIVE: "#912018", Mood.MIXED: "#93370D",
    Mood.NEUTRAL: "#93370D", Mood.INSUFFICIENT: "#475467",
}
RUN_STATUS = {
    "success": ("#18603A", "#ECFDF3", "✓ Running normally"),
    "partial": ("#93370D", "#FFFAEB", "! Partly successful"),
    "failed": ("#912018", "#FEF3F2", "✕ Last update failed"),
    "running": ("#475467", "#F2F4F7", "… Updating now"),
    "abandoned": ("#912018", "#FEF3F2", "✕ Last update did not finish"),
    None: ("#475467", "#F2F4F7", "○ No updates yet"),
}

CSS = """
<style>
html, body, [class*="css"] { font-family: Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.block-container { padding-top: 2rem; max-width: 1180px; }
.vp-title { font-size: 2.1rem; font-weight: 750; color: #10243E; margin: 0; letter-spacing: -0.02em; }
.vp-subtitle { font-size: 1.08rem; color: #475467; margin: .2rem 0 .9rem 0; }
.vp-chips { display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: .4rem; }
.vp-chip { font-size: .86rem; padding: .28rem .7rem; border-radius: 999px; border: 1px solid #EAECF0;
           background: #FFFFFF; color: #344054; }
.vp-card { background: #FFFFFF; border: 1px solid #EAECF0; border-radius: 14px; padding: 1.05rem 1.1rem .9rem;
           box-shadow: 0 1px 2px rgba(16,24,40,.05); min-height: 238px; margin-bottom: .35rem; }
.vp-card-issue { font-size: 1.06rem; font-weight: 700; color: #10243E; margin-bottom: .55rem; }
.vp-badge { display: inline-block; font-size: .82rem; font-weight: 700; padding: .22rem .62rem;
            border-radius: 999px; margin-bottom: .55rem; }
.vp-mood { font-size: 1.02rem; color: #1B1F24; font-weight: 600; line-height: 1.35; margin-bottom: .55rem; }
.vp-meta { font-size: .86rem; color: #475467; line-height: 1.5; }
.vp-alert { border-left: 4px solid; background: #FFFFFF; border-radius: 10px; padding: .75rem 1rem;
            margin-bottom: .55rem; border-top: 1px solid #EAECF0; border-right: 1px solid #EAECF0;
            border-bottom: 1px solid #EAECF0; }
.vp-alert-title { font-weight: 650; color: #1B1F24; }
.vp-disclaimer { background: #EEF4FB; border: 1px solid #C7D7EA; color: #10243E; border-radius: 10px;
                 padding: .7rem 1rem; font-size: .95rem; margin: .4rem 0 1.1rem 0; }
.vp-demo { background: #FFF4E5; border: 2px dashed #B54708; color: #7A2E0E; border-radius: 10px;
           padding: .7rem 1rem; font-weight: 650; margin-bottom: 1rem; }
.vp-big-mood { font-size: 1.6rem; font-weight: 750; line-height: 1.25; margin: .2rem 0 .5rem 0; }
.vp-evidence mark { background: #FEF0C7; color: #1B1F24; padding: 0 .15rem; border-radius: 3px; }
@media (max-width: 640px) { .vp-title { font-size: 1.6rem; } .vp-card { min-height: 0; } }
</style>
"""


def page_setup(title: str) -> AppConfig:
    st.set_page_config(page_title=f"{title} · VoterPulse Lite", page_icon="🗳️", layout="wide")
    setup_from_config()
    st.markdown(CSS, unsafe_allow_html=True)
    return get_config()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


@st.cache_data(ttl=60, show_spinner=False)
def cached_snapshot(db_path: str) -> da.Snapshot:
    # db_path is part of the cache key so demo and live data are never mixed.
    return da.load_snapshot(get_config())


def get_snapshot(cfg: AppConfig) -> da.Snapshot:
    return cached_snapshot(str(cfg.db_path))


# --------------------------------------------------------------------------- header
def render_demo_banner(cfg: AppConfig, snap: da.Snapshot) -> None:
    if cfg.settings.demo_enabled or snap.dataset_kind == "demo":
        st.markdown(
            '<div class="vp-demo">DEMO DATA — every headline shown is synthetic and was generated '
            "for demonstration. Nothing here describes real news coverage.</div>",
            unsafe_allow_html=True,
        )


def render_disclaimer() -> None:
    st.markdown(f'<div class="vp-disclaimer"><b>Please note:</b> {escape(DISCLAIMER)}</div>',
                unsafe_allow_html=True)


def render_header(cfg: AppConfig, snap: da.Snapshot, now: datetime) -> None:
    render_demo_banner(cfg, snap)
    left, right = st.columns([5, 1.3], vertical_alignment="bottom")
    with left:
        st.markdown('<div class="vp-title" role="heading" aria-level="1">VoterPulse Lite</div>'
                    '<div class="vp-subtitle">Understand how India\'s election issues are being covered in the news.</div>',
                    unsafe_allow_html=True)
    with right:
        with st.popover("ⓘ What is measured?", width="stretch"):
            st.markdown(
                "**What this shows:** the *tone of news headlines* about six election issues, "
                "collected from Google News every 15 minutes.\n\n"
                "**What it does not show:** what voters think, who is popular, or who will win.\n\n"
                "Each headline is sorted into an issue by an AI model running on this computer, "
                "then its wording is scored as positive, neutral or negative. "
                "See the **Methodology** page for details and limitations."
            )
    status = snap.last_run["status"] if snap.last_run else None
    color, bg, text = RUN_STATUS.get(status, RUN_STATUS[None])
    from voterpulse.sentiment.translator import freshness_sentence
    fresh = freshness_sentence(snap.last_success, now, cfg.settings.aggregation.stale_after_minutes)
    last = da.format_ist(snap.last_success) if snap.last_success else "never"
    st.markdown(
        '<div class="vp-chips">'
        f'<span class="vp-chip">Latest successful update: <b>{escape(last)}</b></span>'
        f'<span class="vp-chip" style="color:{color};background:{bg}">Data collection: <b>{escape(text)}</b></span>'
        f'<span class="vp-chip">Data freshness: <b>{escape(fresh)}</b></span>'
        "</div>",
        unsafe_allow_html=True,
    )
    render_disclaimer()


# --------------------------------------------------------------------------- selectors
PERIOD_OPTIONS = {"24h": "Last 24 hours", "7d": "Last 7 days", "30d": "Last 30 days"}


def period_selector(snap: da.Snapshot, now: datetime, key: str, allow_custom: bool = False) -> agg.Window:
    options = list(PERIOD_OPTIONS) + (["custom"] if allow_custom else [])
    labels = {**PERIOD_OPTIONS, "custom": "Custom dates"}
    choice = st.segmented_control("Time period", options, format_func=labels.get, default="7d",
                                  key=key) or "7d"
    if choice == "custom":
        today = now.astimezone(IST).date()
        picked = st.date_input("Dates (IST)", value=(today - timedelta(days=13), today),
                               max_value=today, key=f"{key}_dates")
        if isinstance(picked, (tuple, list)) and len(picked) == 2:
            start, end = picked
        else:
            start = end = picked[0] if isinstance(picked, (tuple, list)) else picked
        return agg.custom_window(start if isinstance(start, date) else today,
                                 end if isinstance(end, date) else today, now, snap.coverage_start)
    return agg.rolling_window(choice, now, snap.coverage_start)


# --------------------------------------------------------------------------- cards
def mood_badge(report: MoodReport) -> str:
    return (f'<span class="vp-badge" style="background:{report.background};color:{MOOD_TEXT[report.mood]};'
            f'border:1px solid {report.color}">{report.icon} {escape(report.label)}</span>')


def render_issue_card(view: da.IssueView, key: str) -> None:
    r = view.report
    window_text = view.summary.window.label.lower() if view.summary.window.key != "custom" else view.summary.window.label
    count_text = f"{r.count} headline{'s' if r.count != 1 else ''} · {window_text}"
    latest = (f"Latest headline: {da.format_ist(view.summary.latest_event_time)}"
              if view.summary.latest_event_time else "No headlines in this period")
    st.markdown(
        f'<div class="vp-card" style="border-top:5px solid {r.color}">'
        f'<div class="vp-card-issue">{escape(view.issue.name)}</div>'
        f"{mood_badge(r)}"
        f'<div class="vp-mood">Mood: {escape(r.headline)}</div>'
        f'<div class="vp-meta"><b>Trend:</b> {escape(r.trend_label)}</div>'
        f'<div class="vp-meta">{escape(count_text)}</div>'
        f'<div class="vp-meta">{escape(latest)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    if st.button("View details →", key=key, width="stretch"):
        st.session_state["selected_issue"] = view.issue.name
        st.switch_page("pages/1_Issue_Explorer.py")


# --------------------------------------------------------------------------- alerts
def render_alerts(snap: da.Snapshot, compact: bool = False) -> None:
    if snap.alerts.empty:
        st.markdown('<div class="vp-meta">No mood-shift alerts right now. Alerts appear when the '
                    "tone of the last 24 hours differs clearly from the week before, and only when "
                    "there are enough headlines to judge.</div>", unsafe_allow_html=True)
        return
    for _, alert in snap.alerts.iterrows():
        negative = alert["direction"] == "more_negative"
        color = "#B42318" if negative else "#1E7B46"
        icon = "↘" if negative else "↗"
        detail = (f"Average tone moved from {alert['previous_sentiment']:+.2f} (previous week, "
                  f"{alert['reference_count']} headlines) to {alert['current_sentiment']:+.2f} "
                  f"(last 24 hours, {alert['current_count']} headlines).")
        body = "" if compact else f'<div class="vp-meta">{escape(detail)}</div>'
        st.markdown(
            f'<div class="vp-alert" style="border-left-color:{color}">'
            f'<div class="vp-alert-title">{icon} {escape(alert["message"])}</div>'
            f'<div class="vp-meta">Severity: {escape(str(alert["severity"]).capitalize())} · '
            f'first noticed {escape(da.format_ist(alert["triggered_at"]))}</div>{body}</div>',
            unsafe_allow_html=True,
        )
    st.caption("Alerts describe changes in news tone only. A shift can be caused by a single big story "
               "or by a few outlets publishing many similar headlines.")


def render_empty_state(cfg: AppConfig, snap: da.Snapshot) -> None:
    st.info(
        "No headlines have been collected yet. To start collecting, open a terminal in the project "
        "folder and run:\n\n"
        "`python scripts/run_pipeline.py` (once) or `python scripts/run_scheduler.py` (every 15 minutes).\n\n"
        "To explore the dashboard with synthetic data instead, run `python scripts/generate_demo_data.py` "
        "and start the app with `VOTERPULSE_DEMO_MODE=1`.",
        icon="ℹ️",
    )


def highlight_evidence(headline: str, evidence: str | None) -> str:
    """HTML-escape the headline and wrap verified evidence spans in <mark>."""
    from voterpulse.classification.validator import evidence_spans

    text = escape(headline)
    for span in sorted(evidence_spans(evidence or ""), key=len, reverse=True):
        esc = escape(span)
        lower = text.lower()
        idx = lower.find(esc.lower())
        if idx >= 0 and "<mark>" not in text[max(0, idx - 6):idx]:
            text = f"{text[:idx]}<mark>{text[idx:idx + len(esc)]}</mark>{text[idx + len(esc):]}"
    return f'<span class="vp-evidence">{text}</span>'


def page_link(page: str, label: str) -> None:
    """``st.page_link`` that degrades to plain text if the page registry is unavailable
    (e.g. when a page file is executed on its own by Streamlit's AppTest)."""
    try:
        st.page_link(page, label=label)
    except Exception:  # StreamlitPageNotFoundError and similar
        st.markdown(f"{label} (see the sidebar)")
