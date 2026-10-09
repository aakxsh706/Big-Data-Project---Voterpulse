"""How VoterPulse Lite works, in plain language, with its limitations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st  # noqa: E402

from voterpulse.dashboard import components as ui  # noqa: E402

cfg = ui.page_setup("Methodology")
snap = ui.get_snapshot(cfg)
ui.render_demo_banner(cfg, snap)
st.title("Methodology")
ui.render_disclaimer()

a = cfg.thresholds.aggregate
h = cfg.thresholds.headline
tr = cfg.thresholds.trend
al = cfg.settings.alerts
n = cfg.settings.aggregation.min_headlines

st.header("1. Where the news comes from")
st.markdown(
    f"Every {cfg.settings.pipeline.interval_minutes} minutes, VoterPulse asks **Google News** (India edition, "
    "English) for recent headlines using one search per issue. Only the **headline**, publication, link and date "
    "are stored — not the article text. The same headline arriving from several searches or publishers' "
    "variants is stored once.\n\n"
    "The searches only *find candidates*. They do not decide the issue."
)
with st.expander("Show the exact searches used"):
    for feed in cfg.rss.feeds:
        st.code(feed.query, language=None)

st.header("2. How issues are assigned")
st.markdown(
    f"Each headline is read by an AI language model (**{cfg.model.ollama.model}**, Llama 3.2 3B Instruct) "
    "running **on this computer** through Ollama — no data is sent to any AI service. The model picks exactly "
    "one of seven labels: the six issues below, or *Not an election issue*. It must also quote the words "
    "from the headline that justify its choice, which you can see on the Headlines page.\n\n"
    "A simple keyword list checks every answer. When the two disagree, the AI's answer is kept and the "
    "headline is put in a review queue for a person to check. If the AI model is unavailable, the keyword "
    "list is used instead and the headline is clearly marked *Keyword rules*."
)
for issue in cfg.issues.issues:
    st.markdown(f"- **{issue.name}** — {issue.description}")
st.markdown("Headlines labelled *Not an election issue* are kept for auditing but never count towards any mood.")

st.header("3. How tone is scored")
st.markdown(
    "Each accepted headline's wording is scored by **VADER**, a widely used open-source sentiment tool. It "
    "gives a score from −1 (very negative wording) to +1 (very positive wording).\n\n"
    f"- **Positive** headline: score ≥ {h.positive_min}\n"
    f"- **Negative** headline: score ≤ {h.negative_max}\n"
    "- **Neutral** headline: anything in between\n\n"
    "These cut-offs are the ones recommended by VADER's authors."
)

st.header("4. What the colours and moods mean")
st.markdown(
    f"| Colour | Label | Rule (over the selected period) |\n|---|---|---|\n"
    f"| 🟢 Green | Mostly positive | average score ≥ {a.positive_mean} **and** positive headlines outnumber negative by ≥ {round(a.share_margin*100)} points |\n"
    f"| 🔴 Red | Mostly negative | average score ≤ {a.negative_mean} **and** negative headlines outnumber positive by ≥ {round(a.share_margin*100)} points |\n"
    f"| 🟠 Amber | Mostly neutral | ≥ {round(a.neutral_dominant_share*100)}% neutral headlines and a weak average |\n"
    f"| 🟠 Amber | Mixed | none of the above |\n"
    f"| ⚪ Grey | Not enough data | fewer than {n} headlines |\n\n"
    "Colour is never the only signal — every card also shows the label in words."
)

st.header("5. How trends and alerts are calculated")
st.markdown(
    f"**Weekly trend:** the average score of the last **7 complete days** (ending at midnight IST) is compared "
    f"with the 7 days before. A change of at least {tr.change_min} counts as *more positive* or *more negative*; "
    f"smaller changes are *broadly unchanged*. Both weeks need at least {tr.min_headlines_each_window} headlines.\n\n"
    f"**Mood-shift alerts:** the last {al.current_window_hours} hours are compared with the "
    f"{al.reference_window_days} days before them. An alert appears when the average moves by at least "
    f"{al.trigger_delta} (\"marked\" from {al.large_delta}), with at least {al.min_current_headlines} headlines in "
    f"both windows. One alert per issue stays open while the shift continues, and closes when the gap falls "
    f"below {al.reset_delta}."
)

st.header("6. Limitations — please read")
st.markdown(
    "- **News tone is not public opinion.** Headlines reflect editorial choices, not what voters think.\n"
    "- **Google News coverage is incomplete.** It favours large English-language outlets; regional and "
    "Indian-language media are under-represented. A few outlets can dominate an issue (see *sources* on each issue).\n"
    "- **Headline-only scoring.** A headline can be misleading on its own; the article may say something different.\n"
    "- **VADER was not built for Indian politics.** It can misread sarcasm, negation, Hindi/Hinglish words "
    "(e.g. *mehngai*) and policy terms. *“Crime falls”* contains the negative word *crime*.\n"
    "- **The AI model makes mistakes.** A 3-billion-parameter model can mislabel headlines; the keyword "
    "cross-check and review queue catch some, not all.\n"
    "- **Small numbers are unreliable.** That is why issues with few headlines show *Not enough data*.\n"
    "- **Delays and gaps.** If the computer is off or offline, nothing is collected; older data remains visible "
    "and the freshness label says so.\n"
    "- **No rankings or predictions.** VoterPulse does not compare parties or forecast results."
)
st.caption("Full technical documentation: docs/methodology.md and docs/thresholds.md in the project folder.")
