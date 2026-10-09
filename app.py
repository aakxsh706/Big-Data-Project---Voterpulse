"""VoterPulse Lite entry point.  Run from the project root with:  streamlit run app.py

The dashboard only *reads* the database. Collection runs in a separate process
(scripts/run_scheduler.py), so Streamlit reruns can never start duplicate schedulers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st  # noqa: E402

navigation = st.navigation([
    st.Page("pages/0_Overview.py", title="Overview", icon=":material/dashboard:", default=True),
    st.Page("pages/1_Issue_Explorer.py", title="Issue Explorer", icon=":material/query_stats:",
            url_path="issue-explorer"),
    st.Page("pages/2_Headlines.py", title="Headlines", icon=":material/newspaper:", url_path="headlines"),
    st.Page("pages/3_Methodology.py", title="Methodology", icon=":material/menu_book:", url_path="methodology"),
    st.Page("pages/4_Developer_Insights.py", title="Developer Insights", icon=":material/developer_mode:",
            url_path="developer"),
])
navigation.run()
