"""Headless render tests for every dashboard page (Streamlit AppTest)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from voterpulse import config as config_module

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["app.py", "pages/0_Overview.py", "pages/1_Issue_Explorer.py", "pages/2_Headlines.py",
         "pages/3_Methodology.py", "pages/4_Developer_Insights.py"]


@pytest.fixture
def empty_db(tmp_path, monkeypatch):
    monkeypatch.setenv("VOTERPULSE_DB_PATH", str(tmp_path / "empty.db"))
    monkeypatch.setenv("VOTERPULSE_DEMO_MODE", "0")
    monkeypatch.setenv("VOTERPULSE_OLLAMA_HOST", "http://127.0.0.1:9")  # nothing listens here
    config_module.get_config.cache_clear()
    yield tmp_path / "empty.db"
    config_module.get_config.cache_clear()


def _run(page: str) -> AppTest:
    at = AppTest.from_file(str(ROOT / page), default_timeout=30)
    at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_with_empty_database(empty_db, page):
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]
    assert empty_db.exists()  # created on first start


def test_overview_empty_state_shows_six_grey_cards(empty_db):
    at = _run("app.py")
    html = " ".join(m.value for m in at.markdown)
    assert html.count('class="vp-card"') == 6
    assert html.count("Not enough data") >= 6
    assert "measures sentiment in news coverage, not public opinion" in html
    assert any("No headlines have been collected yet" in i.value for i in at.info)
    assert "DEMO DATA" not in html


def test_demo_database_renders_with_banner(tmp_path, monkeypatch):
    import subprocess, sys
    demo = tmp_path / "demo.db"
    # point the generator at a temp demo path via a temp settings override
    monkeypatch.setenv("VOTERPULSE_DEMO_MODE", "1")
    config_module.get_config.cache_clear()
    cfg = config_module.get_config()
    cfg.settings.database.demo_path = str(demo)
    sys.path.insert(0, str(ROOT / "scripts"))
    import generate_demo_data
    generate_demo_data.generate(days=30, seed=7)
    for page in PAGES:
        at = _run(page)
        assert not at.exception, (page, [e.value for e in at.exception])
        assert any("DEMO DATA" in m.value for m in at.markdown), page
    config_module.get_config.cache_clear()
