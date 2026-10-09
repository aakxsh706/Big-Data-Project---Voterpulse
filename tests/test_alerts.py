from datetime import timedelta

from conftest import NOW, seed
from voterpulse.monitoring.alerts import alert_message, evaluate_alerts

JOBS = "Jobs and Unemployment"


def _setup(conn, ref_mean, cur_mean, ref_n=30, cur_n=30, now=NOW):
    seed(conn, JOBS, ref_mean, now - timedelta(days=3), n=ref_n)
    seed(conn, JOBS, cur_mean, now - timedelta(hours=2), n=cur_n)


def _alerts(conn):
    return conn.execute("SELECT * FROM alerts ORDER BY id").fetchall()


def _check(results):
    return next(c for c in results if c.issue == JOBS)


def test_alert_triggers_on_negative_shift(cfg, conn):
    _setup(conn, ref_mean=0.0, cur_mean=-0.2)
    check = _check(evaluate_alerts(conn, cfg, NOW))
    assert check.action == "created"
    (alert,) = _alerts(conn)
    assert alert["direction"] == "more_negative" and alert["severity"] == "noticeable"
    assert alert["message"].startswith("Noticeable increase in negative coverage about jobs and unemployment")
    assert alert["magnitude"] == 0.2 or abs(alert["magnitude"] - 0.2) < 1e-9


def test_no_alert_below_trigger(cfg, conn):
    _setup(conn, ref_mean=0.0, cur_mean=-0.149)
    assert _check(evaluate_alerts(conn, cfg, NOW)).action == "none"
    assert _alerts(conn) == []


def test_marked_severity(cfg, conn):
    _setup(conn, ref_mean=-0.1, cur_mean=0.25)
    evaluate_alerts(conn, cfg, NOW)
    (alert,) = _alerts(conn)
    assert alert["severity"] == "marked" and alert["direction"] == "more_positive"
    assert "markedly more positive" in alert["message"]


def test_insufficient_data_never_alerts(cfg, conn):
    _setup(conn, ref_mean=0.0, cur_mean=-0.9, cur_n=24)
    assert _check(evaluate_alerts(conn, cfg, NOW)).action == "insufficient"
    assert _alerts(conn) == []


def test_continuing_shift_does_not_duplicate(cfg, conn):
    _setup(conn, ref_mean=0.0, cur_mean=-0.2)
    evaluate_alerts(conn, cfg, NOW)
    later = NOW + timedelta(minutes=15)
    assert _check(evaluate_alerts(conn, cfg, later)).action == "updated"
    assert len(_alerts(conn)) == 1


def test_alert_resets_with_hysteresis(cfg, conn):
    _setup(conn, ref_mean=0.0, cur_mean=-0.2)
    evaluate_alerts(conn, cfg, NOW)
    # Add strongly positive current headlines so the 24h mean returns close to the reference.
    seed(conn, JOBS, 0.2, NOW - timedelta(hours=1), n=30)        # current mean now 0.0
    check = _check(evaluate_alerts(conn, cfg, NOW))
    assert check.action == "resolved"
    (alert,) = _alerts(conn)
    assert alert["status"] == "resolved" and alert["resolved_at"] is not None


def test_hysteresis_keeps_alert_between_reset_and_trigger(cfg, conn):
    _setup(conn, ref_mean=0.0, cur_mean=-0.2)
    evaluate_alerts(conn, cfg, NOW)
    seed(conn, JOBS, 0.0, NOW - timedelta(hours=1), n=30)       # current mean -0.1 (>= reset 0.075)
    assert _check(evaluate_alerts(conn, cfg, NOW)).action == "updated"
    assert _alerts(conn)[0]["status"] == "active"


def test_direction_flip_resolves_and_creates(cfg, conn):
    _setup(conn, ref_mean=0.0, cur_mean=-0.2)
    evaluate_alerts(conn, cfg, NOW)
    seed(conn, JOBS, 0.6, NOW - timedelta(hours=1), n=30)       # current mean +0.2
    assert _check(evaluate_alerts(conn, cfg, NOW)).action == "resolved_and_created"
    rows = _alerts(conn)
    assert [r["status"] for r in rows] == ["resolved", "active"]
    assert rows[1]["direction"] == "more_positive"


def test_messages_are_non_alarmist():
    for direction in ("more_negative", "more_positive"):
        for severity in ("noticeable", "marked"):
            text = alert_message("Law and Order", direction, severity)
            assert "!" not in text and "crisis" not in text.lower() and "coverage" in text
