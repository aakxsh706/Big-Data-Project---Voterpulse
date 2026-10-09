"""Shared deterministic fixtures. No test touches the network or a real Ollama server."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from voterpulse import database as db
from voterpulse.config import AppConfig, load_config

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 10, 9, 6, 30, tzinfo=timezone.utc)  # 12:00 IST


@pytest.fixture
def cfg() -> AppConfig:
    # env={} -> ignore the developer's environment and .env for reproducibility
    return load_config(env={})


@pytest.fixture
def conn(tmp_path: Path):
    connection = db.init_db(tmp_path / "test.db", "live")
    yield connection
    connection.close()


@pytest.fixture
def rss_bytes() -> bytes:
    return (FIXTURES / "google_news_sample.xml").read_bytes()


class FakeResponse:
    def __init__(self, status_code: int = 200, body: Any = None, content: bytes | None = None) -> None:
        self.status_code = status_code
        self._body = body
        self.content = content if content is not None else json.dumps(body).encode()
        self.text = self.content.decode("utf-8", "replace")

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no JSON body")
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """Minimal stand-in for requests.Session driven by handler callables."""

    def __init__(self, get: Callable[..., Any] | None = None, post: Callable[..., Any] | None = None) -> None:
        self._get, self._post = get, post
        self.get_calls: list[str] = []
        self.post_calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.get_calls.append(url)
        result = self._get(url, **kwargs) if self._get else FakeResponse(404, {})
        if isinstance(result, Exception):
            raise result
        return result

    def post(self, url: str, json: Any = None, **kwargs: Any) -> FakeResponse:
        self.post_calls.append(json)
        result = self._post(url, json, **kwargs) if self._post else FakeResponse(404, {})
        if isinstance(result, Exception):
            raise result
        return result


def ollama_tags(model: str = "llama3.2:3b", digest: str = "a80c4f17acd5ffffffff") -> FakeResponse:
    return FakeResponse(200, {"models": [{"name": model, "model": model, "digest": digest}]})


def ollama_chat(content: str) -> FakeResponse:
    return FakeResponse(200, {"message": {"role": "assistant", "content": content}})


def ollama_answer(issue: str, evidence: str, secondary: str | None = None) -> FakeResponse:
    return ollama_chat(json.dumps({"issue": issue, "evidence": evidence, "secondary": secondary}))


@pytest.fixture
def no_sleep() -> Callable[[float], None]:
    return lambda _seconds: None


_seed_counter = 0


def seed(conn, issue: str, compound: float, when: datetime, n: int = 1, label: str | None = None,
         source: str = "Test Source", relevant: bool = True) -> None:
    """Insert ``n`` classified + scored articles directly (bypassing RSS/LLM)."""
    global _seed_counter
    from voterpulse.models import (ClassificationMethod, ClassificationOutcome, PreparedArticle,
                                   SentimentLabel, SentimentScores, ValidationStatus)
    from voterpulse.sentiment.vader_analyzer import label_for
    from voterpulse.config import HeadlineThresholds

    for _ in range(n):
        _seed_counter += 1
        art = PreparedArticle(f"seed {_seed_counter}", f"seed {_seed_counter}", f"h{_seed_counter:08d}",
                              "https://x.test", source, when, "test")
        aid = db.insert_article(conn, art, ingested_at=when)
        outcome = ClassificationOutcome(issue if relevant else "Not an election issue", None, "seed",
                                        ClassificationMethod.LLM, ValidationStatus.VALID, issue)
        db.save_classification(conn, aid, outcome, relevant, when)
        if relevant:
            lab = SentimentLabel(label) if label else label_for(compound, HeadlineThresholds())
            db.save_sentiment(conn, aid, SentimentScores(0.0, 0.0, 1.0, compound, lab), when)
