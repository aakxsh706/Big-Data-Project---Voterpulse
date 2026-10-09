"""Domain data structures shared across pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_utc_iso(dt: datetime | None) -> str | None:
    """Serialise a datetime as an ISO-8601 UTC string (naive values are assumed UTC)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_utc_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class ClassificationMethod(str, Enum):
    LLM = "llm"
    KEYWORD_FALLBACK = "keyword_fallback"


class ValidationStatus(str, Enum):
    VALID = "valid"                          # LLM output passed schema + evidence checks
    EVIDENCE_UNVERIFIED = "evidence_unverified"  # schema OK, evidence not found in headline
    LLM_INVALID = "llm_invalid"              # LLM answered but output failed validation -> fallback
    LLM_UNAVAILABLE = "llm_unavailable"      # Ollama down / timed out -> fallback


class SentimentLabel(str, Enum):
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"


@dataclass(frozen=True)
class RawHeadline:
    """One RSS entry as fetched, before normalisation."""

    headline: str
    url: str | None
    source: str | None
    published_at: datetime | None
    feed_key: str


@dataclass(frozen=True)
class PreparedArticle:
    """A headline that has been cleaned and hashed, ready to store."""

    headline: str
    normalized_headline: str
    headline_hash: str
    url: str | None
    source: str | None
    published_at: datetime | None
    feed_key: str


@dataclass
class KeywordResult:
    issue: str
    scores: dict[str, float]
    matched: dict[str, list[str]]

    @property
    def evidence(self) -> str:
        return " ... ".join(self.matched.get(self.issue, []))


@dataclass
class ClassificationOutcome:
    """Final classification of one headline, with full audit information."""

    primary_issue: str
    secondary_issue: str | None
    evidence: str
    method: ClassificationMethod
    validation_status: ValidationStatus
    keyword_issue: str
    model_id: str | None = None
    raw_response: str | None = None
    prompt_version: str | None = None
    latency_ms: float | None = None
    error: str | None = None
    from_cache: bool = False
    review_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SentimentScores:
    positive: float
    negative: float
    neutral: float
    compound: float
    label: SentimentLabel


@dataclass
class FeedResult:
    feed_key: str
    ok: bool
    headlines: list[RawHeadline] = field(default_factory=list)
    error: str | None = None
    status_code: int | None = None
