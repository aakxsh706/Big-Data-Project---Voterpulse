"""Strict validation of LLM output. Nothing invalid is ever silently accepted."""

from __future__ import annotations

import json
import re
import unicodedata

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from voterpulse.config import IssuesConfig


class LLMResponseError(ValueError):
    """The model answered, but the answer is not acceptable. Carries the raw text."""

    def __init__(self, message: str, raw: str | None = None) -> None:
        super().__init__(message)
        self.raw = raw


class LLMClassification(BaseModel):
    """Shape of a model answer. Label membership is checked in :func:`validate_response`."""

    model_config = ConfigDict(extra="forbid", strict=True)

    issue: str
    evidence: str
    secondary: str | None

    @field_validator("issue", "evidence")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @field_validator("secondary")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


def validate_response(raw: str, issues_cfg: IssuesConfig) -> LLMClassification:
    """Parse and validate raw model text. Raises :class:`LLMResponseError` on any problem."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise LLMResponseError(f"invalid JSON: {exc}", raw) from exc
    if not isinstance(data, dict):
        raise LLMResponseError("JSON root is not an object", raw)
    try:
        parsed = LLMClassification.model_validate(data)
    except ValidationError as exc:
        raise LLMResponseError(f"schema violation: {exc.errors(include_url=False)}", raw) from exc

    if parsed.issue not in issues_cfg.all_labels:
        raise LLMResponseError(f"unknown issue label {parsed.issue!r}", raw)
    if parsed.secondary is not None:
        if parsed.secondary not in issues_cfg.issue_names:
            raise LLMResponseError(f"unknown secondary label {parsed.secondary!r}", raw)
        if parsed.secondary == parsed.issue:
            raise LLMResponseError("secondary duplicates the primary issue", raw)
        if parsed.issue == issues_cfg.not_relevant_label:
            raise LLMResponseError("secondary given for a non-election headline", raw)
    return parsed


_SPAN_SPLIT = re.compile(r"\s*(?:\.\.\.|…)\s*")
_WS = re.compile(r"\s+")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                         "–": "-", "—": "-"})


def _canon(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES).casefold()
    return _WS.sub(" ", text).strip()


def evidence_spans(evidence: str) -> list[str]:
    text = (evidence or "").translate(_QUOTES)
    spans = [s.strip().strip("\"'").strip() for s in _SPAN_SPLIT.split(text)]
    return [s for s in spans if s]


def evidence_is_verified(evidence: str, headline: str) -> bool:
    """True when every evidence span occurs verbatim (ignoring case/whitespace/quote style)."""
    spans = evidence_spans(evidence)
    if not spans:
        return False
    canon_headline = _canon(headline)
    return all(_canon(span) in canon_headline for span in spans)
