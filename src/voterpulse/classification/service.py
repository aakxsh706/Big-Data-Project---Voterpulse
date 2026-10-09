"""Classification service: cache -> local LLM -> validation -> keyword fallback.

Every outcome records its method honestly: ``llm`` only when a validated LLM answer
(fresh or cached) was used, otherwise ``keyword_fallback``.
"""

from __future__ import annotations

import sqlite3

from voterpulse import database as db
from voterpulse.classification.keyword_classifier import KeywordClassifier
from voterpulse.classification.llm_classifier import OllamaClassifier, OllamaUnavailable
from voterpulse.classification.validator import LLMResponseError, evidence_is_verified
from voterpulse.config import IssuesConfig
from voterpulse.logging_utils import fields, get_logger
from voterpulse.models import ClassificationMethod, ClassificationOutcome, KeywordResult, ValidationStatus

log = get_logger(__name__)

REASON_DISAGREEMENT = "llm_keyword_disagreement"
REASON_EVIDENCE = "evidence_not_in_headline"
REASON_INVALID = "llm_invalid_output"


class HeadlineClassifier:
    def __init__(self, issues_cfg: IssuesConfig, llm: OllamaClassifier | None,
                 conn: sqlite3.Connection | None = None, breaker_failures: int = 3) -> None:
        self.issues_cfg = issues_cfg
        self.keyword = KeywordClassifier(issues_cfg)
        self.llm = llm
        self.conn = conn
        self.breaker_failures = breaker_failures
        self._consecutive_failures = 0
        self.llm_disabled_reason: str | None = None if llm else "LLM disabled"

    @property
    def llm_active(self) -> bool:
        return self.llm is not None and self.llm_disabled_reason is None

    def _fallback(self, kw: KeywordResult, status: ValidationStatus, error: str | None,
                  raw: str | None = None, latency_ms: float | None = None) -> ClassificationOutcome:
        outcome = ClassificationOutcome(
            primary_issue=kw.issue, secondary_issue=None, evidence=kw.evidence,
            method=ClassificationMethod.KEYWORD_FALLBACK, validation_status=status,
            keyword_issue=kw.issue, model_id=self.llm._model_id if self.llm else None,
            raw_response=raw, prompt_version=self.llm.prompt_version if self.llm else None,
            latency_ms=latency_ms, error=error,
        )
        if status is ValidationStatus.LLM_INVALID:
            outcome.review_reasons.append(REASON_INVALID)
        return outcome

    def _record_unavailable(self, error: str) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.breaker_failures and self.llm_disabled_reason is None:
            self.llm_disabled_reason = f"circuit breaker open after {self._consecutive_failures} failures: {error}"
            log.warning("llm_circuit_open", extra=fields(reason=self.llm_disabled_reason))

    def _llm_outcome(self, headline: str, kw: KeywordResult, issue: str, secondary: str | None,
                     evidence: str, raw: str, model_id: str, latency_ms: float | None,
                     from_cache: bool) -> ClassificationOutcome:
        verified = evidence_is_verified(evidence, headline)
        outcome = ClassificationOutcome(
            primary_issue=issue, secondary_issue=secondary, evidence=evidence,
            method=ClassificationMethod.LLM,
            validation_status=ValidationStatus.VALID if verified else ValidationStatus.EVIDENCE_UNVERIFIED,
            keyword_issue=kw.issue, model_id=model_id, raw_response=raw,
            prompt_version=self.llm.prompt_version if self.llm else None,
            latency_ms=latency_ms, from_cache=from_cache,
        )
        if not verified:
            outcome.review_reasons.append(REASON_EVIDENCE)
        if issue != kw.issue:
            outcome.review_reasons.append(REASON_DISAGREEMENT)
        return outcome

    def classify(self, headline: str, headline_hash: str | None = None) -> ClassificationOutcome:
        kw = self.keyword.classify(headline)
        if not self.llm_active:
            return self._fallback(kw, ValidationStatus.LLM_UNAVAILABLE, self.llm_disabled_reason)
        assert self.llm is not None
        try:
            model_id = self.llm.resolve_model_id()
        except OllamaUnavailable as exc:
            self._record_unavailable(str(exc))
            return self._fallback(kw, ValidationStatus.LLM_UNAVAILABLE, str(exc))

        if self.conn is not None and headline_hash:
            cached = db.get_cached_classification(self.conn, headline_hash, model_id, self.llm.prompt_version)
            if cached is not None:
                return self._llm_outcome(headline, kw, cached["primary_issue"], cached["secondary_issue"],
                                         cached["evidence"] or "", cached["raw_response"] or "",
                                         model_id, None, from_cache=True)
        try:
            result = self.llm.classify(headline)
        except OllamaUnavailable as exc:
            self._record_unavailable(str(exc))
            return self._fallback(kw, ValidationStatus.LLM_UNAVAILABLE, str(exc))
        except LLMResponseError as exc:
            self._consecutive_failures = 0  # server is up; the answer was just bad
            log.warning("llm_invalid_output", extra=fields(headline=headline[:80], error=str(exc)[:200]))
            return self._fallback(kw, ValidationStatus.LLM_INVALID, str(exc), raw=exc.raw)
        self._consecutive_failures = 0
        outcome = self._llm_outcome(headline, kw, result.parsed.issue, result.parsed.secondary,
                                    result.parsed.evidence, result.raw, model_id,
                                    result.latency_ms, from_cache=False)
        if self.conn is not None and headline_hash:
            db.put_cached_classification(self.conn, headline_hash, outcome)
        return outcome
