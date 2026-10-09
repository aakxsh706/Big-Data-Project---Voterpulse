import json

import pytest
import requests

from conftest import FakeResponse, FakeSession, ollama_answer, ollama_chat, ollama_tags
from voterpulse import database as db
from voterpulse.classification.keyword_classifier import KeywordClassifier
from voterpulse.classification.llm_classifier import OllamaClassifier, OllamaUnavailable
from voterpulse.classification.prompts import build_system_prompt, prompt_version, response_schema
from voterpulse.classification.service import (REASON_DISAGREEMENT, REASON_EVIDENCE, REASON_INVALID,
                                               HeadlineClassifier)
from voterpulse.classification.validator import LLMResponseError, evidence_is_verified, validate_response
from voterpulse.models import ClassificationMethod, ValidationStatus

FARM = "Farmers and Agriculture"
NOT = "Not an election issue"


# ------------------------------------------------------------------ keyword baseline
@pytest.mark.parametrize("headline,expected", [
    ("Power struggle in state cabinet", NOT),
    ("MSP hike fails to calm mandi protests", FARM),
    ("Free laptops and bus rides promised during election campaign", "Health and Education"),
    ("Party promises 300 free units of electricity", "Electricity and Utilities"),
    ("Load-shedding returns as polls near", "Electricity and Utilities"),
    ("UNEMPLOYMENT rate rises among graduates", "Jobs and Unemployment"),
    ("LPG cylinder price hike angers households", "Prices and Inflation"),
    ("Police step up security after riots", "Law and Order"),
    ("Cricket team wins series in Australia", NOT),
    ("Jobless youth and job fairs", "Jobs and Unemployment"),
])
def test_keyword_examples(cfg, headline, expected):
    assert KeywordClassifier(cfg.issues).classify(headline).issue == expected


def test_keyword_word_boundaries(cfg):
    kc = KeywordClassifier(cfg.issues)
    assert kc.classify("Farmhouse party for MLAs").issue == NOT  # 'farm' inside a word / exclusion
    assert kc.classify("Schooling of opinions").issue == NOT     # 'school' must be a whole word


def test_keyword_tie_break_is_deterministic(cfg):
    kc = KeywordClassifier(cfg.issues)
    # 'jobs' (2) vs 'hospitals' (2): equal score and phrase count -> config order (Jobs first).
    result = kc.classify("jobs and hospitals")
    assert result.scores["Jobs and Unemployment"] == result.scores["Health and Education"]
    assert result.issue == "Jobs and Unemployment"
    assert all(kc.classify("jobs and hospitals").issue == result.issue for _ in range(5))


def test_keyword_evidence(cfg):
    result = KeywordClassifier(cfg.issues).classify("MSP hike fails to calm mandi protests")
    assert "MSP" in result.evidence and "mandi" in result.evidence


# ------------------------------------------------------------------ prompt
def test_prompt_generated_from_config(cfg):
    prompt = build_system_prompt(cfg.issues)
    for issue in cfg.issues.issues:
        assert issue.name in prompt
    assert "Power struggle in state cabinet" in prompt   # irrelevant example
    assert "MSP hike fails to calm mandi protests" in prompt
    schema = response_schema(cfg.issues)
    assert schema["properties"]["issue"]["enum"] == cfg.issues.all_labels
    assert len(prompt_version(cfg.issues)) == 12


# ------------------------------------------------------------------ validator
def test_validator_accepts_valid(cfg):
    parsed = validate_response(json.dumps({"issue": FARM, "evidence": "MSP hike", "secondary": None}), cfg.issues)
    assert parsed.issue == FARM and parsed.secondary is None


@pytest.mark.parametrize("raw", [
    "not json",
    "[1, 2]",
    json.dumps({"issue": "Sports", "evidence": "x", "secondary": None}),
    json.dumps({"issue": FARM, "evidence": "x"}),                                  # missing key
    json.dumps({"issue": FARM, "evidence": "x", "secondary": None, "extra": 1}),   # extra key
    json.dumps({"issue": FARM, "evidence": "x", "secondary": FARM}),               # duplicate
    json.dumps({"issue": FARM, "evidence": "x", "secondary": "Weather"}),
    json.dumps({"issue": NOT, "evidence": "x", "secondary": FARM}),
    json.dumps({"issue": 3, "evidence": "x", "secondary": None}),
])
def test_validator_rejects_invalid(cfg, raw):
    with pytest.raises(LLMResponseError) as info:
        validate_response(raw, cfg.issues)
    assert info.value.raw == raw


def test_evidence_verification():
    h = "MSP hike fails to calm mandi protests"
    assert evidence_is_verified("MSP hike ... mandi protests", h)
    assert evidence_is_verified("msp HIKE", h)
    assert evidence_is_verified("“MSP hike”", h)
    assert not evidence_is_verified("minimum support price", h)
    assert not evidence_is_verified("", h)
    assert not evidence_is_verified("MSP hike ... farmers", h)


# ------------------------------------------------------------------ Ollama client
def _client(cfg, post, get=None, no_sleep=lambda s: None):
    session = FakeSession(get=get or (lambda url, **kw: ollama_tags()), post=post)
    return OllamaClassifier.from_config(cfg, session=session, sleep=no_sleep), session


def test_llm_request_is_deterministic_and_structured(cfg):
    clf, session = _client(cfg, lambda url, payload, **kw: ollama_answer(FARM, "MSP hike"))
    result = clf.classify("MSP hike fails to calm mandi protests")
    assert result.parsed.issue == FARM
    payload = session.post_calls[0]
    assert payload["options"]["temperature"] == 0
    assert payload["format"]["properties"]["issue"]["enum"][-1] == NOT
    assert payload["stream"] is False
    assert clf.resolve_model_id() == "llama3.2:3b@a80c4f17acd5"


def test_llm_retries_transient_then_succeeds(cfg):
    calls = iter([requests.Timeout("slow"), FakeResponse(500, {"error": "x"}), ollama_answer(FARM, "MSP")])
    clf, session = _client(cfg, lambda url, payload, **kw: next(calls))
    assert clf.classify("MSP row").parsed.issue == FARM
    assert len(session.post_calls) == 3


def test_llm_unavailable_after_bounded_retries(cfg):
    clf, session = _client(cfg, lambda url, payload, **kw: requests.ConnectionError("refused"))
    with pytest.raises(OllamaUnavailable):
        clf.classify("anything")
    assert len(session.post_calls) == cfg.model.ollama.max_retries + 1


def test_model_not_pulled(cfg):
    clf, _ = _client(cfg, None, get=lambda url, **kw: FakeResponse(200, {"models": []}))
    with pytest.raises(OllamaUnavailable, match="ollama pull"):
        clf.resolve_model_id()


# ------------------------------------------------------------------ service / fallback
def test_service_llm_path_with_agreement(cfg):
    clf, _ = _client(cfg, lambda url, payload, **kw: ollama_answer(FARM, "MSP hike ... mandi protests"))
    out = HeadlineClassifier(cfg.issues, clf).classify("MSP hike fails to calm mandi protests")
    assert out.method is ClassificationMethod.LLM
    assert out.validation_status is ValidationStatus.VALID
    assert out.model_id == "llama3.2:3b@a80c4f17acd5"
    assert out.review_reasons == []
    assert json.loads(out.raw_response)["issue"] == FARM


def test_service_power_struggle_llm_overrides_nothing(cfg):
    clf, _ = _client(cfg, lambda url, payload, **kw: ollama_answer(NOT, "Power struggle"))
    out = HeadlineClassifier(cfg.issues, clf).classify("Power struggle in state cabinet")
    assert out.primary_issue == NOT and out.keyword_issue == NOT and not out.review_reasons


def test_service_disagreement_keeps_llm_and_queues_review(cfg):
    clf, _ = _client(cfg, lambda url, payload, **kw: ollama_answer("Health and Education", "Free laptops"))
    out = HeadlineClassifier(cfg.issues, clf).classify("Free laptops and jobs promised in campaign")
    assert out.primary_issue == "Health and Education"
    assert out.keyword_issue == "Jobs and Unemployment"
    assert REASON_DISAGREEMENT in out.review_reasons


def test_service_flags_unverifiable_evidence(cfg):
    clf, _ = _client(cfg, lambda url, payload, **kw: ollama_answer(FARM, "minimum support price"))
    out = HeadlineClassifier(cfg.issues, clf).classify("MSP hike fails to calm mandi protests")
    assert out.method is ClassificationMethod.LLM
    assert out.validation_status is ValidationStatus.EVIDENCE_UNVERIFIED
    assert REASON_EVIDENCE in out.review_reasons


def test_service_invalid_output_falls_back_and_keeps_raw(cfg):
    clf, _ = _client(cfg, lambda url, payload, **kw: ollama_chat('{"issue": "Weather"}'))
    out = HeadlineClassifier(cfg.issues, clf).classify("MSP hike fails to calm mandi protests")
    assert out.method is ClassificationMethod.KEYWORD_FALLBACK
    assert out.validation_status is ValidationStatus.LLM_INVALID
    assert out.primary_issue == FARM            # from keywords, labelled honestly
    assert out.raw_response == '{"issue": "Weather"}'
    assert REASON_INVALID in out.review_reasons


def test_service_ollama_down_uses_fallback_and_circuit_breaker(cfg):
    session = FakeSession(get=lambda url, **kw: requests.ConnectionError("refused"))
    clf = OllamaClassifier.from_config(cfg, session=session, sleep=lambda s: None)
    service = HeadlineClassifier(cfg.issues, clf, breaker_failures=2)
    for _ in range(5):
        out = service.classify("Youth unemployment rises")
        assert out.method is ClassificationMethod.KEYWORD_FALLBACK
        assert out.validation_status is ValidationStatus.LLM_UNAVAILABLE
        assert out.primary_issue == "Jobs and Unemployment"
    assert len(session.get_calls) == 2           # breaker stopped further attempts
    assert service.llm_disabled_reason is not None


def test_service_without_llm(cfg):
    out = HeadlineClassifier(cfg.issues, None).classify("Police crackdown on crime")
    assert out.method is ClassificationMethod.KEYWORD_FALLBACK and out.primary_issue == "Law and Order"


def test_cache_prevents_repeat_llm_calls(cfg, conn):
    clf, session = _client(cfg, lambda url, payload, **kw: ollama_answer(FARM, "MSP hike"))
    service = HeadlineClassifier(cfg.issues, clf, conn)
    first = service.classify("MSP hike fails to calm mandi protests", "hash1")
    second = service.classify("MSP hike fails to calm mandi protests", "hash1")
    assert len(session.post_calls) == 1
    assert second.from_cache and second.method is ClassificationMethod.LLM
    assert second.primary_issue == first.primary_issue


def test_fallback_results_are_not_cached(cfg, conn):
    clf, _ = _client(cfg, lambda url, payload, **kw: ollama_chat("garbage"))
    HeadlineClassifier(cfg.issues, clf, conn).classify("MSP row", "hash2")
    assert conn.execute("SELECT COUNT(*) FROM classification_cache").fetchone()[0] == 0
