"""Ollama client for local issue classification.

Reusable on its own (the evaluation scripts use it directly)::

    clf = OllamaClassifier.from_config(get_config())
    result = clf.classify("MSP hike fails to calm mandi protests")
    result.parsed.issue  # -> "Farmers and Agriculture"
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import requests

from voterpulse.config import AppConfig, IssuesConfig, OllamaConfig
from voterpulse.classification.prompts import (
    build_system_prompt,
    build_user_prompt,
    prompt_version,
    response_schema,
)
from voterpulse.classification.validator import LLMClassification, LLMResponseError, validate_response
from voterpulse.logging_utils import fields, get_logger

log = get_logger(__name__)


class OllamaUnavailable(RuntimeError):
    """Ollama could not be reached, timed out, or does not have the model."""


@dataclass
class LLMResult:
    parsed: LLMClassification
    raw: str
    latency_ms: float


class OllamaClassifier:
    def __init__(self, ollama_cfg: OllamaConfig, options: dict[str, Any], issues_cfg: IssuesConfig,
                 session: requests.Session | None = None,
                 sleep: Callable[[float], None] = time.sleep, region: str | None = None) -> None:
        self.cfg = ollama_cfg
        self.options = dict(options)
        self.options["temperature"] = 0  # fixed by design; not overridable
        self.issues_cfg = issues_cfg
        self.session = session or requests.Session()
        self._sleep = sleep
        self.region = region
        self.system_prompt = build_system_prompt(issues_cfg, region)
        self.schema = response_schema(issues_cfg)
        self.prompt_version = prompt_version(issues_cfg, region)
        self._model_id: str | None = None

    @classmethod
    def from_config(cls, cfg: AppConfig, **kwargs: Any) -> "OllamaClassifier":
        kwargs.setdefault("region", cfg.settings.region.name)
        return cls(cfg.model.ollama, cfg.model.options, cfg.issues, **kwargs)

    @property
    def base_url(self) -> str:
        return self.cfg.host.rstrip("/")

    # ------------------------------------------------------------------ model identity
    def resolve_model_id(self) -> str:
        """Return ``"<model>@<digest12>"`` from the local Ollama install.

        Raises :class:`OllamaUnavailable` if the server is down or the model is not pulled.
        """
        if self._model_id:
            return self._model_id
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
        except (requests.RequestException, ValueError) as exc:
            raise OllamaUnavailable(f"cannot reach Ollama at {self.base_url}: {exc}") from exc
        wanted = self.cfg.model
        candidates = {wanted, wanted if ":" in wanted else f"{wanted}:latest"}
        for model in models:
            if model.get("name") in candidates or model.get("model") in candidates:
                digest = (model.get("digest") or "unknown")[:12]
                expected = self.cfg.expected_digest
                if expected and not digest.startswith(expected[:12]):
                    log.warning("model_digest_mismatch",
                                extra=fields(model=wanted, expected=expected, actual=digest))
                self._model_id = f"{wanted}@{digest}"
                return self._model_id
        raise OllamaUnavailable(f"model {wanted!r} is not installed; run: ollama pull {wanted}")

    def is_available(self) -> bool:
        try:
            self.resolve_model_id()
            return True
        except OllamaUnavailable:
            return False

    # ------------------------------------------------------------------ classification
    def _payload(self, headline: str) -> dict[str, Any]:
        return {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": build_user_prompt(headline)},
            ],
            "format": self.schema,
            "stream": False,
            "keep_alive": self.cfg.keep_alive,
            "options": self.options,
        }

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /api/chat with bounded exponential backoff on transient failures."""
        last_error = "unknown"
        for attempt in range(self.cfg.max_retries + 1):
            try:
                response = self.session.post(f"{self.base_url}/api/chat", json=payload,
                                             timeout=self.cfg.timeout_seconds)
                if response.status_code == 404:
                    raise OllamaUnavailable(f"model {self.cfg.model!r} not found on Ollama server")
                if response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                elif response.status_code >= 400:
                    raise OllamaUnavailable(f"HTTP {response.status_code}: {response.text[:200]}")
                else:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise LLMResponseError(f"Ollama returned non-JSON body: {exc}",
                                               response.text[:2000]) from exc
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.cfg.max_retries:
                delay = min(self.cfg.backoff_seconds * (2 ** attempt), self.cfg.backoff_max_seconds)
                log.info("llm_retry", extra=fields(attempt=attempt + 1, delay_s=delay, error=last_error))
                self._sleep(delay)
        raise OllamaUnavailable(f"Ollama request failed after retries: {last_error}")

    def classify(self, headline: str) -> LLMResult:
        """Classify one headline.

        Raises :class:`OllamaUnavailable` (transport problems) or
        :class:`LLMResponseError` (model answered with invalid content; ``.raw`` is kept).
        """
        start = time.perf_counter()
        body = self._post_chat(self._payload(headline))
        latency_ms = (time.perf_counter() - start) * 1000
        raw = (body.get("message") or {}).get("content")
        if not isinstance(raw, str):
            raise LLMResponseError("response has no message content", json.dumps(body)[:2000])
        parsed = validate_response(raw, self.issues_cfg)
        return LLMResult(parsed=parsed, raw=raw, latency_ms=latency_ms)
