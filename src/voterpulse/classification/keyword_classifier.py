"""Deterministic keyword baseline classifier.

Used (1) as a cross-check on every LLM label and (2) as the fallback when the LLM is
unavailable. Rules, applied per issue:

* Phrases match case-insensitively on word boundaries; spaces inside a phrase match
  any whitespace or hyphen, so "load shedding" also matches "load-shedding".
* Text covered by one of the issue's *exclusion* phrases is masked before that
  issue's keywords are scored (e.g. "power struggle" never counts as electricity).
* Each distinct matched phrase adds its weight once.
* Highest score wins. Ties: more distinct matched phrases wins, then the issue that
  appears first in ``config/issues.yaml``.
* No match at all -> the "Not an election issue" label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from voterpulse.config import IssuesConfig
from voterpulse.models import KeywordResult


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    words = [re.escape(w) for w in re.split(r"[\s\-]+", phrase.strip()) if w]
    body = r"[\s\-]+".join(words)
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


def _parse_keyword(entry: str) -> tuple[str, float]:
    phrase, sep, weight = entry.rpartition("|")
    if sep and phrase:
        try:
            return phrase.strip(), float(weight)
        except ValueError as exc:
            raise ValueError(f"issues.yaml: bad keyword weight in {entry!r}") from exc
    return entry.strip(), 1.0


@dataclass(frozen=True)
class _CompiledIssue:
    name: str
    keywords: tuple[tuple[str, float, re.Pattern[str]], ...]
    exclusions: tuple[re.Pattern[str], ...]


class KeywordClassifier:
    def __init__(self, issues_cfg: IssuesConfig) -> None:
        self.not_relevant = issues_cfg.not_relevant_label
        compiled = []
        for issue in issues_cfg.issues:
            keywords = []
            for entry in issue.keywords:
                phrase, weight = _parse_keyword(entry)
                keywords.append((phrase, weight, _phrase_pattern(phrase)))
            compiled.append(_CompiledIssue(
                name=issue.name,
                keywords=tuple(keywords),
                exclusions=tuple(_phrase_pattern(p) for p in issue.exclusions),
            ))
        self._issues = tuple(compiled)

    def classify(self, headline: str) -> KeywordResult:
        scores: dict[str, float] = {}
        matched: dict[str, list[str]] = {}
        for issue in self._issues:
            text = headline
            for pattern in issue.exclusions:
                text = pattern.sub(lambda m: " " * len(m.group(0)), text)
            hits: list[str] = []
            score = 0.0
            for _phrase, weight, pattern in issue.keywords:
                found = pattern.search(text)
                if found:
                    hits.append(found.group(0))
                    score += weight
            if hits:
                scores[issue.name] = score
                matched[issue.name] = hits
        if not scores:
            return KeywordResult(issue=self.not_relevant, scores={}, matched={})
        order = {issue.name: index for index, issue in enumerate(self._issues)}
        best = min(scores, key=lambda name: (-scores[name], -len(matched[name]), order[name]))
        return KeywordResult(issue=best, scores=scores, matched=matched)
