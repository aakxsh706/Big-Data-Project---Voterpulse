"""Deterministic geographic scope check, applied after issue classification.

Rule: a headline is *out of region* when it names at least one place from
``region.other_places`` and none of ``region.local_terms``. Everything else is in
region — including headlines that name no place at all, because the RSS searches
already target the configured region. Matching is case-insensitive on word
boundaries, except all-caps abbreviations (e.g. "TN") which match case-sensitively
so that ordinary words are not mistaken for them.

The small LLM is also told about the region, but tests showed it often keeps
headlines about other states; this filter makes the explicit cases reliable and
auditable (the matched place is stored).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from voterpulse.config import RegionSettings

IN_REGION = "in_region"
OUT_OF_REGION = "out_of_region"
NOT_CHECKED = "not_checked"


def _pattern(terms: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    compiled = []
    for term in terms:
        body = r"[\s\-]+".join(re.escape(w) for w in term.split())
        flags = 0 if (term.isupper() and len(term) <= 4) else re.IGNORECASE
        compiled.append((term, re.compile(rf"(?<!\w){body}(?!\w)", flags)))
    return compiled


@dataclass(frozen=True)
class RegionCheck:
    status: str
    evidence: str | None  # the matched place(s), for audit


class RegionFilter:
    def __init__(self, region: RegionSettings) -> None:
        self.name = region.name
        self._local = _pattern(region.local_terms)
        self._other = _pattern(region.other_places)

    @property
    def active(self) -> bool:
        return bool(self.name and self._other)

    def check(self, headline: str) -> RegionCheck:
        if not self.active:
            return RegionCheck(NOT_CHECKED, None)
        local = [t for t, p in self._local if p.search(headline)]
        if local:
            return RegionCheck(IN_REGION, ", ".join(local))
        other = [t for t, p in self._other if p.search(headline)]
        if other:
            return RegionCheck(OUT_OF_REGION, ", ".join(other))
        return RegionCheck(IN_REGION, None)
