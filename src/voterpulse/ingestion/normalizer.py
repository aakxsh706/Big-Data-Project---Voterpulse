"""Headline cleaning (for display) and normalisation (for duplicate detection)."""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                         "–": "-", "—": "-", " ": " "})


def clean_headline(text: str, source: str | None = None) -> str:
    """Return a readable headline: HTML unescaped, tags removed, whitespace collapsed.

    Google News appends `` - Publisher`` to titles; that suffix is removed when the
    publisher is known so the stored headline is the publisher's own wording.
    """
    if not text:
        return ""
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFC", text).translate(_QUOTES)
    text = _WS_RE.sub(" ", text).strip()
    if source:
        suffix = f" - {source.strip()}"
        # Some publishers already end their own title with their name, so Google's
        # appended copy can make it appear twice; strip every trailing copy.
        while text.endswith(suffix) and len(text) > len(suffix):
            text = text[: -len(suffix)].rstrip()
    return text


def normalize_for_dedup(text: str) -> str:
    """Canonical form used only for hashing: case-folded, punctuation-free, single-spaced.

    Two headlines that differ only in case, quotes, dashes or punctuation produce the
    same normalised text and therefore the same hash.
    """
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES).casefold()
    text = _NON_WORD_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def headline_hash(normalized: str) -> str:
    """Stable SHA-256 hex digest of the normalised headline."""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
