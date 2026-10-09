"""LLM prompt and JSON schema, generated from ``config/issues.yaml``.

The prompt version is a short hash of the exact prompt text and schema; it is stored
with every LLM classification and is part of the cache key, so editing issues.yaml
automatically invalidates cached classifications made with the old prompt.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from voterpulse.config import IssuesConfig


def build_system_prompt(issues_cfg: IssuesConfig) -> str:
    lines = [
        "You classify Indian news headlines by the election policy issue they are about.",
        "",
        "Allowed issue labels:",
    ]
    for issue in issues_cfg.issues:
        lines.append(f'- "{issue.name}": {issue.description}')
    lines.append(
        f'- "{issues_cfg.not_relevant_label}": the headline is not about any of the issues above '
        "(for example sport, entertainment, party infighting, candidate lists, or election "
        "procedure with no policy content)."
    )
    lines += [
        "",
        "Rules:",
        "1. Choose exactly one primary \"issue\" from the allowed labels.",
        "2. Decide from the meaning of the whole headline, not from single words. "
        "\"Power\" meaning political control is not electricity.",
        "3. \"evidence\": copy the exact words from the headline that justify the label. "
        "Copy them character for character. Join separate spans with \" ... \".",
        "4. \"secondary\": almost always null. Only fill it when the headline clearly covers a "
        "second, DIFFERENT policy issue as well. Never repeat the primary issue in \"secondary\". "
        f"It must be null when the issue is \"{issues_cfg.not_relevant_label}\".",
        "5. Do not judge whether the news is good or bad. Only identify the issue.",
        "6. Reply with JSON only.",
        "",
        "Examples:",
    ]
    for ex in issues_cfg.prompt_examples:
        answer = {"issue": ex.issue, "evidence": ex.evidence, "secondary": ex.secondary}
        lines.append(f'Headline: "{ex.headline}"')
        lines.append(f"Answer: {json.dumps(answer, ensure_ascii=False)}")
    return "\n".join(lines)


def build_user_prompt(headline: str) -> str:
    return f'Headline: "{headline}"\nAnswer:'


def response_schema(issues_cfg: IssuesConfig) -> dict[str, Any]:
    """Strict JSON schema passed to Ollama's ``format`` parameter."""
    return {
        "type": "object",
        "properties": {
            "issue": {"type": "string", "enum": issues_cfg.all_labels},
            "evidence": {"type": "string"},
            "secondary": {
                "anyOf": [
                    {"type": "string", "enum": issues_cfg.issue_names},
                    {"type": "null"},
                ]
            },
        },
        "required": ["issue", "evidence", "secondary"],
        "additionalProperties": False,
    }


def prompt_version(issues_cfg: IssuesConfig) -> str:
    payload = build_system_prompt(issues_cfg) + json.dumps(response_schema(issues_cfg), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
