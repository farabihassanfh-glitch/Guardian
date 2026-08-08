"""PII redaction and Fair Lending guardrails.

Two guardrails, both plain Python, both sitting *outside* the model:

* :func:`sanitize_pii` limits what the model is allowed to see, before the call.
* :func:`detect_bias_signals` inspects what the model said, after the call.

Neither trusts the model, which is the point.
"""

from __future__ import annotations

import copy
import re
from typing import Any

#: Fields forwarded to the model. An **allowlist**, deliberately.
#:
#: A blocklist ("redact ssn, name, address, phone") only removes the identifiers
#: you remembered. The original implementation used one and leaked ``email`` --
#: which for the primary fixture was ``sarah.johnson@email.com``, i.e. the full
#: name that had just been redacted one line earlier. An allowlist fails closed:
#: a new field added upstream is withheld until someone consciously permits it.
UNDERWRITING_FIELDS = frozenset(
    {
        "case_id",
        "credit_score",
        "credit_history",
        "employment",
        "debts",
        "assets",
        "loan",
        "property",
        "dti_ratio",
    }
)

#: Identifiers kept in redacted form so a human can still match the file to a
#: record without the value being exposed.
PARTIAL_FIELDS = {"ssn": "***-**-{}", "phone": "***-***-{}"}

PROTECTED_TERMS = (
    "race",
    "color",
    "religion",
    "national origin",
    "ethnicity",
    "sex",
    "gender",
    "marital status",
    "age",
    "disability",
    "familial status",
    "pregnancy",
    "citizenship",
)

#: Terms that are legitimate underwriting vocabulary and must not trip the
#: detector even though they overlap a protected term.
_ALLOWED_CONTEXTS = re.compile(
    r"\b(?:age of the (?:property|home)|property age|loan age|average|coverage)\b",
    re.IGNORECASE,
)


def sanitize_pii(data: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copied, PII-redacted projection safe to send to a model."""
    sanitized: dict[str, Any] = {
        # Deep copy: a shallow ``.copy()`` leaves nested dicts aliased to the
        # original, so redacting one mutates the source record.
        key: copy.deepcopy(value)
        for key, value in data.items()
        if key in UNDERWRITING_FIELDS
    }

    for field, mask in PARTIAL_FIELDS.items():
        raw = str(data.get(field, ""))
        digits = re.sub(r"\D", "", raw)
        sanitized[field] = mask.format(digits[-4:] if len(digits) >= 4 else "XXXX")

    sanitized["name"] = "[APPLICANT_NAME]"
    sanitized["address"] = "[ADDRESS]"
    sanitized["email"] = "[EMAIL]"
    return sanitized


def detect_bias_signals(analysis: str, applicant_data: dict[str, Any] | None = None) -> list[str]:
    """Flag language that could indicate a Fair Lending Act problem.

    Matching is on **word boundaries**, not substrings. Substring matching looks
    equivalent and is not: ``"age" in "mortgage"`` is ``True``, so a substring
    detector fires on every mortgage document ever written. A guardrail that
    always fires carries no information and trains reviewers to ignore it.
    """
    flags: list[str] = []
    text = (analysis or "").lower()
    benign_spans = {m.group(0).lower() for m in _ALLOWED_CONTEXTS.finditer(text)}

    for term in PROTECTED_TERMS:
        for match in re.finditer(rf"\b{re.escape(term)}\b", text):
            window = text[max(0, match.start() - 25) : match.end() + 25]
            if any(span in window for span in benign_spans):
                continue
            flags.append(f"Protected characteristic referenced: '{term}'")
            break

    # Redlining proxy: geographic language combined with location data.
    if applicant_data and any(k in applicant_data for k in ("zip", "zipcode", "address", "census_tract")):
        if re.search(r"\b(neighborhood|neighbourhood|zip code|area demographics|part of town)\b", text):
            flags.append("Geographic language present - review for redlining risk")

    return flags


def policy_gate(sanitized: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    """Hard policy violations, decided in code before any model is consulted.

    Some rules are bright lines, not judgement calls. Encoding them here means
    they cannot be talked around by a persuasive analysis, and they hold even if
    the model is unavailable.
    """
    violations: list[str] = []
    credit = sanitized.get("credit_score", 0)
    history = sanitized.get("credit_history", {}) or {}

    if credit and credit < 620:
        violations.append(f"Credit score {credit} is below the 620 conventional minimum (policy 1.1)")

    if history.get("bankruptcies", 0) and history.get("bankruptcy_years_since_discharge") is not None:
        if history["bankruptcy_years_since_discharge"] < 2:
            violations.append("Bankruptcy discharged within the 2-year seasoning window (policy 1.2)")

    if history.get("late_payments_12mo", 0) > 2:
        violations.append(
            f"{history['late_payments_12mo']} late payments in 12 months exceeds the maximum of 2 (policy 1.4)"
        )

    dti = metrics.get("dti_ratio")
    if dti is not None and dti > 50:
        violations.append(f"Back-end DTI of {dti:.1f}% exceeds the 50% ceiling (policy 2.6)")

    repairs = (sanitized.get("property", {}) or {}).get("required_repairs", 0)
    if repairs and repairs > 5000:
        violations.append(
            f"Required repairs of ${repairs:,.0f} exceed the $5,000 escrow holdback cap (policy 4.2)"
        )

    return violations
