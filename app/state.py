"""Shared state for the underwriting workflow.

Every agent is a pure function ``state -> partial state``. LangGraph merges each
partial back into the shared state using the reducer declared on each field.

Two kinds of field live here:

* **Overwrite fields** (the default) -- ``final_decision``, ``risk_score``.
  There is one current value; a later write replaces the earlier one.
* **Accumulating fields** -- annotated with ``operator.add`` so writes are
  concatenated. These carry the audit trail.

The distinction matters more than it looks. Because ``reasoning_chain`` and
``bias_flags`` accumulate, agents return **only what they just produced**, never
a rebuilt copy of the whole list. That is what makes it safe to run agents
concurrently: two agents finishing at the same instant both contribute, instead
of the slower one clobbering the faster one.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict


class UnderwritingState(TypedDict, total=False):
    """The complete state of a loan application as it moves through the system."""

    # --- Application ------------------------------------------------------
    case_id: str
    applicant_data: dict[str, Any]
    #: PII-redacted projection of ``applicant_data``; the only thing agents see.
    sanitized_data: dict[str, Any]

    # --- Specialist findings ---------------------------------------------
    credit_analysis: Optional[str]
    income_analysis: Optional[str]
    asset_analysis: Optional[str]
    collateral_analysis: Optional[str]

    # --- Coordination and decision ---------------------------------------
    critic_review: Optional[str]
    decision_memo: Optional[str]
    final_decision: Optional[str]  # APPROVED | CONDITIONAL_APPROVAL | DENIED
    risk_score: Optional[int]  # 0-100, higher is riskier
    conditions: list[str]

    # --- Workflow control -------------------------------------------------
    next_agent: Optional[str]
    analysis_complete: bool
    human_review_required: bool
    human_review_reasons: list[str]

    # --- Compliance and audit (accumulating) ------------------------------
    bias_flags: Annotated[list[str], operator.add]
    policy_violations: Annotated[list[str], operator.add]
    reasoning_chain: Annotated[list[str], operator.add]

    # --- Deterministic figures, computed in Python, never by the model ----
    metrics: dict[str, Any]
    timestamp: str


#: Decision labels the system may emit. Kept in one place so the API, the UI and
#: the test fixtures cannot drift apart -- the original notebook had three
#: different vocabularies for the same three outcomes.
DECISIONS = ("APPROVED", "CONDITIONAL_APPROVAL", "DENIED")

#: Maps our internal labels onto the shorter labels used by the test fixtures.
FIXTURE_LABELS = {
    "APPROVED": "APPROVED",
    "CONDITIONAL_APPROVAL": "CONDITIONAL",
    "DENIED": "REJECTED",
}
