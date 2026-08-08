"""Final synthesis into a decision, a risk score and a credit memo.

This agent returns **structured output**, not prose to be parsed.

The original implementation asked for a free-text memo and then scraped it::

    if "APPROVED" in content and "CONDITIONAL" not in content: ...
    elif "DENIED" in content: ...

which classifies a memo containing the sentence "the application was not denied"
as a denial. Substring matching over generated prose is a category of bug you
can design away: define a schema, let the provider validate against it, and the
parsing step disappears.
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.llm import get_llm
from app.state import UnderwritingState


class UnderwritingDecision(BaseModel):
    """The schema the model is required to return."""

    risk_score: int = Field(
        ge=0, le=100, description="0 = lowest risk, 100 = highest risk"
    )
    decision: Literal["APPROVED", "CONDITIONAL_APPROVAL", "DENIED"]
    conditions: list[str] = Field(
        default_factory=list,
        description="Stipulations to clear before closing. Empty for a clean approval.",
    )
    principal_reasons: list[str] = Field(
        default_factory=list,
        description="For a denial, the reasons for the ECOA adverse action notice.",
    )
    credit_memo: str = Field(description="Audit-ready narrative justifying the decision.")


SYSTEM = """You are a Senior Mortgage Underwriter with final decision authority.

Synthesise the specialist analyses and the QA review into one defensible decision.

RISK SCORE BANDS — the score you assign must agree with the decision you make:
- 0-29   APPROVED             meets all standards, no material conditions
- 30-70  CONDITIONAL_APPROVAL sound file, specific stipulations required
- 71-100 DENIED               does not meet lending standards

RULES:
- Any hard policy violation listed below is disqualifying on its own. If violations are
  present, the decision is DENIED and the score is at least 71.
- Cite figures — credit score, DTI, LTV, reserves — for every conclusion you draw.
- Decide only on financial, employment, asset and property grounds. Never reference or
  infer protected characteristics (Fair Lending Act, ECOA).
- For CONDITIONAL_APPROVAL, list conditions that are specific and clearable.
- For DENIED, populate principal_reasons for the adverse action notice.
- The credit memo must let a regulator follow your reasoning with no other context."""

USER = """Case: {case_id}

CREDIT ANALYSIS:
{credit}

INCOME ANALYSIS:
{income}

ASSET ANALYSIS:
{asset}

COLLATERAL ANALYSIS:
{collateral}

QA REVIEW:
{critic}

DETERMINISTIC METRICS (authoritative):
{metrics}

HARD POLICY VIOLATIONS DETECTED IN CODE:
{violations}

BIAS FLAGS:
{bias}

Issue your decision."""


def _fmt_metrics(m: dict[str, Any]) -> str:
    def pct(key: str) -> str:
        v = m.get(key)
        return f"{v:.2f}%" if isinstance(v, (int, float)) else "n/a"

    def num(key: str, suffix: str = "") -> str:
        v = m.get(key)
        return f"{v:.1f}{suffix}" if isinstance(v, (int, float)) else "n/a"

    return "\n".join(
        [
            f"- Credit score: {m.get('credit_score', 'n/a')}",
            f"- Back-end DTI: {pct('dti_ratio')}",
            f"- Housing ratio: {pct('housing_ratio')}",
            f"- LTV: {pct('ltv_ratio')}",
            f"- Down payment: {pct('down_payment_pct')}",
            f"- Reserves: {num('reserve_months', ' months')}",
        ]
    )


def decision_node(state: UnderwritingState) -> dict[str, Any]:
    """Produce the final decision, risk score, conditions and credit memo."""
    violations = state.get("policy_violations") or []

    prompt = USER.format(
        case_id=state.get("case_id", "unknown"),
        credit=state.get("credit_analysis") or "n/a",
        income=state.get("income_analysis") or "n/a",
        asset=state.get("asset_analysis") or "n/a",
        collateral=state.get("collateral_analysis") or "n/a",
        critic=state.get("critic_review") or "n/a",
        metrics=_fmt_metrics(state.get("metrics", {})),
        violations="\n".join(f"- {v}" for v in violations) or "none",
        bias=state.get("bias_flags") or "none",
    )

    model = get_llm().with_structured_output(UnderwritingDecision)
    result: UnderwritingDecision = model.invoke(
        [SystemMessage(content=SYSTEM), HumanMessage(content=prompt)]
    )

    # Belt and braces: a hard violation forces a denial regardless of what the
    # model concluded. Policy gates are not advisory.
    decision = result.decision
    risk_score = result.risk_score
    if violations and decision != "DENIED":
        decision = "DENIED"
        risk_score = max(risk_score, 71)

    reasons: list[str] = []
    if risk_score >= 65:
        reasons.append(f"Risk score {risk_score} at or above the 65 escalation threshold")
    if state.get("bias_flags"):
        reasons.append("Fair Lending flags raised during analysis")
    if decision == "DENIED":
        reasons.append("Denials require human sign-off before an adverse action notice issues")

    return {
        "final_decision": decision,
        "risk_score": risk_score,
        "conditions": result.conditions,
        "decision_memo": result.credit_memo,
        "human_review_required": bool(reasons),
        "human_review_reasons": reasons,
        "reasoning_chain": [f"Decision: {decision} at risk score {risk_score}"],
    }
