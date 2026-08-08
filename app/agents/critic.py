"""Quality assurance pass over the four specialist memos.

The critic sees no application data and no policy text -- only what the
specialists wrote. Its job is not to re-underwrite the file but to catch the
failure the specialists structurally cannot: a contradiction *between* them.
Four agents each reasoning correctly in isolation can still produce a file where
credit says "low risk" and income says "cannot service the debt".

A second pass whose only job is to attack the first is the cheapest reliability
win available in an LLM system.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import get_llm
from app.state import UnderwritingState

SYSTEM = """You are a Quality Assurance reviewer in a mortgage underwriting shop.

You are reviewing four specialist memos. You do NOT have the underlying file, and you
must not invent facts that are not in the memos.

Your job:
1. Completeness — is any required analysis missing or superficial?
2. Consistency — do any two memos contradict each other? Quote both sides if so.
3. Policy compliance — is any cited policy applied incorrectly?
4. Gaps — what would an experienced underwriter ask for that nobody requested?
5. Synthesis — three to five sentences on where this file actually stands.

Be specific and sceptical. If the analyses are sound, say so plainly rather than
manufacturing criticism. Aim for 250 words."""

USER = """Case: {case_id}

CREDIT ANALYSIS:
{credit}

INCOME ANALYSIS:
{income}

ASSET ANALYSIS:
{asset}

COLLATERAL ANALYSIS:
{collateral}

AUTOMATED COMPLIANCE FLAGS:
- Bias flags raised: {bias}
- Hard policy violations detected in code: {violations}

Provide your review."""


def critic_node(state: UnderwritingState) -> dict[str, Any]:
    """Review all specialist analyses for consistency and completeness."""
    missing = "not completed"
    prompt = USER.format(
        case_id=state.get("case_id", "unknown"),
        credit=state.get("credit_analysis") or missing,
        income=state.get("income_analysis") or missing,
        asset=state.get("asset_analysis") or missing,
        collateral=state.get("collateral_analysis") or missing,
        bias=state.get("bias_flags") or "none",
        violations=state.get("policy_violations") or "none",
    )

    response = get_llm().invoke(
        [SystemMessage(content=SYSTEM), HumanMessage(content=prompt)]
    )

    return {
        "critic_review": response.content,
        "reasoning_chain": ["Critic: cross-checked all specialist analyses"],
    }
