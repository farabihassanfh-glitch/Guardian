"""The four specialist desks, defined as data.

Each spec answers four questions: what policy do I need, what facts may I see,
what figures are settled for me, and what am I required to work through.
"""

from __future__ import annotations

import json
from typing import Any

from app.agents.base import SpecialistSpec, make_node
from app.tools import calculators as calc


def _money(v: Any) -> str:
    try:
        return f"${float(v):,.2f}"
    except (TypeError, ValueError):
        return "not provided"


def _dump(v: Any) -> str:
    return json.dumps(v, indent=2, default=str) if v else "none on file"


# --- Credit ---------------------------------------------------------------
CREDIT = SpecialistSpec(
    name="credit",
    title="Senior Credit Analyst",
    state_key="credit_analysis",
    rag_query="credit score minimums bankruptcy discharge foreclosure late payments collections",
    framework=(
        "Credit score assessment — use the provided tier, do not restate the score differently",
        "Payment history — evaluate lates by recency, severity and pattern",
        "Derogatory items — bankruptcies, foreclosures, collections, judgements",
        "Policy compliance — check each finding against the excerpts above",
        "Risk rating — Low / Medium / High with justification",
        "Recommendations — conditions required, or concerns that cannot be cleared",
    ),
    facts=lambda d: {
        "CREDIT HISTORY": _dump(d.get("credit_history")),
    },
    computed=lambda d, m: {
        "CREDIT SCORE TIER": calc.check_credit_score_policy.invoke(
            {"credit_score": d.get("credit_score", 0)}
        ),
    },
)

# --- Income ---------------------------------------------------------------
INCOME = SpecialistSpec(
    name="income",
    title="Senior Income Analyst",
    state_key="income_analysis",
    rag_query="employment history verification self-employment bonus commission DTI ratio limits",
    framework=(
        "Employment stability — tenure, continuity of field, gaps and their explanations",
        "Income verification — which sources qualify and which must be excluded",
        "DTI assessment — use the provided ratio, do not recompute it",
        "Payment capacity — can this borrower carry the proposed payment",
        "Risk assessment — identify income-specific risks",
        "Recommendations — conditions required, or concerns that cannot be cleared",
    ),
    facts=lambda d: {
        "EMPLOYMENT": _dump(d.get("employment")),
        "MONTHLY OBLIGATIONS": _dump(calc.debt_line_items(d.get("debts", {}))),
    },
    computed=lambda d, m: {
        "DEBT-TO-INCOME": calc.calculate_dti_ratio.invoke(
            {
                "monthly_debt": m.get("total_obligations", 0),
                "monthly_income": m.get("monthly_income", 0),
            }
        ),
        "HOUSING EXPENSE RATIO": calc.calculate_housing_expense_ratio.invoke(
            {
                "monthly_payment": m.get("monthly_piti", 0),
                "monthly_income": m.get("monthly_income", 0),
            }
        ),
        "TOTAL OBLIGATIONS": calc.calculate_total_debt_obligations.invoke(
            {"debts": d.get("debts", {}), "proposed_payment": m.get("monthly_piti", 0)}
        ),
    },
)

# --- Assets ---------------------------------------------------------------
ASSET = SpecialistSpec(
    name="asset",
    title="Senior Asset Analyst",
    state_key="asset_analysis",
    rag_query="down payment sourcing seasoning reserves large deposits gift funds retirement assets",
    framework=(
        "Down payment adequacy — are funds sufficient to close including costs",
        "Reserve requirements — use the provided calculation, do not recompute it",
        "Large deposits — use the provided analysis; state what documentation clears each",
        "Source of funds — sourcing and seasoning per policy 3.2 and 3.3",
        "Risk assessment — identify asset-specific risks",
        "Documentation needs — the exact conditions an underwriter would issue",
    ),
    facts=lambda d: {
        "ASSETS": _dump(d.get("assets")),
        "FUNDS REQUIRED TO CLOSE": (
            f"Down payment: {_money((d.get('loan') or {}).get('down_payment'))}\n"
            f"Closing costs: {_money((d.get('loan') or {}).get('closing_costs'))}"
        ),
    },
    computed=lambda d, m: {
        # Post-closing liquidity, not gross -- policy 3.1 counts what survives
        # the down payment and closing costs.
        "RESERVE COVERAGE": calc.calculate_reserves.invoke(
            {
                "liquid_assets": m.get("post_closing_liquidity", 0),
                "monthly_payment": m.get("monthly_piti", 0),
                "required_months": calc.RESERVES_MONTHS_REQUIRED,
            }
        ),
        "DEPOSIT SOURCING": calc.check_large_deposits.invoke(
            {
                "deposits": (d.get("assets") or {}).get("recent_deposits", []),
                "monthly_income": m.get("monthly_income", 0),
            }
        ),
    },
)

# --- Collateral -----------------------------------------------------------
COLLATERAL = SpecialistSpec(
    name="collateral",
    title="Senior Collateral Analyst",
    state_key="collateral_analysis",
    rag_query="appraisal requirements property condition habitability LTV limits repairs escrow holdback",
    framework=(
        "Appraisal review — value support and any price-to-appraisal gap",
        "LTV assessment — use the provided ratio, do not recompute it",
        "Property condition — habitability, safety, structural soundness per policy 4.2",
        "Repairs — cost against the $5,000 / 3% escrow holdback cap",
        "Risk assessment — marketability and collateral-specific risks",
        "Recommendations — conditions required prior to closing",
    ),
    facts=lambda d: {
        "PROPERTY": _dump(d.get("property")),
        "LOAN": _dump(d.get("loan")),
    },
    computed=lambda d, m: {
        # Uses the same basis the decision agent will see -- the lesser of
        # purchase price and appraised value, per policy 4.3.
        "LOAN-TO-VALUE": calc.calculate_ltv_ratio.invoke(
            {
                "loan_amount": m.get("loan_amount", 0),
                "property_value": m.get("collateral_basis", 0),
            }
        ),
    },
)


SPECIALISTS: tuple[SpecialistSpec, ...] = (CREDIT, INCOME, ASSET, COLLATERAL)

#: ``{"credit": <node fn>, ...}`` -- consumed by the graph builder.
SPECIALIST_NODES = {spec.name: make_node(spec) for spec in SPECIALISTS}

#: ``{"credit": "credit_analysis", ...}`` -- used by the supervisor to work out
#: which desks are still outstanding.
SPECIALIST_KEYS = {spec.name: spec.state_key for spec in SPECIALISTS}
