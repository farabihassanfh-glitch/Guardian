"""Compute every underwriting figure once, up front.

Both the agents and the deterministic policy gate read from this single result,
so a prompt and a hard rule can never be looking at different numbers.
"""

from __future__ import annotations

from typing import Any

from app.tools.calculators import monthly_debt_total


def compute_metrics(sanitized: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical numeric picture of an application."""
    employment = sanitized.get("employment", {}) or {}
    loan = sanitized.get("loan", {}) or {}
    prop = sanitized.get("property", {}) or {}
    assets = sanitized.get("assets", {}) or {}

    income = float(employment.get("monthly_income", 0) or 0)
    piti = float(loan.get("estimated_payment", loan.get("monthly_piti", 0)) or 0)
    existing_debt = monthly_debt_total(sanitized.get("debts", {}))

    liquid = float(assets.get("checking", 0) or 0) + float(assets.get("savings", 0) or 0)
    down_payment = float(loan.get("down_payment", 0) or 0)
    closing_costs = float(loan.get("closing_costs", 0) or 0)
    # Policy 3.1: reserves are what remains *after* closing. Measuring gross
    # liquidity instead overstates every borrower's cushion -- Sarah looks like
    # she has 58 months of reserves when the real figure is 23.
    post_closing = max(0.0, liquid - down_payment - closing_costs)

    loan_amount = float(loan.get("amount", 0) or 0)
    appraised = float(prop.get("appraised_value", 0) or 0)
    purchase = float(prop.get("purchase_price", 0) or 0)
    # Policy 4.3 values the collateral at the lesser of price and appraisal.
    basis = min(v for v in (appraised, purchase) if v) if (appraised or purchase) else 0

    total_obligations = existing_debt + piti

    return {
        "monthly_income": income,
        "monthly_piti": piti,
        "existing_debt": existing_debt,
        "total_obligations": total_obligations,
        # Back-end DTI *includes* the proposed housing payment (policy 2.6).
        "dti_ratio": (total_obligations / income * 100) if income else None,
        "housing_ratio": (piti / income * 100) if income else None,
        "liquid_assets": liquid,
        "funds_to_close": down_payment + closing_costs,
        "post_closing_liquidity": post_closing,
        "reserve_months": (post_closing / piti) if piti else None,
        #: Policy 4.3 values the collateral at the lesser of price and appraisal.
        #: Exposed so the collateral agent and the decision agent cannot end up
        #: quoting two different LTVs for the same file.
        "collateral_basis": basis,
        "loan_amount": loan_amount,
        "ltv_ratio": (loan_amount / basis * 100) if basis else None,
        "down_payment_pct": (
            float(loan.get("down_payment", 0) or 0) / purchase * 100 if purchase else None
        ),
        "credit_score": sanitized.get("credit_score"),
    }
