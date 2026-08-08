"""Deterministic underwriting arithmetic.

Every figure an agent reasons about is computed here, in Python, and handed to
the model already settled. The model's job is judgement, never calculation.

The thresholds are lifted from ``data/underwriting_policies.pdf`` and are cited
inline. Keeping them in code rather than in a prompt makes them auditable and
means a policy change is a diff, not a re-read of a paragraph.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from langchain_core.tools import tool

# --- Policy thresholds ----------------------------------------------------
DTI_MAX_CONVENTIONAL = 43.0  # policy 2.6
DTI_MAX_WITH_FACTORS = 50.0  # policy 2.6
HOUSING_RATIO_MAX = 28.0  # policy 2.6 front-end
CREDIT_SCORE_MIN_CONVENTIONAL = 620  # policy 1.1
RESERVES_MONTHS_REQUIRED = 2  # policy 3.1, primary residence
#: Policy 3.3: "$1,000 **or** 25% of monthly income, whichever is less."
LARGE_DEPOSIT_FLOOR = 1000.0
LARGE_DEPOSIT_INCOME_SHARE = 0.25

#: Keys that appear inside a ``debts`` mapping but are roll-ups rather than
#: individual obligations. Summing them alongside the line items double-counts
#: the borrower's debt -- the single most consequential bug in the original
#: notebook, which inflated every applicant's DTI to roughly twice its true
#: value while every cell still reported success.
DEBT_SUMMARY_KEYS = frozenset({"total_monthly_debt", "total", "total_debt", "monthly_total"})


def monthly_debt_total(debts: dict) -> float:
    """Sum a borrower's monthly obligations, ignoring roll-up keys.

    >>> monthly_debt_total({"car_loan": 1200, "student_loan": 800,
    ...                     "credit_cards": 1800, "total_monthly_debt": 3800})
    3800.0
    """
    if not debts:
        return 0.0
    line_items = {k: v for k, v in debts.items() if k.lower() not in DEBT_SUMMARY_KEYS}
    # A payload carrying only a roll-up is still usable -- fall back to it.
    if not line_items:
        return float(sum(v for v in debts.values() if isinstance(v, (int, float))))
    return float(sum(v for v in line_items.values() if isinstance(v, (int, float))))


def debt_line_items(debts: dict) -> dict[str, float]:
    """The individual obligations, roll-up keys removed."""
    return {
        k: float(v)
        for k, v in (debts or {}).items()
        if k.lower() not in DEBT_SUMMARY_KEYS and isinstance(v, (int, float))
    }


@dataclass(frozen=True)
class Ratio:
    """A computed figure plus the verdict policy attaches to it."""

    value: float
    status: str
    detail: str

    def as_dict(self) -> dict:
        return asdict(self)


# --- Tools ----------------------------------------------------------------
# Each is decorated so an agent *could* call it directly, but in this workflow
# the orchestrator calls them first and injects the results into the prompt.
# That ordering is deliberate: it removes the model's opportunity to do
# arithmetic at all.


@tool
def calculate_dti_ratio(monthly_debt: float, monthly_income: float) -> str:
    """Calculate the back-end debt-to-income ratio.

    Args:
        monthly_debt: All monthly obligations *including* the proposed housing payment.
        monthly_income: Gross monthly income.
    """
    if monthly_income <= 0:
        return "Error: monthly income must be greater than 0"
    dti = (monthly_debt / monthly_income) * 100
    if dti <= DTI_MAX_CONVENTIONAL:
        status = "Acceptable"
    elif dti <= DTI_MAX_WITH_FACTORS:
        status = "High - requires compensating factors"
    else:
        status = "Excessive - exceeds policy maximum"
    return (
        f"DTI Ratio: {dti:.2f}% ({status}) - "
        f"Debt: ${monthly_debt:,.2f}, Income: ${monthly_income:,.2f}. "
        f"Policy 2.6 maximum is {DTI_MAX_CONVENTIONAL:.0f}% conventional."
    )


@tool
def calculate_ltv_ratio(loan_amount: float, property_value: float) -> str:
    """Calculate the loan-to-value ratio against appraised value.

    Args:
        loan_amount: Requested loan amount.
        property_value: Appraised value (not purchase price -- policy 4.3 uses the lesser).
    """
    if property_value <= 0:
        return "Error: property value must be greater than 0"
    ltv = (loan_amount / property_value) * 100
    if ltv <= 80:
        status = "Excellent - no mortgage insurance required"
    elif ltv <= 90:
        status = "Good"
    elif ltv <= 97:
        status = "High - mortgage insurance required"
    else:
        status = "Excessive - exceeds conventional maximum"
    return (
        f"LTV Ratio: {ltv:.2f}% ({status}) - "
        f"Loan: ${loan_amount:,.2f}, Appraised Value: ${property_value:,.2f}."
    )


@tool
def calculate_reserves(
    liquid_assets: float, monthly_payment: float, required_months: int = RESERVES_MONTHS_REQUIRED
) -> str:
    """Calculate post-closing reserve coverage in months of PITI.

    Args:
        liquid_assets: Liquid funds remaining after closing.
        monthly_payment: Monthly PITI.
        required_months: Months of reserves policy requires (2 for a conventional primary).
    """
    if monthly_payment <= 0:
        return "Error: monthly payment must be greater than 0"
    months = liquid_assets / monthly_payment
    required_amount = monthly_payment * required_months
    status = "Adequate" if months >= required_months else "Insufficient"
    return (
        f"Reserves: {months:.1f} months coverage ({status}) - "
        f"Available: ${liquid_assets:,.2f}, Required: ${required_amount:,.2f} "
        f"({required_months} months PITI per policy 3.1). "
        f"Surplus/deficit: ${liquid_assets - required_amount:,.2f}."
    )


@tool
def calculate_housing_expense_ratio(monthly_payment: float, monthly_income: float) -> str:
    """Calculate the front-end housing expense ratio.

    Args:
        monthly_payment: Monthly PITI.
        monthly_income: Gross monthly income.
    """
    if monthly_income <= 0:
        return "Error: monthly income must be greater than 0"
    ratio = (monthly_payment / monthly_income) * 100
    if ratio <= HOUSING_RATIO_MAX:
        status = "Acceptable"
    elif ratio <= 35:
        status = "Elevated"
    else:
        status = "High"
    return (
        f"Housing Ratio: {ratio:.2f}% ({status}) - "
        f"Payment: ${monthly_payment:,.2f}, Income: ${monthly_income:,.2f}. "
        f"Policy 2.6 front-end maximum is {HOUSING_RATIO_MAX:.0f}%."
    )


@tool
def check_credit_score_policy(credit_score: int) -> str:
    """Map a credit score onto its policy tier.

    Args:
        credit_score: The borrower's representative FICO score.
    """
    if credit_score >= 740:
        tier, note = "Excellent", "Best rates available"
    elif credit_score >= 700:
        tier, note = "Very Good", "Favorable rates"
    elif credit_score >= 660:
        tier, note = "Good", "Standard rates"
    elif credit_score >= CREDIT_SCORE_MIN_CONVENTIONAL:
        tier, note = "Fair", "Higher rates, may require compensating factors"
    else:
        tier, note = (
            "Below Minimum",
            f"Below the {CREDIT_SCORE_MIN_CONVENTIONAL} conventional minimum (policy 1.1)",
        )
    return f"Credit Score: {credit_score} - Tier: {tier} - {note}"


@tool
def check_large_deposits(deposits: list, monthly_income: float) -> str:
    """Identify deposits that require sourcing documentation.

    Args:
        deposits: ``[{"amount": float, "date": str, "description": str}, ...]``
        monthly_income: Gross monthly income, used for the threshold.
    """
    # Policy 3.3 says "whichever is less" -- the original implementation used
    # only the income share, so it silently ignored every deposit between
    # $1,000 and 25% of income.
    threshold = min(LARGE_DEPOSIT_FLOOR, monthly_income * LARGE_DEPOSIT_INCOME_SHARE)
    flagged = [d for d in (deposits or []) if float(d.get("amount", 0)) >= threshold]

    if not flagged:
        return f"No deposits above the ${threshold:,.2f} sourcing threshold (policy 3.3)."

    lines = [
        f"{len(flagged)} deposit(s) require sourcing documentation "
        f"(threshold ${threshold:,.2f}, policy 3.3):"
    ]
    for i, d in enumerate(flagged, 1):
        desc = d.get("description", "no description provided")
        lines.append(f"  {i}. ${float(d['amount']):,.2f} on {d.get('date', 'unknown date')} - {desc}")
    return "\n".join(lines)


@tool
def calculate_total_debt_obligations(debts: dict, proposed_payment: float) -> str:
    """Total monthly obligations including the proposed mortgage payment.

    Args:
        debts: Mapping of obligation name to monthly amount. Roll-up keys such as
            ``total_monthly_debt`` are detected and excluded.
        proposed_payment: Proposed monthly PITI.
    """
    items = debt_line_items(debts)
    current = monthly_debt_total(debts)
    breakdown = "\n".join(f"  - {k.replace('_', ' ').title()}: ${v:,.2f}" for k, v in items.items())
    return (
        f"Total Monthly Obligations: ${current + proposed_payment:,.2f}\n"
        f"Existing Debt: ${current:,.2f}\n{breakdown}\n"
        f"Proposed Housing Payment: ${proposed_payment:,.2f}"
    )


ALL_TOOLS = [
    calculate_dti_ratio,
    calculate_ltv_ratio,
    calculate_reserves,
    calculate_housing_expense_ratio,
    check_credit_score_policy,
    check_large_deposits,
    calculate_total_debt_obligations,
]
