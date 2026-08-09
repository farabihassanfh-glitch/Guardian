"""Build a full application record from a small set of user-supplied fields.

The demo lets a visitor invent an applicant. That means accepting input from
the public internet, so the surface is deliberately narrow: ten bounded numeric
fields and one enum, never free-form JSON. Everything the agents need beyond
that — credit notes, deposit history, property condition — is derived here from
the numbers, so a visitor cannot inject prose into an agent prompt.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

#: Rough national-average assumptions, used only to derive a monthly payment
#: when the visitor doesn't supply one. Stated in the UI so the number isn't
#: mistaken for a quote.
ANNUAL_RATE = 0.065
TERM_MONTHS = 360
#: Property tax + hazard insurance as a share of value per year.
ESCROW_RATE = 0.015


class CustomApplication(BaseModel):
    """The fields a visitor may set. Every one is bounded."""

    credit_score: int = Field(700, ge=300, le=850)
    monthly_income: float = Field(9000, ge=500, le=200_000)
    monthly_debts: float = Field(1200, ge=0, le=100_000)
    loan_amount: float = Field(350_000, ge=10_000, le=5_000_000)
    down_payment: float = Field(70_000, ge=0, le=5_000_000)
    property_value: float = Field(420_000, ge=10_000, le=10_000_000)
    savings: float = Field(90_000, ge=0, le=10_000_000)
    employment_type: Literal["W2", "Self-Employed"] = "W2"
    years_employed: float = Field(5, ge=0, le=60)
    late_payments_12mo: int = Field(0, ge=0, le=20)
    required_repairs: float = Field(0, ge=0, le=500_000)


def estimate_monthly_payment(loan_amount: float, property_value: float) -> float:
    """Principal, interest, taxes and insurance at the assumed rate."""
    r = ANNUAL_RATE / 12
    principal_interest = loan_amount * r / (1 - (1 + r) ** -TERM_MONTHS) if loan_amount else 0.0
    escrow = property_value * ESCROW_RATE / 12
    return round(principal_interest + escrow, 2)


def build_case(app: CustomApplication, case_id: str) -> dict:
    """Expand the visitor's inputs into the record shape the graph expects."""
    piti = estimate_monthly_payment(app.loan_amount, app.property_value)
    purchase_price = app.loan_amount + app.down_payment
    self_employed = app.employment_type == "Self-Employed"

    return {
        "case_id": case_id,
        # Placeholder identity: the redaction layer replaces these anyway, and
        # a visitor is never asked for a real person's details.
        "name": "Custom Applicant",
        "ssn": "000-00-0000",
        "email": "applicant@example.com",
        "phone": "555-000-0000",
        "address": "Not provided",
        "credit_score": app.credit_score,
        "credit_history": {
            "bankruptcies": 0,
            "foreclosures": 0,
            "late_payments_12mo": app.late_payments_12mo,
            "late_payments_24mo": app.late_payments_12mo,
            "collections": [],
            "inquiries_6mo": 2,
            "credit_notes": (
                f"{app.late_payments_12mo} late payment(s) in the last 12 months. "
                "No bankruptcies, foreclosures or collections on file."
            ),
        },
        "employment": {
            "employer": "Not provided",
            "position": "Not provided",
            "years": app.years_employed,
            "monthly_income": app.monthly_income,
            "type": app.employment_type,
            "employment_gap": "None reported",
            "gap_explanation": "N/A",
            "income_details": {
                "base_salary": round(app.monthly_income * 12, 2),
                "bonus_stable": not self_employed,
                "employer_confirmation": (
                    "Self-employed; income verified via 2 years of tax returns."
                    if self_employed
                    else "W2 income verified via paystubs and W2s."
                ),
            },
        },
        "debts": {
            "monthly_obligations": app.monthly_debts,
            "total_monthly_debt": app.monthly_debts,
        },
        "assets": {
            "checking": 0,
            "savings": app.savings,
            "liquid_assets_total": app.savings,
            "401k": 0,
            "recent_deposits": [],
            "deposit_explanations": "No large deposits reported.",
            "reserves_months": (
                round(max(0.0, app.savings - app.down_payment) / piti, 1) if piti else 0
            ),
        },
        "loan": {
            "amount": app.loan_amount,
            "down_payment": app.down_payment,
            "closing_costs": round(purchase_price * 0.03, 2),
            "estimated_payment": piti,
            "monthly_piti": piti,
            "property_type": "Single Family",
            "use": "Primary Residence",
        },
        "property": {
            "purchase_price": purchase_price,
            "appraised_value": app.property_value,
            "condition": "C4 - Fair" if app.required_repairs > 0 else "C3 - Average",
            "type": "Single Family Home",
            "required_repairs": app.required_repairs,
            "repair_details": (
                f"${app.required_repairs:,.0f} of repairs identified at inspection."
                if app.required_repairs
                else "No repairs required."
            ),
        },
        # No expected outcome for an invented applicant — there is nothing to
        # compare against, and the UI hides the match badge accordingly.
        "expected_decision": None,
    }
