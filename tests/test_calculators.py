"""Tests for the deterministic layer.

Several of these are regression tests for defects in the original coursework
implementation. Each names the defect it guards, because a test whose failure
message explains the bug is worth three that just go red.
"""

from __future__ import annotations

import pytest

from app.tools.calculators import (
    calculate_dti_ratio,
    calculate_ltv_ratio,
    calculate_reserves,
    check_credit_score_policy,
    check_large_deposits,
    debt_line_items,
    monthly_debt_total,
)
from app.tools.metrics import compute_metrics

# The fixture as it ships: line items *and* a roll-up in the same mapping.
SARAH_DEBTS = {
    "car_loan": 1200,
    "student_loan": 800,
    "credit_cards": 1800,
    "total_monthly_debt": 3800,
}


class TestDebtDoubleCounting:
    """Regression: ``sum(debts.values())`` counted the roll-up as a fourth debt.

    Effect in the original notebook: Sarah's back-end DTI was reported as 86.4%
    instead of 56.0%, pushing a file that should approve toward denial — while
    every cell still printed a success message.
    """

    def test_rollup_key_is_excluded(self):
        assert monthly_debt_total(SARAH_DEBTS) == 3800.0

    def test_naive_sum_is_what_we_are_guarding_against(self):
        assert sum(SARAH_DEBTS.values()) == 7600  # the bug, for the record
        assert monthly_debt_total(SARAH_DEBTS) != sum(SARAH_DEBTS.values())

    def test_line_items_exclude_rollup(self):
        assert set(debt_line_items(SARAH_DEBTS)) == {"car_loan", "student_loan", "credit_cards"}

    def test_payload_with_only_a_rollup_still_works(self):
        assert monthly_debt_total({"total_monthly_debt": 2500}) == 2500.0

    def test_empty_payload(self):
        assert monthly_debt_total({}) == 0.0

    @pytest.mark.parametrize("alias", ["total", "total_debt", "monthly_total"])
    def test_other_rollup_spellings(self, alias):
        assert monthly_debt_total({"car_loan": 500, alias: 500}) == 500.0


class TestBackEndDTI:
    """Back-end DTI includes the proposed housing payment (policy 2.6).

    The shipped fixtures record ``dti_ratio`` *excluding* it, which is a
    different ratio wearing the same name.
    """

    def test_includes_proposed_payment(self):
        metrics = compute_metrics(
            {
                "employment": {"monthly_income": 12500},
                "debts": SARAH_DEBTS,
                "loan": {"estimated_payment": 3200},
            }
        )
        assert metrics["existing_debt"] == 3800
        assert metrics["total_obligations"] == 7000
        assert metrics["dti_ratio"] == pytest.approx(56.0, abs=0.1)

    def test_fixture_value_is_the_non_housing_ratio(self):
        # 3800 / 12500 = 30.4%, which is what the JSON records.
        assert 3800 / 12500 * 100 == pytest.approx(30.4, abs=0.1)

    def test_zero_income_does_not_divide_by_zero(self):
        assert compute_metrics({"employment": {"monthly_income": 0}})["dti_ratio"] is None


class TestReserves:
    """Policy 3.1: reserves are liquid assets remaining *after* closing.

    Measuring gross liquidity instead overstates the cushion by the entire down
    payment -- Sarah reads as 58 months of reserves rather than 23.
    """

    def test_down_payment_and_closing_costs_are_deducted(self):
        m = compute_metrics(
            {
                "assets": {"checking": 85000, "savings": 100000},
                "loan": {"down_payment": 100000, "closing_costs": 12000, "estimated_payment": 3200},
            }
        )
        assert m["liquid_assets"] == 185000
        assert m["funds_to_close"] == 112000
        assert m["post_closing_liquidity"] == 73000
        assert m["reserve_months"] == pytest.approx(22.8, abs=0.1)

    def test_insufficient_funds_to_close_floors_at_zero(self):
        m = compute_metrics(
            {
                "assets": {"checking": 12000, "savings": 10000},
                "loan": {"down_payment": 35000, "closing_costs": 6000, "estimated_payment": 2400},
            }
        )
        assert m["post_closing_liquidity"] == 0
        assert m["reserve_months"] == 0


class TestLargeDeposits:
    """Regression: policy 3.3 says "$1,000 or 25% of income, whichever is less".

    The original used only the income share, so on Sarah's $12,500 income the
    threshold was $3,125 and every deposit between $1,000 and $3,125 was missed.
    """

    def test_threshold_is_the_lesser_of_the_two(self):
        out = check_large_deposits.invoke(
            {"deposits": [{"amount": 1500, "date": "2024-12-01"}], "monthly_income": 12500}
        )
        assert "1 deposit(s) require sourcing" in out
        assert "$1,000.00" in out

    def test_low_income_uses_the_income_share(self):
        out = check_large_deposits.invoke(
            {"deposits": [{"amount": 600, "date": "2024-12-01"}], "monthly_income": 2000}
        )
        assert "1 deposit(s) require sourcing" in out  # threshold is $500

    def test_nothing_flagged_when_all_small(self):
        out = check_large_deposits.invoke(
            {"deposits": [{"amount": 100, "date": "2024-12-01"}], "monthly_income": 12500}
        )
        assert "No deposits above" in out

    def test_empty_list(self):
        assert "No deposits above" in check_large_deposits.invoke(
            {"deposits": [], "monthly_income": 5000}
        )


class TestRatios:
    def test_dti_bands(self):
        assert "Acceptable" in calculate_dti_ratio.invoke(
            {"monthly_debt": 4000, "monthly_income": 10000}
        )
        assert "High" in calculate_dti_ratio.invoke(
            {"monthly_debt": 4500, "monthly_income": 10000}
        )
        assert "Excessive" in calculate_dti_ratio.invoke(
            {"monthly_debt": 6000, "monthly_income": 10000}
        )

    def test_ltv_uses_appraised_value(self):
        out = calculate_ltv_ratio.invoke({"loan_amount": 400000, "property_value": 515000})
        assert "77.67%" in out
        assert "Excellent" in out

    def test_ltv_guards_zero_value(self):
        assert "Error" in calculate_ltv_ratio.invoke({"loan_amount": 1, "property_value": 0})

    def test_reserves_shortfall_is_reported(self):
        out = calculate_reserves.invoke(
            {"liquid_assets": 1000, "monthly_payment": 3200, "required_months": 2}
        )
        assert "Insufficient" in out
        assert "-5,400.00" in out

    @pytest.mark.parametrize(
        "score,tier",
        [(760, "Excellent"), (710, "Very Good"), (670, "Good"), (630, "Fair"), (595, "Below Minimum")],
    )
    def test_credit_tiers(self, score, tier):
        assert tier in check_credit_score_policy.invoke({"credit_score": score})

    def test_below_minimum_cites_policy(self):
        assert "620" in check_credit_score_policy.invoke({"credit_score": 595})
