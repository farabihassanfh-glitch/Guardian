"""Routing and state-merge tests. No model calls, so these run in milliseconds."""

from __future__ import annotations

import json
import operator
from typing import Annotated, get_args, get_origin, get_type_hints

import pytest

from app.config import get_settings
from app.graph import route_from_supervisor, supervisor_node
from app.state import UnderwritingState


class TestSupervisorRouting:
    """The supervisor routes on *state* — 'which memo is blank?' — not a counter.
    That is what makes a half-finished file resumable."""

    def test_empty_file_goes_to_credit(self):
        assert supervisor_node({})["next_agent"] == "credit"

    def test_skips_completed_desks(self):
        state = {"credit_analysis": "done", "income_analysis": "done"}
        assert supervisor_node(state)["next_agent"] == "asset"

    def test_all_complete_routes_to_critic(self):
        state = {f"{n}_analysis": "done" for n in ("credit", "income", "asset", "collateral")}
        out = supervisor_node(state)
        assert out["next_agent"] == "critic"
        assert out["analysis_complete"] is True

    def test_out_of_order_completion_is_handled(self):
        """A desk finishing early must not confuse the dispatcher."""
        state = {"collateral_analysis": "done"}
        assert supervisor_node(state)["next_agent"] == "credit"

    def test_router_reads_the_note(self):
        assert route_from_supervisor({"next_agent": "asset"}) == "asset"
        assert route_from_supervisor({"analysis_complete": True}) == "critic"
        assert route_from_supervisor({}) == "credit"


class TestStateReducers:
    """Accumulating fields must use ``operator.add``.

    Without a reducer these channels overwrite, so in the parallel topology three
    of four concurrent specialist writes would be silently lost.
    """

    @pytest.mark.parametrize(
        "field", ["bias_flags", "policy_violations", "reasoning_chain"]
    )
    def test_audit_fields_accumulate(self, field):
        hint = get_type_hints(UnderwritingState, include_extras=True)[field]
        assert get_origin(hint) is Annotated, f"{field} needs a reducer annotation"
        assert operator.add in get_args(hint), f"{field} must accumulate, not overwrite"

    @pytest.mark.parametrize("field", ["final_decision", "risk_score", "next_agent"])
    def test_current_value_fields_overwrite(self, field):
        hint = get_type_hints(UnderwritingState, include_extras=True)[field]
        assert get_origin(hint) is not Annotated, f"{field} should overwrite, not accumulate"


class TestFixtures:
    """The fixtures are the ground truth everything else is measured against, so
    they get checked too."""

    @pytest.fixture(scope="class")
    @classmethod
    def cases(cls):
        return json.loads(get_settings().test_cases.read_text(encoding="utf-8"))["test_cases"]

    def test_three_tiers_present(self, cases):
        assert {c["expected_decision"] for c in cases} == {"APPROVED", "CONDITIONAL", "REJECTED"}

    def test_every_case_has_required_sections(self, cases):
        for c in cases:
            for key in ("credit_score", "employment", "debts", "assets", "loan", "property"):
                assert key in c, f"{c['case_id']} is missing {key}"

    def test_rollup_matches_line_items(self, cases):
        """Guards the data, not the code: if a fixture's roll-up ever disagrees
        with its line items, every downstream number is quietly wrong."""
        from app.tools.calculators import monthly_debt_total

        for c in cases:
            stated = c["debts"].get("total_monthly_debt")
            if stated is not None:
                assert monthly_debt_total(c["debts"]) == stated, c["case_id"]
