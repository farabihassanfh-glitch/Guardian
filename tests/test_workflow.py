"""End-to-end workflow tests with a stubbed model.

These exercise the real graph — real routing, real merge semantics, real policy
gate, real PII redaction — while substituting the LLM and the retriever. That
keeps the suite free and fast, and it isolates orchestration bugs from prompt
bugs. When something misbehaves you want to know which of the two it was.
"""

from __future__ import annotations

import json

import pytest

from app.agents.decision import UnderwritingDecision
from app.config import get_settings
from app.graph import build_graph, initial_state


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """Returns canned prose, or a validated object when structured output is asked for."""

    def __init__(self, decision: UnderwritingDecision):
        self._decision = decision
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return FakeResponse("Stubbed analysis. Risk rating: Low. Ratios are within policy.")

    def with_structured_output(self, schema):
        outer = self

        class Structured:
            def invoke(self, messages):
                outer.calls += 1
                return outer._decision

        return Structured()


@pytest.fixture
def cases():
    return {
        c["case_id"]: c
        for c in json.loads(get_settings().test_cases.read_text(encoding="utf-8"))["test_cases"]
    }


@pytest.fixture
def stub(monkeypatch):
    """Patch the model and the retriever at their point of use."""

    def _install(decision: UnderwritingDecision) -> FakeLLM:
        fake = FakeLLM(decision)
        for module in ("app.agents.base", "app.agents.critic", "app.agents.decision"):
            monkeypatch.setattr(f"{module}.get_llm", lambda: fake)
        monkeypatch.setattr(
            "app.agents.base.retrieve_policies", lambda q, k=None: "STUBBED POLICY TEXT"
        )
        return fake

    return _install


CLEAN = UnderwritingDecision(
    risk_score=18, decision="APPROVED", conditions=[], credit_memo="Approved on the merits."
)
MODEL_SAYS_APPROVE = UnderwritingDecision(
    risk_score=20, decision="APPROVED", conditions=[], credit_memo="Looks fine to me."
)


def run(case, stub_fn, decision):
    fake = stub_fn(decision)
    graph = build_graph(mode="parallel")
    config = {"configurable": {"thread_id": case["case_id"]}}
    final = graph.invoke(initial_state(case), config)
    return final, fake


class TestEndToEnd:
    def test_strong_file_completes_and_approves(self, cases, stub):
        final, fake = run(cases["MTG-2025-001"], stub, CLEAN)

        assert final["final_decision"] == "APPROVED"
        assert final["risk_score"] == 18
        # 4 specialists + critic + decision
        assert fake.calls == 6

    def test_every_specialist_wrote_its_memo(self, cases, stub):
        final, _ = run(cases["MTG-2025-001"], stub, CLEAN)
        for key in ("credit_analysis", "income_analysis", "asset_analysis", "collateral_analysis"):
            assert final.get(key), f"{key} was never written"
        assert final.get("critic_review")
        assert final.get("decision_memo")

    def test_audit_trail_accumulates_across_parallel_agents(self, cases, stub):
        """The reducer test that matters: four concurrent writers, nothing lost."""
        final, _ = run(cases["MTG-2025-001"], stub, CLEAN)
        trail = final["reasoning_chain"]
        assert len(trail) == 7  # init + 4 specialists + critic + decision
        assert len(set(trail)) == len(trail), "duplicated entries — reducer double-applied"

    def test_pii_never_reaches_the_agents(self, cases, stub):
        final, _ = run(cases["MTG-2025-001"], stub, CLEAN)
        blob = str(final["sanitized_data"]).lower()
        for leaked in ("sarah", "johnson", "123-45-6789", "oak street"):
            assert leaked not in blob

    def test_metrics_are_computed_before_any_model_call(self, cases, stub):
        final, _ = run(cases["MTG-2025-001"], stub, CLEAN)
        m = final["metrics"]
        # 650 + 350 + 0, not 2000 -- the roll-up key is excluded.
        assert m["existing_debt"] == 1000
        assert m["dti_ratio"] == pytest.approx(33.6, abs=0.1)
        assert m["housing_ratio"] == pytest.approx(25.6, abs=0.1)
        # Policy 4.3 values collateral at the lesser of price ($500k) and
        # appraisal ($515k), so 400/500 = 80%, not 400/515 = 77.67%.
        assert m["collateral_basis"] == 500000
        assert m["ltv_ratio"] == pytest.approx(80.0, abs=0.1)

    def test_stated_ratios_agree_with_computed_ones(self, cases, stub):
        """The fixtures claim a DTI; the engine computes one. If they ever
        disagree, every downstream conclusion is measured against the wrong
        yardstick -- which is exactly how the original went wrong."""
        for case in cases.values():
            final, _ = run(case, stub, CLEAN)
            assert final["metrics"]["dti_ratio"] == pytest.approx(
                case["dti_ratio"] * 100, abs=0.5
            ), case["case_id"]


class TestPolicyGateOverride:
    """A bright-line violation is not negotiable, whatever the model concluded."""

    def test_weak_file_is_denied_even_when_the_model_approves(self, cases, stub):
        final, _ = run(cases["MTG-2025-003"], stub, MODEL_SAYS_APPROVE)

        assert final["policy_violations"], "expected hard violations on the weak file"
        assert final["final_decision"] == "DENIED"
        assert final["risk_score"] >= 71

    def test_denial_escalates_to_a_human(self, cases, stub):
        final, _ = run(cases["MTG-2025-003"], stub, MODEL_SAYS_APPROVE)
        assert final["human_review_required"] is True
        assert any("adverse action" in r.lower() for r in final["human_review_reasons"])

    def test_clean_file_does_not_escalate(self, cases, stub):
        """The counter-case. If everything escalates, escalation means nothing —
        which is exactly what the original bias detector caused."""
        final, _ = run(cases["MTG-2025-001"], stub, CLEAN)
        assert final["policy_violations"] == []
        assert final["bias_flags"] == []
        assert final["human_review_required"] is False


class TestTopologyEquivalence:
    def test_sequential_reaches_the_same_decision(self, cases, stub):
        fake = stub(CLEAN)
        graph = build_graph(mode="sequential")
        final = graph.invoke(
            initial_state(cases["MTG-2025-001"]), {"configurable": {"thread_id": "seq"}}
        )
        assert final["final_decision"] == "APPROVED"
        assert fake.calls == 6
        assert len(final["reasoning_chain"]) == 7
