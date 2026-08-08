"""Tests for the PII and Fair Lending guardrails."""

from __future__ import annotations

from app.compliance import detect_bias_signals, policy_gate, sanitize_pii

APPLICANT = {
    "case_id": "MTG-2025-001",
    "name": "Sarah Johnson",
    "ssn": "123-45-6789",
    "email": "sarah.johnson@email.com",
    "phone": "555-234-5678",
    "address": "1234 Oak Street, San Francisco, CA 94102",
    "credit_score": 760,
    "employment": {"employer": "Tech Solutions Inc", "monthly_income": 12500},
    "debts": {"car_loan": 1200, "total_monthly_debt": 1200},
}


class TestSanitisation:
    def test_direct_identifiers_are_removed(self):
        out = sanitize_pii(APPLICANT)
        assert out["name"] == "[APPLICANT_NAME]"
        assert out["address"] == "[ADDRESS]"
        assert out["ssn"] == "***-**-6789"
        assert out["phone"] == "***-***-5678"

    def test_email_is_redacted(self):
        """Regression: the original blocklist forgot ``email`` — and the fixture's
        address is ``sarah.johnson@email.com``, i.e. the name it had just redacted."""
        out = sanitize_pii(APPLICANT)
        assert out["email"] == "[EMAIL]"
        assert "sarah.johnson" not in str(out).lower()

    def test_no_identifier_survives_anywhere_in_the_payload(self):
        blob = str(sanitize_pii(APPLICANT)).lower()
        for leaked in ("sarah", "johnson", "123-45-6789", "oak street", "94102"):
            assert leaked not in blob, f"{leaked!r} leaked into the sanitised payload"

    def test_underwriting_data_is_preserved(self):
        out = sanitize_pii(APPLICANT)
        assert out["credit_score"] == 760
        assert out["employment"]["monthly_income"] == 12500

    def test_unknown_fields_are_withheld_by_default(self):
        """An allowlist fails closed: a new upstream field is not forwarded until
        someone consciously permits it."""
        out = sanitize_pii({**APPLICANT, "spouse_name": "Alex", "date_of_birth": "1985-03-02"})
        assert "spouse_name" not in out
        assert "date_of_birth" not in out

    def test_source_record_is_not_mutated(self):
        """Regression: a shallow ``.copy()`` leaves nested dicts aliased."""
        original = APPLICANT["employment"]["employer"]
        out = sanitize_pii(APPLICANT)
        out["employment"]["employer"] = "MUTATED"
        assert APPLICANT["employment"]["employer"] == original


class TestBiasDetection:
    def test_clean_analysis_raises_nothing(self):
        """The decisive case. A guardrail that fires on every input carries no
        information — and the original did, because ``"age" in "mortgage"``."""
        clean = (
            "The borrower's mortgage payment history is excellent. Average balances "
            "are low and coverage of reserves is adequate. Percentage of income "
            "committed to debt service is well within policy."
        )
        assert detect_bias_signals(clean, APPLICANT) == []

    def test_substring_matching_would_have_failed_here(self):
        text = "the mortgage is well covered on average"
        assert "age" in text  # substring: true
        assert detect_bias_signals(text, APPLICANT) == []  # word boundary: silent

    def test_genuine_violation_is_caught(self):
        flags = detect_bias_signals(
            "The applicant's marital status suggests instability in future income.", APPLICANT
        )
        assert any("marital status" in f for f in flags)

    def test_protected_term_standing_alone_is_caught(self):
        flags = detect_bias_signals("Given the applicant's age, tenure is a concern.", APPLICANT)
        assert any("'age'" in f for f in flags)

    def test_property_age_is_not_a_violation(self):
        assert detect_bias_signals("The age of the property is 40 years.", APPLICANT) == []

    def test_redlining_language_is_flagged(self):
        flags = detect_bias_signals("Values in this neighborhood are declining.", APPLICANT)
        assert any("redlining" in f for f in flags)


class TestPolicyGate:
    def test_low_score_is_a_hard_violation(self):
        v = policy_gate({"credit_score": 595, "credit_history": {}}, {})
        assert any("620" in x for x in v)

    def test_recent_bankruptcy_is_a_hard_violation(self):
        v = policy_gate(
            {
                "credit_score": 700,
                "credit_history": {"bankruptcies": 1, "bankruptcy_years_since_discharge": 0.5},
            },
            {},
        )
        assert any("seasoning" in x for x in v)

    def test_excessive_repairs_are_flagged(self):
        v = policy_gate(
            {"credit_score": 700, "credit_history": {}, "property": {"required_repairs": 8500}}, {}
        )
        assert any("escrow holdback" in x for x in v)

    def test_clean_file_passes(self):
        assert policy_gate(
            {"credit_score": 760, "credit_history": {"late_payments_12mo": 0}},
            {"dti_ratio": 30.4},
        ) == []
