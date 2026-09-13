from __future__ import annotations

import pytest

from pdm.evaluation.promotion import evaluate_promotion_gate

pytestmark = pytest.mark.unit

GATE_CONFIG = {"min_f2_delta": 0.02, "max_precision_drop": 0.03}


def test_bootstrap_without_confirmation_is_rejected():
    candidate = {"f2_score": 0.9, "precision": 0.9}  # even a great candidate...
    decision = evaluate_promotion_gate(candidate, None, GATE_CONFIG)
    assert decision.approved is False  # ...cannot auto-approve the first promotion
    assert "bootstrap_confirmed" in decision.reasons[0]


def test_bootstrap_with_confirmation_is_approved():
    candidate = {"f2_score": 0.0, "precision": 0.0}  # score is irrelevant once confirmed
    decision = evaluate_promotion_gate(candidate, None, GATE_CONFIG, bootstrap_confirmed=True)
    assert decision.approved is True
    assert "manually" in decision.reasons[0].lower()


def test_bootstrap_confirmed_flag_is_ignored_once_a_baseline_exists():
    # bootstrap_confirmed must have zero effect once there IS a prod baseline: a bad
    # candidate can't sneak past the real gate by setting it.
    prod = {"f2_score": 0.55, "precision": 0.60}
    candidate = {"f2_score": 0.0, "precision": 0.0}
    decision = evaluate_promotion_gate(candidate, prod, GATE_CONFIG, bootstrap_confirmed=True)
    assert decision.approved is False


def test_candidate_meeting_both_thresholds_is_approved():
    prod = {"f2_score": 0.30, "precision": 0.40}
    candidate = {"f2_score": 0.32, "precision": 0.37}  # exactly at both boundaries
    decision = evaluate_promotion_gate(candidate, prod, GATE_CONFIG)
    assert decision.approved is True


def test_candidate_failing_f2_is_rejected():
    prod = {"f2_score": 0.30, "precision": 0.40}
    candidate = {"f2_score": 0.31, "precision": 0.50}  # f2 short by 0.01
    decision = evaluate_promotion_gate(candidate, prod, GATE_CONFIG)
    assert decision.approved is False
    assert any("FAIL" in r and "F2" in r for r in decision.reasons)


def test_candidate_failing_precision_is_rejected_even_if_f2_passes():
    prod = {"f2_score": 0.30, "precision": 0.40}
    candidate = {"f2_score": 0.50, "precision": 0.30}  # precision short by 0.07
    decision = evaluate_promotion_gate(candidate, prod, GATE_CONFIG)
    assert decision.approved is False
    assert any("PASS" in r and "F2" in r for r in decision.reasons)
    assert any("FAIL" in r and "Precision" in r for r in decision.reasons)


def test_candidate_failing_both_is_rejected_with_both_reasons_listed():
    prod = {"f2_score": 0.55, "precision": 0.60}
    candidate = {"f2_score": 0.0, "precision": 0.0}
    decision = evaluate_promotion_gate(candidate, prod, GATE_CONFIG)
    assert decision.approved is False
    assert len(decision.reasons) == 2
    assert all("FAIL" in r for r in decision.reasons)


def test_boundary_is_inclusive():
    # candidate f2 exactly equal to prod + min_f2_delta must PASS (>=, not >).
    prod = {"f2_score": 0.50, "precision": 0.50}
    candidate = {"f2_score": 0.52, "precision": 0.47}
    decision = evaluate_promotion_gate(candidate, prod, GATE_CONFIG)
    assert decision.approved is True
