"""Staging -> Production promotion gate: pure decision logic, no MLflow/K8s I/O.

Gate (thresholds from config/promotion_gate.yaml):
    candidate.f2_score  >= prod_baseline.f2_score  + min_f2_delta
    candidate.precision >= prod_baseline.precision - max_precision_drop
Both must hold. If `prod_baseline` is None (no model has ever been promoted), there is
nothing to compare against, so the automatic gate cannot evaluate — this does NOT mean
auto-approve. The very first promotion is never fully automatic: it requires
`bootstrap_confirmed=True` (wired to a `confirm_bootstrap` workflow_dispatch input /
manual approval, never a default), because a first promotion is exactly the case where
there's no automatic signal to catch a bad candidate. Every promotion after that goes
through the ordinary F2/precision comparison with no manual step.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromotionDecision:
    approved: bool
    reasons: list[str]


def evaluate_promotion_gate(
    candidate: dict[str, float],
    prod_baseline: dict[str, float] | None,
    gate_config: dict[str, float],
    bootstrap_confirmed: bool = False,
) -> PromotionDecision:
    """`bootstrap_confirmed` only has any effect when `prod_baseline` is None (the
    first-ever promotion); it's ignored for every later promotion, which always goes
    through the automatic F2/precision comparison regardless of its value.
    """
    if prod_baseline is None:
        if bootstrap_confirmed:
            return PromotionDecision(
                approved=True,
                reasons=[
                    "No existing production baseline (first promotion) - manually "
                    "confirmed via bootstrap_confirmed=True, gate approves."
                ],
            )
        return PromotionDecision(
            approved=False,
            reasons=[
                "No existing production baseline (first promotion) - the automatic "
                "F2/precision gate has nothing to compare against and cannot approve "
                "this on its own. Re-run with bootstrap_confirmed=True (confirm_bootstrap "
                "workflow input) after a human has reviewed the candidate's scores."
            ],
        )

    min_f2_delta = gate_config["min_f2_delta"]
    max_precision_drop = gate_config["max_precision_drop"]

    required_f2 = prod_baseline["f2_score"] + min_f2_delta
    required_precision = prod_baseline["precision"] - max_precision_drop

    f2_ok = candidate["f2_score"] >= required_f2
    precision_ok = candidate["precision"] >= required_precision

    reasons = [
        f"F2: candidate={candidate['f2_score']:.4f} "
        f"{'>=' if f2_ok else '<'} required={required_f2:.4f} "
        f"(prod={prod_baseline['f2_score']:.4f} + min_f2_delta={min_f2_delta:.4f}) "
        f"-> {'PASS' if f2_ok else 'FAIL'}",
        f"Precision: candidate={candidate['precision']:.4f} "
        f"{'>=' if precision_ok else '<'} required={required_precision:.4f} "
        f"(prod={prod_baseline['precision']:.4f} - max_precision_drop={max_precision_drop:.4f}) "
        f"-> {'PASS' if precision_ok else 'FAIL'}",
    ]
    return PromotionDecision(approved=f2_ok and precision_ok, reasons=reasons)
