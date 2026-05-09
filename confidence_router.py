"""
confidence_router.py — FYP: Accelerated Medical Imaging using ML Architectures
Ahmed Rasheed (M3) — Day 7 Deliverable

The confidence router sits between mc_dropout_head.py output and the rest of
the pipeline. It takes raw inference results and decides:

    1. Routing Tier  (1 / 2 / 3)
    2. GPIO Alert    (GREEN / RED)
    3. Worklist flag (auto-queue for active learning)
    4. Action bundle (what downstream systems must do)

Tier definitions (from M1's math specs + gpio_alerts.py):
    Tier 1 — AUTO-APPROVE
        • Non-critical class AND uncertainty ≤ 2.5%
        • GPIO: GREEN LED
        • No human review needed
        • SR sent automatically

    Tier 2 — RADIOLOGIST REVIEW
        • Non-critical class AND 2.5% < uncertainty ≤ 5.0%
        • GPIO: GREEN LED (advisory only)
        • Added to soft worklist — radiologist reviews when available
        • SR sent with REVIEW flag

    Tier 3 — URGENT
        • Critical class (in CRITICAL_FINDINGS) OR uncertainty > 5.0%
        • GPIO: RED LED + Buzzer
        • Immediately added to active_learning_queue (PENDING)
        • SR sent with URGENT flag
        • Triggers physical alert on Jetson Nano GPIO

Sources this file is aligned to:
    db_manager.py        → CRITICAL_FINDINGS, insert_record(), queue_for_review()
    flask_api.py         → /analyze response contract (routing_tier, gpio_alert_type)
    gpio_alerts.py       → CRITICAL_FINDINGS list, uncertainty > 5.0 threshold
    mc_dropout_head.py   → mean_probs, std_probs, uncertainty_percent
    db_schema.sql        → inference_records.routing_tier CHECK(1,2,3)
                           inference_records.gpio_alert_type CHECK('GREEN','RED','NONE')
                           active_learning_queue.status DEFAULT 'PENDING'
    send_dicom_results.py → send_sr() called based on routing decision
    dicom_listener.py    → forwards to /analyze which calls route()

Usage:
    from confidence_router import ConfidenceRouter

    router = ConfidenceRouter()
    decision = router.route(
        predicted_class    = "Glioma",
        mean_confidence    = 92.5,
        uncertainty_percent= 1.2,
        modality           = "brain_mri",
        ood_flag           = False,
    )
    # decision.tier, decision.gpio_alert, decision.actions, decision.reason

Run self-test:
    python confidence_router.py --test
"""

import argparse
import sys
from dataclasses import dataclass, field
from typing import List

# ── Local project import ───────────────────────────────────────────────────────
try:
    from db_manager import CRITICAL_FINDINGS, DatabaseManager
except ImportError:
    print("[WARNING] db_manager.py not found — using embedded CRITICAL_FINDINGS.")
    DatabaseManager = None
    # Embedded fallback — exact copy from db_manager.py
    CRITICAL_FINDINGS = [
        "Glioma", "Meningioma", "Pituitary",
        "1", "2", "3", "4",
        "MEL", "BCC", "AKIEC", "BKL", "DF", "VASC",
        "Pneumonia",
        "Fracture",
        'L','R','B','A','a','J','S','V','r','F','e','j','n','E','/','f','Q','?'
    ]

# ══════════════════════════════════════════════════════════════════════════════
#  THRESHOLDS  (from M1's math specs + gpio_alerts.py)
# ══════════════════════════════════════════════════════════════════════════════

UNCERTAINTY_TIER3 = 5.0     # > 5.0%  → Tier 3 URGENT   (gpio_alerts.py trigger)
UNCERTAINTY_TIER2 = 2.5     # > 2.5%  → Tier 2 REVIEW
CONFIDENCE_MIN    = 50.0    # < 50.0% → force Tier 3 regardless of class
                            #           (too low confidence to auto-approve anything)
OOD_FORCE_TIER3  = True     # OOD flag always forces Tier 3 — result is unreliable

# ══════════════════════════════════════════════════════════════════════════════
#  ACTION CONSTANTS
#  These string constants are consumed by flask_api.py to decide what to call
# ══════════════════════════════════════════════════════════════════════════════

ACTION_SEND_SR          = "SEND_SR"          # send_dicom_results.send_sr()
ACTION_GPIO_GREEN       = "GPIO_GREEN"       # gpio_alerts.py → GREEN LED
ACTION_GPIO_RED         = "GPIO_RED"         # gpio_alerts.py → RED LED + Buzzer
ACTION_QUEUE_REVIEW     = "QUEUE_REVIEW"     # db_manager.queue_for_review()
ACTION_NOTIFY_URGENT    = "NOTIFY_URGENT"    # future: push notification to radiologist
ACTION_LOG_OOD          = "LOG_OOD"          # log OOD rejection in system_logs


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTING DECISION DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RoutingDecision:
    """
    Output of ConfidenceRouter.route().
    Consumed directly by flask_api.py /analyze to build the response
    and trigger downstream actions.

    Fields map to:
        flask_api.py response  → tier, gpio_alert, gpio_triggered
        db_manager.py          → tier, gpio_alert, gpio_triggered (insert_record)
        gpio_alerts.py         → gpio_alert ("RED" → physical alert fires)
        send_dicom_results.py  → ACTION_SEND_SR in actions list
        active_learning_queue  → ACTION_QUEUE_REVIEW in actions list
    """
    # Core routing output
    tier:           int             # 1, 2, or 3
    gpio_alert:     str             # "GREEN" or "RED"
    gpio_triggered: int             # 1 if RED, 0 if GREEN
    actions:        List[str]       # ordered list of ACTION_* constants
    reason:         str             # human-readable explanation (for logs + SR)

    # Echo back inputs for convenience (flask_api.py bundles these into response)
    predicted_class:     str   = ""
    mean_confidence:     float = 0.0
    uncertainty_percent: float = 0.0
    modality:            str   = ""
    ood_flag:            bool  = False

    # Derived flags (convenient for flask_api.py conditionals)
    is_critical:         bool  = False   # class in CRITICAL_FINDINGS
    is_high_uncertainty: bool  = False   # uncertainty > UNCERTAINTY_TIER3
    is_ood:              bool  = False   # ood_flag = True
    is_low_confidence:   bool  = False   # mean_confidence < CONFIDENCE_MIN
    requires_review:     bool  = False   # tier >= 2
    is_urgent:           bool  = False   # tier == 3

    def to_dict(self) -> dict:
        """
        Serialize to dict for flask_api.py JSON response and db_manager.insert_record().
        Keys match the flask_api.py /analyze response contract exactly.
        """
        return {
            "routing_tier":    self.tier,
            "gpio_alert_type": self.gpio_alert,
            "gpio_triggered":  self.gpio_triggered,
            "routing_reason":  self.reason,
            "actions":         self.actions,
            "is_urgent":       self.is_urgent,
            "requires_review": self.requires_review,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIDENCE ROUTER
# ══════════════════════════════════════════════════════════════════════════════

class ConfidenceRouter:
    """
    Central routing engine for the FYP medical imaging pipeline.

    Takes mc_dropout_head.py output and produces a RoutingDecision
    that tells flask_api.py exactly what to do next.

    Design principle: every path through this router must produce a
    deterministic, logged, DB-compatible decision. No silent failures.

    Thresholds (from M1's math specs):
        uncertainty > 5.0%  → Tier 3 (gpio_alerts.py RED trigger)
        uncertainty > 2.5%  → Tier 2 (soft review)
        class in CRITICAL_FINDINGS → Tier 3 always
        ood_flag = True     → Tier 3 always (result unreliable)
        confidence < 50%    → Tier 3 always (not enough signal)
    """

    def __init__(self,
                 uncertainty_tier3: float = UNCERTAINTY_TIER3,
                 uncertainty_tier2: float = UNCERTAINTY_TIER2,
                 confidence_min:    float = CONFIDENCE_MIN,
                 ood_force_tier3:   bool  = OOD_FORCE_TIER3):
        self.uncertainty_tier3 = uncertainty_tier3
        self.uncertainty_tier2 = uncertainty_tier2
        self.confidence_min    = confidence_min
        self.ood_force_tier3   = ood_force_tier3

    # ── Main routing method ────────────────────────────────────────────────────

    def route(self,
              predicted_class:     str,
              mean_confidence:     float,
              uncertainty_percent: float,
              modality:            str,
              ood_flag:            bool = False) -> RoutingDecision:
        """
        Compute the full routing decision for one inference result.

        Args:
            predicted_class     : e.g. "Glioma" — from mc_dropout_head.py top-1
            mean_confidence     : 0–100 scale — mc_dropout_head mean_probs[top] * 100
            uncertainty_percent : mc_dropout_head std_probs[top] * 100
            modality            : "brain_mri" | "retinal" | "skin" |
                                  "chest_xray" | "bone" | "ecg"
            ood_flag            : True if OOD autoencoder rejected this image (Day 8)

        Returns:
            RoutingDecision — consumed by flask_api.py /analyze
        """
        # ── Evaluate all trigger conditions ───────────────────────────────────
        is_critical        = predicted_class in CRITICAL_FINDINGS
        is_high_uncertainty= uncertainty_percent > self.uncertainty_tier3
        is_med_uncertainty = uncertainty_percent > self.uncertainty_tier2
        is_low_confidence  = mean_confidence < self.confidence_min
        is_ood             = ood_flag and self.ood_force_tier3

        # ── Determine tier + build reason string ──────────────────────────────
        tier, reason, gpio_alert, actions = self._decide(
            predicted_class     = predicted_class,
            is_critical         = is_critical,
            is_high_uncertainty = is_high_uncertainty,
            is_med_uncertainty  = is_med_uncertainty,
            is_low_confidence   = is_low_confidence,
            is_ood              = is_ood,
            uncertainty_percent = uncertainty_percent,
            mean_confidence     = mean_confidence,
            modality            = modality,
        )

        gpio_triggered = 1 if gpio_alert == "RED" else 0

        return RoutingDecision(
            tier                = tier,
            gpio_alert          = gpio_alert,
            gpio_triggered      = gpio_triggered,
            actions             = actions,
            reason              = reason,
            predicted_class     = predicted_class,
            mean_confidence     = mean_confidence,
            uncertainty_percent = uncertainty_percent,
            modality            = modality,
            ood_flag            = ood_flag,
            is_critical         = is_critical,
            is_high_uncertainty = is_high_uncertainty,
            is_ood              = is_ood,
            is_low_confidence   = is_low_confidence,
            requires_review     = tier >= 2,
            is_urgent           = tier == 3,
        )

    # ── Decision logic ─────────────────────────────────────────────────────────

    def _decide(self, predicted_class, is_critical, is_high_uncertainty,
                is_med_uncertainty, is_low_confidence, is_ood,
                uncertainty_percent, mean_confidence, modality):
        """
        Core decision tree. Returns (tier, reason, gpio_alert, actions).

        Priority order (highest wins):
            1. OOD flag           → always Tier 3
            2. Critical class     → always Tier 3
            3. Low confidence     → always Tier 3
            4. High uncertainty   → Tier 3
            5. Medium uncertainty → Tier 2
            6. Default            → Tier 1
        """

        # ── TIER 3 paths ───────────────────────────────────────────────────────

        if is_ood:
            return (
                3,
                f"OOD DETECTED on {modality.upper()} — image rejected as out-of-distribution. "
                f"Result unreliable. Manual review required.",
                "RED",
                [ACTION_GPIO_RED, ACTION_QUEUE_REVIEW, ACTION_SEND_SR, ACTION_LOG_OOD]
            )

        if is_critical and is_high_uncertainty:
            return (
                3,
                f"CRITICAL class '{predicted_class}' ({modality.upper()}) with HIGH uncertainty "
                f"{uncertainty_percent:.2f}% > {self.uncertainty_tier3}%. "
                f"Dual trigger: critical finding + uncertain prediction.",
                "RED",
                [ACTION_GPIO_RED, ACTION_QUEUE_REVIEW, ACTION_SEND_SR, ACTION_NOTIFY_URGENT]
            )

        if is_critical:
            return (
                3,
                f"CRITICAL class '{predicted_class}' detected in {modality.upper()} "
                f"(confidence={mean_confidence:.1f}%, uncertainty={uncertainty_percent:.2f}%). "
                f"Class is in CRITICAL_FINDINGS — immediate clinical attention required.",
                "RED",
                [ACTION_GPIO_RED, ACTION_QUEUE_REVIEW, ACTION_SEND_SR, ACTION_NOTIFY_URGENT]
            )

        if is_low_confidence:
            return (
                3,
                f"LOW CONFIDENCE: {mean_confidence:.1f}% < {self.confidence_min}% minimum. "
                f"Modality: {modality.upper()}. Predicted: '{predicted_class}'. "
                f"Insufficient signal — human review mandatory.",
                "RED",
                [ACTION_GPIO_RED, ACTION_QUEUE_REVIEW, ACTION_SEND_SR]
            )

        if is_high_uncertainty:
            return (
                3,
                f"HIGH UNCERTAINTY: {uncertainty_percent:.2f}% > {self.uncertainty_tier3}% "
                f"(mc_dropout_head.py threshold). "
                f"Class: '{predicted_class}' on {modality.upper()}. "
                f"MC Dropout variance too high for auto-approval.",
                "RED",
                [ACTION_GPIO_RED, ACTION_QUEUE_REVIEW, ACTION_SEND_SR]
            )

        # ── TIER 2 path ────────────────────────────────────────────────────────

        if is_med_uncertainty:
            return (
                2,
                f"MODERATE UNCERTAINTY: {uncertainty_percent:.2f}% "
                f"({self.uncertainty_tier2}%–{self.uncertainty_tier3}% range). "
                f"Class: '{predicted_class}' on {modality.upper()} "
                f"(confidence={mean_confidence:.1f}%). "
                f"Soft review — radiologist to confirm when available.",
                "GREEN",
                [ACTION_GPIO_GREEN, ACTION_SEND_SR]
            )

        # ── TIER 1 — auto-approve ──────────────────────────────────────────────

        return (
            1,
            f"AUTO-APPROVED: '{predicted_class}' on {modality.upper()} — "
            f"confidence={mean_confidence:.1f}%, uncertainty={uncertainty_percent:.2f}% "
            f"(below {self.uncertainty_tier2}% threshold). "
            f"Non-critical class. No human review needed.",
            "GREEN",
            [ACTION_GPIO_GREEN, ACTION_SEND_SR]
        )

    # ── Batch routing (for stress test Day 14) ─────────────────────────────────

    def route_batch(self, records: list) -> list:
        """
        Route a list of inference dicts. Used by Day 14 concurrent stress test.

        Args:
            records: list of dicts with keys:
                predicted_class, mean_confidence, uncertainty_percent,
                modality, ood_flag (optional)

        Returns:
            list of RoutingDecision objects in same order as input
        """
        return [
            self.route(
                predicted_class     = r.get("predicted_class", ""),
                mean_confidence     = r.get("mean_confidence", 0.0),
                uncertainty_percent = r.get("uncertainty_percent", 0.0),
                modality            = r.get("modality", "brain_mri"),
                ood_flag            = r.get("ood_flag", False),
            )
            for r in records
        ]

    # ── DB integration helper ──────────────────────────────────────────────────

    def apply_to_db_record(self, record: dict) -> dict:
        """
        Take a raw pipeline record dict and inject routing fields into it.
        Used by flask_api.py to enrich the record before db_manager.insert_record().

        Args:
            record: dict with at minimum:
                predicted_class, mean_confidence, uncertainty_percent,
                modality, ood_flag

        Returns:
            Same dict with routing_tier, gpio_alert_type, gpio_triggered added.
        """
        decision = self.route(
            predicted_class     = record.get("predicted_class", ""),
            mean_confidence     = record.get("mean_confidence", 0.0),
            uncertainty_percent = record.get("uncertainty_percent", 0.0),
            modality            = record.get("modality", "brain_mri"),
            ood_flag            = record.get("ood_flag", False),
        )
        record["routing_tier"]    = decision.tier
        record["gpio_alert_type"] = decision.gpio_alert
        record["gpio_triggered"]  = decision.gpio_triggered
        return record, decision


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
#  python confidence_router.py --test
# ══════════════════════════════════════════════════════════════════════════════

def run_self_test():
    """
    Tests every routing path with representative inputs from each modality.
    All assertions match gpio_alerts.py CRITICAL_FINDINGS + M1's thresholds.
    """
    print("\n" + "═"*65)
    print("  confidence_router.py — M3 Day 7 Self-Test")
    print("═"*65)

    router = ConfidenceRouter()
    passed = 0
    failed = 0

    test_cases = [
        # (description, inputs, expected_tier, expected_gpio)
        # ── Tier 3: Critical class ─────────────────────────────────────────
        ("Brain MRI — Glioma (critical class, low uncertainty)",
         dict(predicted_class="Glioma",     mean_confidence=92.5, uncertainty_percent=1.2,
              modality="brain_mri", ood_flag=False),
         3, "RED"),

        ("Brain MRI — Meningioma (critical class)",
         dict(predicted_class="Meningioma", mean_confidence=87.0, uncertainty_percent=2.0,
              modality="brain_mri", ood_flag=False),
         3, "RED"),

        ("Chest X-Ray — Pneumonia (critical class)",
         dict(predicted_class="Pneumonia",  mean_confidence=88.3, uncertainty_percent=3.1,
              modality="chest_xray", ood_flag=False),
         3, "RED"),

        ("Bone — Fracture (critical class)",
         dict(predicted_class="Fracture",   mean_confidence=91.0, uncertainty_percent=1.5,
              modality="bone", ood_flag=False),
         3, "RED"),

        ("Skin — MEL (critical class)",
         dict(predicted_class="MEL",        mean_confidence=79.0, uncertainty_percent=2.3,
              modality="skin", ood_flag=False),
         3, "RED"),

        ("ECG — Arrhythmia class 'V' (critical)",
         dict(predicted_class="V",          mean_confidence=85.0, uncertainty_percent=1.8,
              modality="ecg", ood_flag=False),
         3, "RED"),

        # ── Tier 3: High uncertainty ───────────────────────────────────────
        ("Brain MRI — Normal but HIGH uncertainty (>5.0%)",
         dict(predicted_class="Normal",     mean_confidence=60.0, uncertainty_percent=6.8,
              modality="brain_mri", ood_flag=False),
         3, "RED"),

        ("Retinal — No DR but uncertainty=5.1%",
         dict(predicted_class="No DR",      mean_confidence=72.0, uncertainty_percent=5.1,
              modality="retinal", ood_flag=False),
         3, "RED"),

        # ── Tier 3: Low confidence ─────────────────────────────────────────
        ("Chest X-Ray — Normal but confidence=45% (below 50%)",
         dict(predicted_class="Normal",     mean_confidence=45.0, uncertainty_percent=2.0,
              modality="chest_xray", ood_flag=False),
         3, "RED"),

        # ── Tier 3: OOD flag ───────────────────────────────────────────────
        ("Brain MRI — OOD flag (dog photo sent by mistake)",
         dict(predicted_class="Glioma",     mean_confidence=55.0, uncertainty_percent=1.0,
              modality="brain_mri", ood_flag=True),
         3, "RED"),

        # ── Tier 3: Critical + High uncertainty (dual trigger) ─────────────
        ("Brain MRI — Glioma + HIGH uncertainty (dual trigger)",
         dict(predicted_class="Glioma",     mean_confidence=62.0, uncertainty_percent=7.2,
              modality="brain_mri", ood_flag=False),
         3, "RED"),

        # ── Tier 2: Moderate uncertainty ──────────────────────────────────
        ("Brain MRI — Pituitary with moderate uncertainty (3.5%)",
         # Note: Pituitary IS in CRITICAL_FINDINGS → should be Tier 3
         # Use a non-critical class for Tier 2 test
         dict(predicted_class="Normal",     mean_confidence=80.0, uncertainty_percent=3.5,
              modality="brain_mri", ood_flag=False),
         2, "GREEN"),

        ("Retinal — No DR, uncertainty=4.9% (just below Tier 3)",
         dict(predicted_class="No DR",      mean_confidence=76.0, uncertainty_percent=4.9,
              modality="retinal", ood_flag=False),
         2, "GREEN"),

        ("ECG — Normal, uncertainty=2.6%",
         dict(predicted_class="Normal",     mean_confidence=88.0, uncertainty_percent=2.6,
              modality="ecg", ood_flag=False),
         2, "GREEN"),

        # ── Tier 1: Auto-approve ───────────────────────────────────────────
        ("Brain MRI — Normal, low uncertainty",
         dict(predicted_class="Normal",     mean_confidence=96.1, uncertainty_percent=0.8,
              modality="brain_mri", ood_flag=False),
         1, "GREEN"),

        ("ECG — Normal sinus rhythm",
         dict(predicted_class="Normal",     mean_confidence=97.3, uncertainty_percent=0.5,
              modality="ecg", ood_flag=False),
         1, "GREEN"),

        ("Skin — NV (naevus, non-critical)",
         dict(predicted_class="NV",         mean_confidence=93.0, uncertainty_percent=1.1,
              modality="skin", ood_flag=False),
         1, "GREEN"),

        ("Chest X-Ray — Normal, high confidence",
         dict(predicted_class="Normal",     mean_confidence=94.2, uncertainty_percent=2.0,
              modality="chest_xray", ood_flag=False),
         1, "GREEN"),
    ]

    for description, inputs, exp_tier, exp_gpio in test_cases:
        decision = router.route(**inputs)
        ok = (decision.tier == exp_tier and decision.gpio_alert == exp_gpio)

        status = "✓" if ok else "✗"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"\n  {status} {description}")
        print(f"     class={inputs['predicted_class']!r:12s} "
              f"conf={inputs['mean_confidence']:5.1f}% "
              f"unc={inputs['uncertainty_percent']:4.1f}% "
              f"ood={inputs['ood_flag']}")
        print(f"     tier={decision.tier} gpio={decision.gpio_alert:5s} "
              f"actions={decision.actions}")
        print(f"     reason: {decision.reason[:90]}...")

        if not ok:
            print(f"     ✗ EXPECTED tier={exp_tier} gpio={exp_gpio} "
                  f"GOT tier={decision.tier} gpio={decision.gpio_alert}")

    # ── Batch routing test ─────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Batch routing test (4 records):")
    batch = [
        dict(predicted_class="Glioma",  mean_confidence=92.5, uncertainty_percent=1.2,
             modality="brain_mri", ood_flag=False),
        dict(predicted_class="Normal",  mean_confidence=96.0, uncertainty_percent=0.5,
             modality="chest_xray", ood_flag=False),
        dict(predicted_class="Normal",  mean_confidence=70.0, uncertainty_percent=6.0,
             modality="ecg", ood_flag=False),
        dict(predicted_class="NV",      mean_confidence=88.0, uncertainty_percent=3.0,
             modality="skin", ood_flag=True),
    ]
    decisions = router.route_batch(batch)
    assert len(decisions) == 4
    assert decisions[0].tier == 3   # Glioma → critical
    assert decisions[1].tier == 1   # Normal → auto
    assert decisions[2].tier == 3   # high uncertainty
    assert decisions[3].tier == 3   # OOD
    print(f"  ✓ Batch: tiers={[d.tier for d in decisions]} "
          f"gpios={[d.gpio_alert for d in decisions]}")
    passed += 1

    # ── to_dict() contract test ────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  to_dict() contract test (flask_api.py compatibility):")
    d = router.route("Glioma", 92.5, 1.2, "brain_mri")
    dct = d.to_dict()
    required_keys = {"routing_tier", "gpio_alert_type", "gpio_triggered",
                     "routing_reason", "actions", "is_urgent", "requires_review"}
    assert required_keys.issubset(dct.keys()), f"Missing keys: {required_keys - dct.keys()}"
    assert dct["routing_tier"]    == 3
    assert dct["gpio_alert_type"] == "RED"
    assert dct["gpio_triggered"]  == 1
    assert dct["is_urgent"]       == True
    print(f"  ✓ to_dict() keys: {sorted(dct.keys())}")
    passed += 1

    print(f"\n{'═'*65}")
    print(f"  Results: {passed} passed, {failed} failed out of {passed+failed} tests")
    if failed == 0:
        print("  All tests passed. confidence_router.py is ready.")
        print("  Next: wire into flask_api.py /analyze (replace inline routing logic)")
    else:
        print("  ✗ Some tests failed — review threshold config above.")
    print("═"*65 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Confidence Router — FYP Medical Imaging Pipeline (M3 Day 7)"
    )
    parser.add_argument("--test", action="store_true",
                        help="Run self-test suite")
    parser.add_argument("--predict", type=str,
                        help="Quick route: predicted class e.g. --predict Glioma")
    parser.add_argument("--confidence", type=float, default=90.0)
    parser.add_argument("--uncertainty", type=float, default=1.5)
    parser.add_argument("--modality",   type=str,   default="brain_mri")
    parser.add_argument("--ood",        action="store_true")
    args = parser.parse_args()

    if args.test:
        run_self_test()

    elif args.predict:
        router   = ConfidenceRouter()
        decision = router.route(
            predicted_class     = args.predict,
            mean_confidence     = args.confidence,
            uncertainty_percent = args.uncertainty,
            modality            = args.modality,
            ood_flag            = args.ood,
        )
        print(f"\n{'═'*50}")
        print(f"  Input:  class={args.predict!r} conf={args.confidence}% "
              f"unc={args.uncertainty}% modality={args.modality} ood={args.ood}")
        print(f"  Tier:   {decision.tier}")
        print(f"  GPIO:   {decision.gpio_alert}  (triggered={decision.gpio_triggered})")
        print(f"  Actions:{decision.actions}")
        print(f"  Reason: {decision.reason}")
        print(f"{'═'*50}\n")

    else:
        parser.print_help()