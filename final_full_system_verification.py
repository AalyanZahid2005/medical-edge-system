import csv
import json
import os
import time
import requests
from datetime import datetime

BASE_URL = "http://localhost:5000"

TESTS = [
    {
        "name": "chest_pneumonia_clear",
        "file": "Final_Defense_Assets/01_chest_pneumonia_clear.jpg.jpeg",
        "modality": "chest_xray",
        "expected_behavior": "accepted_or_routed"
    },
    {
        "name": "skin_lesion_uncertain",
        "file": "Final_Defense_Assets/02_skin_lesion_uncertain.jpg.jpg",
        "modality": "skin",
        "expected_behavior": "accepted_or_routed"
    },
    {
        "name": "brain_emergency_meningioma",
        "file": "Final_Defense_Assets/03_brain_emergency.jpg.jpg",
        "modality": "brain_mri",
        "expected_behavior": "accepted_or_routed"
    },
    {
        "name": "ood_rejection_test",
        "file": "Final_Defense_Assets/04_ood_rejection_test.jpg.jpeg",
        "modality": "brain_mri",
        "expected_behavior": "ood_or_tier3"
    },
    {
        "name": "bone_fracture_explain",
        "file": "Final_Defense_Assets/05_bone_fracture_explain.jpg.png",
        "modality": "bone",
        "expected_behavior": "accepted_or_routed"
    },
    {
        "name": "ecg_current",
        "file": "Final_Defense_Assets/07_ecg_current.jpg.png",
        "modality": "ecg",
        "expected_behavior": "accepted_or_routed"
    },
]

MISMATCH_TEST = {
    "name": "modality_mismatch_guard",
    "file": "Final_Defense_Assets/02_skin_lesion_uncertain.jpg.jpg",
    "modality": "brain_mri",
    "expected_behavior": "modality_mismatch"
}


def get_json(path, timeout=30):
    url = BASE_URL + path
    r = requests.get(url, timeout=timeout)
    return r.status_code, r.json()


def analyze(test, patient_id):
    path = test["file"]
    if not os.path.exists(path):
        return {
            "test_name": test["name"],
            "file": path,
            "modality": test["modality"],
            "http_status": "MISSING_FILE",
            "api_status": "missing_file",
            "passed": False,
            "reason": "Input file does not exist."
        }

    with open(path, "rb") as f:
        files = {
            "file": (os.path.basename(path), f, "application/octet-stream")
        }
        data = {
            "modality": test["modality"],
            "patient_id": patient_id
        }
        r = requests.post(BASE_URL + "/analyze", files=files, data=data, timeout=300)

    try:
        js = r.json()
    except Exception:
        js = {"raw_text": r.text}

    pred = js.get("prediction", {})
    if not isinstance(pred, dict):
        pred = {}

    record_id = js.get("record_id", -1)
    gradcam = js.get("gradcam_base64")

    expected = test["expected_behavior"]

    passed = True
    reasons = []

    if r.status_code != 200:
        passed = False
        reasons.append("HTTP status is not 200.")

    if js.get("status") not in ["ok", "rejected"]:
        passed = False
        reasons.append("API status is not ok/rejected.")

    if expected == "modality_mismatch":
        if not (js.get("modality_mismatch") is True or pred.get("class") == "MODALITY_MISMATCH"):
            passed = False
            reasons.append("Mismatch test did not return modality_mismatch.")
        if js.get("routing_tier") != 3:
            passed = False
            reasons.append("Mismatch test did not route to Tier 3.")
    elif expected == "ood_or_tier3":
        if not (js.get("ood_flag") is True or js.get("routing_tier") == 3):
            passed = False
            reasons.append("OOD test was not rejected or routed to Tier 3.")
    else:
        if record_id in [-1, None, ""]:
            passed = False
            reasons.append("No real DB record_id returned.")
        if js.get("routing_tier") not in [1, 2, 3]:
            passed = False
            reasons.append("Routing tier missing/invalid.")

    return {
        "test_name": test["name"],
        "file": path,
        "selected_modality": test["modality"],
        "http_status": r.status_code,
        "api_status": js.get("status"),
        "record_id": record_id,
        "prediction": pred.get("class"),
        "confidence": pred.get("mean_confidence"),
        "uncertainty": pred.get("uncertainty"),
        "routing_tier": js.get("routing_tier"),
        "routing_decision": js.get("routing_decision"),
        "risk_tier": js.get("risk_tier"),
        "manual_review_required": js.get("manual_review_required"),
        "auto_report_allowed": js.get("auto_report_allowed"),
        "ood_flag": js.get("ood_flag"),
        "modality_mismatch": js.get("modality_mismatch"),
        "detected_modality": js.get("detected_modality"),
        "gpio_alert_type": js.get("gpio_alert_type"),
        "gpio_triggered": js.get("gpio_triggered"),
        "pipeline_ms": js.get("pipeline_ms"),
        "gradcam_present": bool(gradcam),
        "gradcam_length": len(gradcam) if gradcam else 0,
        "llm_report_present": bool(js.get("llm_report")),
        "patient_id": js.get("patient_id", patient_id),
        "passed": passed,
        "reason": "; ".join(reasons) if reasons else "PASS"
    }


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    patient_prefix = "PT-FINAL-VERIFY-" + run_id

    evidence = {
        "run_id": run_id,
        "base_url": BASE_URL,
        "created_at": datetime.now().isoformat(),
        "health": {},
        "stats_before": {},
        "queue_before": {},
        "tests": [],
        "history_after": {},
        "stats_after": {},
        "queue_after": {},
        "summary": {}
    }

    print("\n=== HEALTH ===")
    code, health = get_json("/health")
    evidence["health"] = {"http_status": code, "body": health}
    print(json.dumps(evidence["health"], indent=2))

    print("\n=== STATS BEFORE ===")
    code, stats = get_json("/stats")
    evidence["stats_before"] = {"http_status": code, "body": stats}
    print(json.dumps(evidence["stats_before"], indent=2))

    print("\n=== QUEUE BEFORE ===")
    code, queue = get_json("/queue")
    evidence["queue_before"] = {"http_status": code, "body": queue}
    print(json.dumps(evidence["queue_before"], indent=2))

    all_tests = TESTS + [MISMATCH_TEST]

    for i, test in enumerate(all_tests, start=1):
        print("\n==============================")
        print("RUNNING:", test["name"])
        result = analyze(test, patient_id=f"{patient_prefix}-{i:02d}")
        evidence["tests"].append(result)
        print(json.dumps(result, indent=2))
        time.sleep(3)

    print("\n=== HISTORY AFTER ===")
    code, history = get_json("/history?limit=15")
    evidence["history_after"] = {"http_status": code, "body": history}
    print(json.dumps(evidence["history_after"], indent=2)[:3000])

    print("\n=== STATS AFTER ===")
    code, stats_after = get_json("/stats")
    evidence["stats_after"] = {"http_status": code, "body": stats_after}
    print(json.dumps(evidence["stats_after"], indent=2))

    print("\n=== QUEUE AFTER ===")
    code, queue_after = get_json("/queue")
    evidence["queue_after"] = {"http_status": code, "body": queue_after}
    print(json.dumps(evidence["queue_after"], indent=2))

    passed_count = sum(1 for t in evidence["tests"] if t["passed"])
    failed = [t for t in evidence["tests"] if not t["passed"]]

    gradcam_count = sum(1 for t in evidence["tests"] if t.get("gradcam_present"))
    real_record_ids = [
        t.get("record_id") for t in evidence["tests"]
        if isinstance(t.get("record_id"), int) and t.get("record_id") > 0
    ]

    evidence["summary"] = {
        "total_tests": len(evidence["tests"]),
        "passed": passed_count,
        "failed": len(failed),
        "gradcam_present_count": gradcam_count,
        "real_record_ids_count": len(real_record_ids),
        "real_record_ids": real_record_ids,
        "overall_pass": len(failed) == 0
    }

    json_path = "FINAL_FULL_SYSTEM_VERIFICATION.json"
    csv_path = "FINAL_FULL_SYSTEM_VERIFICATION.csv"

    with open(json_path, "w") as f:
        json.dump(evidence, f, indent=2)

    fieldnames = [
        "test_name", "file", "selected_modality", "http_status", "api_status",
        "record_id", "prediction", "confidence", "uncertainty",
        "routing_tier", "routing_decision", "risk_tier",
        "manual_review_required", "auto_report_allowed",
        "ood_flag", "modality_mismatch", "detected_modality",
        "gpio_alert_type", "gpio_triggered", "pipeline_ms",
        "gradcam_present", "gradcam_length", "llm_report_present",
        "patient_id", "passed", "reason"
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evidence["tests"])

    print("\n==============================")
    print("FINAL SUMMARY")
    print(json.dumps(evidence["summary"], indent=2))
    print("Saved:", json_path)
    print("Saved:", csv_path)

    if evidence["summary"]["overall_pass"]:
        print("OVERALL RESULT: PASS")
    else:
        print("OVERALL RESULT: CHECK FAILED TESTS")


if __name__ == "__main__":
    main()
