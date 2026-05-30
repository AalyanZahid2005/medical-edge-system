import csv
import os
import requests

API_URL = "http://localhost:5000/analyze"

TESTS = [
    ("Final_Defense_Assets/01_chest_pneumonia_clear.jpg.jpeg", "chest_xray", "Pneumonia"),
    ("Final_Defense_Assets/02_skin_lesion_uncertain.jpg.jpg", "skin", "UNKNOWN"),
    ("Final_Defense_Assets/03_brain_emergency.jpg.jpg", "brain_mri", "Meningioma"),
    ("Final_Defense_Assets/04_ood_rejection_test.jpg.jpeg", "brain_mri", "OOD_OR_REJECT"),
    ("Final_Defense_Assets/05_bone_fracture_explain.jpg.png", "bone", "Fracture"),
    ("Final_Defense_Assets/07_ecg_current.jpg.png", "ecg", "UNKNOWN"),
]

rows = []

for path, modality, expected in TESTS:
    print("\n==============================")
    print("FILE:", path)
    print("MODALITY:", modality)

    if not os.path.exists(path):
        print("MISSING:", path)
        rows.append({
            "file": path,
            "modality": modality,
            "expected": expected,
            "http_status": "MISSING",
            "api_status": "",
            "predicted": "",
            "confidence": "",
            "routing_tier": "",
            "gpio": "",
            "decision": "",
            "mismatch": "",
            "detected_modality": "",
            "pass_note": "file missing"
        })
        continue

    with open(path, "rb") as f:
        files = {"file": (os.path.basename(path), f, "image/jpeg")}
        data = {"modality": modality, "patient_id": "PT-FINAL-ASSET-TEST"}
        r = requests.post(API_URL, files=files, data=data, timeout=240)

    js = r.json()
    pred = js.get("prediction", {}).get("class")
    conf = js.get("prediction", {}).get("mean_confidence")
    tier = js.get("routing_tier")
    gpio = js.get("gpio_alert_type")
    decision = js.get("routing_decision")
    mismatch = js.get("modality_mismatch")
    detected = js.get("detected_modality")

    print("HTTP:", r.status_code)
    print("STATUS:", js.get("status"))
    print("PRED:", pred)
    print("CONF:", conf)
    print("TIER:", tier)
    print("GPIO:", gpio)
    print("DECISION:", decision)
    print("MISMATCH:", mismatch)
    print("DETECTED:", detected)

    rows.append({
        "file": path,
        "modality": modality,
        "expected": expected,
        "http_status": r.status_code,
        "api_status": js.get("status"),
        "predicted": pred,
        "confidence": conf,
        "routing_tier": tier,
        "gpio": gpio,
        "decision": decision,
        "mismatch": mismatch,
        "detected_modality": detected,
        "pass_note": ""
    })

with open("FINAL_API_TEST_RESULTS.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved: FINAL_API_TEST_RESULTS.csv")
