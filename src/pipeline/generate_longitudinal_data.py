import json
import os
from datetime import datetime, timedelta

def generate_patient_histories():
    output_dir = r"D:\medical-edge-system\Final_Defense_Assets\longitudinal_patients"
    os.makedirs(output_dir, exist_ok=True)

    base_date = datetime(2026, 4, 1) # Starting 30 days ago

    patients = [
        {
            "id": "PT-001", "name": "Patient A (Rapid Decline)", "trend": [15.2, 45.8, 78.4, 92.1]
        },
        {
            "id": "PT-002", "name": "Patient B (Steady Improvement)", "trend": [88.5, 65.2, 42.1, 18.4]
        },
        {
            "id": "PT-003", "name": "Patient C (Stable/Healthy)", "trend": [12.1, 11.5, 13.2, 12.8]
        },
        {
            "id": "PT-004", "name": "Patient D (Fluctuating)", "trend": [45.2, 22.1, 68.9, 35.4]
        },
        {
            "id": "PT-005", "name": "Patient E (Sudden Emergency)", "trend": [14.2, 15.1, 13.9, 95.8]
        }
    ]

    print(f"--- Generating 5 Longitudinal Profiles ---")
    
    for pt in patients:
        history = []
        for i, score in enumerate(pt["trend"]):
            visit_date = base_date + timedelta(days=(i * 10)) # Visits spaced by 10 days
            
            # Simple tier logic
            tier = "High" if score > 70 else "Moderate" if score > 30 else "Low"
            
            history.append({
                "visit_date": visit_date.strftime("%Y-%m-%d"),
                "model_confidence_abnormal": score,
                "clinical_notes": f"Automated scan evaluation. Risk classified as {tier}.",
                "risk_tier": tier
            })

        profile = {
            "patient_id": pt["id"],
            "name": pt["name"],
            "age": 62,
            "gender": "Female" if "B" in pt["name"] else "Male",
            "primary_modality": "Chest X-ray",
            "longitudinal_history": history
        }

        file_path = os.path.join(output_dir, f"{pt['id']}_history.json")
        with open(file_path, 'w') as f:
            json.dump(profile, f, indent=4)
        
        print(f"[OK] Created {pt['id']}_history.json ({pt['name']})")

if __name__ == "__main__":
    generate_patient_histories()