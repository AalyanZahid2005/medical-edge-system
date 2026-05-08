# src/pipeline/llm_reporter.py

import datetime

# ---------------------------------------------------------
# 1. Advanced Clinical Prompt Templates
# ---------------------------------------------------------
# These templates force the LLM to output structured medical reports 
# instead of casual summaries.

ADVANCED_PROMPTS = {
    "pneumonia": """Instruct: You are an expert AI Radiologist. Generate a formal, structured radiology report based on the following AI telemetry. Do not invent patient history.
--- AI TELEMETRY ---
Scan Type: Chest X-ray
Detected Anomaly: Pneumonia
AI Confidence: {confidence}%
Risk Tier: {risk_tier}
Primary Visual Evidence (SHAP): {shap_region}
--------------------
Format your response EXACTLY with these headings:
FINDINGS: (Detail the visual evidence and its implications)
IMPRESSION: (Concisely state the severity and primary diagnosis)
RECOMMENDATION: (Suggest immediate clinical next steps based on the risk tier)
Output: """,

    "mri": """Instruct: You are an expert AI Neurologist. Generate a formal, structured neurology report based on the following AI telemetry. Do not invent patient history.
--- AI TELEMETRY ---
Scan Type: Brain MRI
Detected Anomaly: Intracranial Anomaly / Tumor
AI Confidence: {confidence}%
Risk Tier: {risk_tier}
Primary Visual Evidence (SHAP): {shap_region}
--------------------
Format your response EXACTLY with these headings:
FINDINGS: (Detail the visual evidence and its implications)
IMPRESSION: (Concisely state the severity and primary diagnosis)
RECOMMENDATION: (Suggest immediate clinical next steps based on the risk tier)
Output: """,

    "retina": """Instruct: You are an expert AI Ophthalmologist. Generate a formal, structured ophthalmology report based on the following AI telemetry. Do not invent patient history.
--- AI TELEMETRY ---
Scan Type: Retinal Fundus
Detected Anomaly: Diabetic Retinopathy
AI Confidence: {confidence}%
Risk Tier: {risk_tier}
Primary Visual Evidence (SHAP): {shap_region}
--------------------
Format your response EXACTLY with these headings:
FINDINGS: (Detail the visual evidence and its implications)
IMPRESSION: (Concisely state the severity and primary diagnosis)
RECOMMENDATION: (Suggest immediate clinical next steps based on the risk tier)
Output: """,

    "skin": """Instruct: You are an expert AI Dermatologist. Generate a formal, structured dermatology report based on the following AI telemetry. Do not invent patient history.
--- AI TELEMETRY ---
Scan Type: Dermoscopy / Skin Lesion
Detected Anomaly: Malignant Melanoma Risk
AI Confidence: {confidence}%
Risk Tier: {risk_tier}
Primary Visual Evidence (SHAP): {shap_region}
--------------------
Format your response EXACTLY with these headings:
FINDINGS: (Detail the visual evidence and its implications)
IMPRESSION: (Concisely state the severity and primary diagnosis)
RECOMMENDATION: (Suggest immediate clinical next steps based on the risk tier)
Output: """,

    "bone": """Instruct: You are an expert AI Orthopedic Surgeon. Generate a formal, structured radiology report based on the following AI telemetry. Do not invent patient history.
--- AI TELEMETRY ---
Scan Type: Musculoskeletal X-ray
Detected Anomaly: Bone Fracture
AI Confidence: {confidence}%
Risk Tier: {risk_tier}
Primary Visual Evidence (SHAP): {shap_region}
--------------------
Format your response EXACTLY with these headings:
FINDINGS: (Detail the visual evidence and its implications)
IMPRESSION: (Concisely state the severity and primary diagnosis)
RECOMMENDATION: (Suggest immediate clinical next steps based on the risk tier)
Output: """,

    "ecg": """Instruct: You are an expert AI Cardiologist. Generate a formal, structured cardiology report based on the following AI telemetry. Do not invent patient history.
--- AI TELEMETRY ---
Scan Type: 12-Lead ECG Spectrogram
Detected Anomaly: Arrhythmia / Abnormal Rhythm
AI Confidence: {confidence}%
Risk Tier: {risk_tier}
Primary Visual Evidence (SHAP): {shap_region}
--------------------
Format your response EXACTLY with these headings:
FINDINGS: (Detail the visual evidence and its implications)
IMPRESSION: (Concisely state the severity and primary diagnosis)
RECOMMENDATION: (Suggest immediate clinical next steps based on the risk tier)
Output: """
}

# ---------------------------------------------------------
# 2. Reporter Engine
# ---------------------------------------------------------
def generate_llm_prompt(modality, confidence, risk_tier, shap_region):
    """
    Takes the raw numbers from the AI pipeline and injects them into the advanced LLM prompt.
    """
    modality = modality.lower()
    if modality not in ADVANCED_PROMPTS:
        return f"[ERROR] Unknown medical modality: {modality}"

    template = ADVANCED_PROMPTS[modality]
    
    # Inject the variables into the template string
    formatted_prompt = template.format(
        confidence=f"{confidence:.1f}", # Format to 1 decimal place
        risk_tier=risk_tier.upper(),    # Make the severity ALL CAPS for emphasis
        shap_region=shap_region
    )
    
    return formatted_prompt

# ==========================================
# TEST BED
# ==========================================
if __name__ == "__main__":
    print("============================================================")
    print("[SYSTEM] Testing Advanced Phi-2 Prompt Generation...")
    print("============================================================\n")

    # Simulating a highly critical patient case
    test_modality = "mri"
    test_confidence = 98.74
    test_tier = "Critical"
    test_shap = "High-intensity hyperdense mass in the right frontal lobe"

    final_prompt = generate_llm_prompt(
        modality=test_modality, 
        confidence=test_confidence, 
        risk_tier=test_tier, 
        shap_region=test_shap
    )

    print(final_prompt)