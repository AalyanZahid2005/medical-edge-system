# src/pipeline/clinical_advisor.py

class ClinicalAdvisor:
    def __init__(self):
        self.confidence_threshold = 75.0 # Strict safety threshold as per FYP docs
        
        # Clinical Knowledge Base (Evidence-Based Guidelines)
        self.knowledge_base = {
            "glioma": {
                "precautions": "Strict fall precautions due to potential motor deficits. Monitor for signs of increased intracranial pressure (severe headache, vomiting).",
                "nutrition_eat": "Ketogenic diet protocols may be considered as an adjunct therapy under clinical supervision (NIH).",
                "nutrition_avoid": "Avoid excessive sodium which may exacerbate cerebral edema.",
                "source": "NIH National Cancer Institute Brain Tumor Guidelines"
            },
            "pneumonia": {
                "precautions": "Maintain isolation if infectious etiology suspected. Monitor SpO2 levels closely. Deep breathing exercises.",
                "nutrition_eat": "High fluid intake (2.5-3L/day) to help clear secretions. Vitamin C and Zinc-rich foods.",
                "nutrition_avoid": "Dairy products if they thicken mucus (patient-dependent). Avoid alcohol and respiratory irritants.",
                "source": "WHO Clinical Management of Severe Acute Respiratory Infection"
            },
            "diabetic retinopathy grade 4": {
                "precautions": "Avoid heavy lifting or strenuous exercise which can trigger vitreous hemorrhage. Immediate ophthalmology consult.",
                "nutrition_eat": "Strict low-glycemic index foods. Leafy greens rich in lutein (NIH).",
                "nutrition_avoid": "Simple carbohydrates, refined sugars, and high-sodium foods to control hypertension.",
                "source": "WHO Global Report on Diabetes & Eye Care"
            },
            "melanoma": {
                "precautions": "Strict UV avoidance. Do not attempt to lance or apply over-the-counter topical treatments to the lesion.",
                "nutrition_eat": "Antioxidant-rich Mediterranean diet to support immune function.",
                "nutrition_avoid": "Inflammatory foods, processed meats.",
                "source": "NIH Skin Cancer Treatment Guidelines"
            }
        }

    def get_care_package(self, diagnosis, confidence):
        """
        Generates the Patient Care Package string.
        Enforces a strict safety threshold: Only gives advice if AI is highly confident.
        """
        clean_diag = diagnosis.lower().strip()
        
        # Safety Gatekeeper
        if confidence < self.confidence_threshold:
            return "\n\n*** CLINICAL ADVISORY ***\nAI confidence is below the 75% safety threshold. Diagnostic confirmation by a human physician is strictly required before issuing a Patient Care Package."
            
        advice = self.knowledge_base.get(clean_diag)
        
        if not advice:
            return "" # Silently return nothing if we don't have guidelines for this specific disease
            
        package = (
            f"\n\n*** PATIENT CARE PACKAGE ***\n"
            f"[Source: {advice['source']}]\n"
            f"• Precautions: {advice['precautions']}\n"
            f"• Recommended Nutrition: {advice['nutrition_eat']}\n"
            f"• Contraindicated Nutrition: {advice['nutrition_avoid']}"
        )
        return package

# Quick standalone test
if __name__ == "__main__":
    advisor = ClinicalAdvisor()
    print("[SYSTEM] Testing Clinical Advisor (>75% Confidence)...")
    print(advisor.get_care_package("Pneumonia", 88.0))
    
    print("\n[SYSTEM] Testing Safety Threshold (<75% Confidence)...")
    print(advisor.get_care_package("Glioma", 60.0))
