class MedicalCodingMapper:
    def __init__(self):
        # 1. Master Mapping Dictionary for SNOMED-CT & ICD-10
        self.code_database = {
            "Brain MRI": {
                "Glioma": {"snomed": "126952004", "icd10": "C71.9", "alt_icd10": ["C71.8", "D43.2"]},
                "Meningioma": {"snomed": "19453003", "icd10": "D32.0", "alt_icd10": ["D32.9", "C70.0"]}
            },
            "Chest X-ray": {
                "Pneumonia": {"snomed": "233604007", "icd10": "J18.9", "alt_icd10": ["J15.9", "J12.9"]},
                "Normal": {"snomed": "165082007", "icd10": "Z00.00", "alt_icd10": ["Z01.89", "Z02.89"]}
            },
            "Skin Lesion": {
                "Melanoma": {"snomed": "372130007", "icd10": "C43.9", "alt_icd10": ["D03.9", "C44.9"]},
                "Benign Nevus": {"snomed": "400096001", "icd10": "D22.9", "alt_icd10": ["D23.9", "L82"]}
            },
            "Retinal Fundus": {
                "Diabetic Retinopathy Grade 4": {"snomed": "59620006", "icd10": "E11.3592", "alt_icd10": ["E11.319", "E11.329"]}
            },
            "ECG Signal": {
                "Myocardial Infarction": {"snomed": "22298006", "icd10": "I21.9", "alt_icd10": ["I21.3", "I22.9"]}
            },
            "Bone Fracture": {
                "Fracture": {"snomed": "125605004", "icd10": "M84.4", "alt_icd10": ["S82.90", "S52.90"]}
            }
        }

    def generate_billing_payload(self, modality, finding, confidence):
        """
        Retrieves the exact SNOMED and ICD-10 codes. 
        If confidence < 80%, returns 3 possible ICD-10 codes for human review.
        """
        print(f"\n[INFO] Processing EMR Routing for: {modality} -> {finding} (Confidence: {confidence}%)")
        
        try:
            codes = self.code_database[modality][finding]
        except KeyError:
            return {"error": "Diagnosis not mapped in current database."}

        snomed_code = codes["snomed"]
        
        # Implement the <80% Confidence Safety Fallback
        if confidence >= 80.0:
            icd10_output = [codes["icd10"]]
            status = "CONFIDENT_MATCH"
        else:
            icd10_output = [codes["icd10"]] + codes["alt_icd10"]
            status = "LOW_CONFIDENCE_REVIEW_REQUIRED"

        return {
            "snomed_ct_code": snomed_code,
            "icd10_codes": icd10_output,
            "billing_status": status
        }

# --- Integration Test ---
if __name__ == "__main__":
    print("--- Phase 5: EMR & Medical Billing Integration ---")
    mapper = MedicalCodingMapper()
    
    # Test 1: High Confidence (Should return 1 ICD-10 code)
    high_conf = mapper.generate_billing_payload("Chest X-ray", "Pneumonia", 92.5)
    print(f"High Confidence Payload: {high_conf}")
    
    # Test 2: Low Confidence (Should return 3 ICD-10 codes for review)
    low_conf = mapper.generate_billing_payload("Brain MRI", "Glioma", 74.2)
    print(f"Low Confidence Payload: {low_conf}")