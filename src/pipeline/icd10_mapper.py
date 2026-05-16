# src/pipeline/icd10_mapper.py

class ICD10Mapper:
    def __init__(self):
        # Official ICD-10-CM Codes for Billing
        self.mapping = {
            "glioma": {"code": "C71.9", "description": "Malignant neoplasm of brain, unspecified"},
            "brain tumor": {"code": "D43.2", "description": "Neoplasm of uncertain behavior of brain, unspecified"},
            "pneumonia": {"code": "J18.9", "description": "Pneumonia, unspecified organism"},
            "melanoma": {"code": "C43.9", "description": "Malignant melanoma of skin, unspecified"},
            "diabetic retinopathy": {"code": "E11.319", "description": "Type 2 diabetes mellitus with unspecified diabetic retinopathy without macular edema"},
            "diabetic retinopathy grade 4": {"code": "E11.3592", "description": "Type 2 diabetes mellitus with proliferative diabetic retinopathy"},
            "arrhythmia": {"code": "I49.9", "description": "Cardiac arrhythmia, unspecified"},
            "bone fracture": {"code": "M84.40XA", "description": "Pathological fracture, unspecified site, initial encounter"},
            "fracture": {"code": "T14.8", "description": "Other injury of unspecified body region"}
        }

    def get_code(self, diagnosis, confidence=100.0):
        """
        Returns the ICD-10 billing code. 
        As per M2's future task, we flag low-confidence predictions for human review.
        """
        clean_diag = diagnosis.lower().strip()
        
        # Default to R69 (Illness, unspecified) if the AI predicts something unknown
        result = self.mapping.get(clean_diag, {"code": "R69", "description": "Illness, unspecified"})

        # Add a flag if confidence is too low to auto-bill
        result["needs_human_review"] = confidence < 80.0

        return result

# Quick standalone test
if __name__ == "__main__":
    mapper = ICD10Mapper()
    print("[SYSTEM] Testing ICD-10 Mapper...")
    print(mapper.get_code("Glioma", 98.0))
    print(mapper.get_code("Pneumonia", 75.0))
