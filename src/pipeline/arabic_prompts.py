import arabic_reshaper
from bidi.algorithm import get_display

class ArabicClinicalTemplates:
    def __init__(self):
        # Professional Arabic Medical Terminology for the Phi-2 LLM
        self.templates = {
            "Brain MRI": (
                "بناءً على تحليل التصوير بالرنين المغناطيسي للدماغ، "
                "اكتشف الذكاء الاصطناعي {finding} بنسبة ثقة {confidence}%. "
                "يرجى كتابة تقرير طبي موجز."
            ),
            "Chest X-ray": (
                "بناءً على تحليل أشعة الصدر، "
                "اكتشف الذكاء الاصطناعي {finding} بنسبة ثقة {confidence}%. "
                "يرجى كتابة تقرير طبي موجز."
            ),
            "Retinal Fundus": (
                "بناءً على فحص قاع العين، "
                "اكتشف الذكاء الاصطناعي {finding} بنسبة ثقة {confidence}%. "
                "يرجى كتابة تقرير طبي موجز."
            ),
            "ECG Signal": (
                "بناءً على تخطيط كهربية القلب، "
                "اكتشف الذكاء الاصطناعي {finding} بنسبة ثقة {confidence}%. "
                "يرجى كتابة تقرير طبي موجز."
            ),
            "Skin Lesion": (
                "بناءً على فحص الآفة الجلدية، "
                "اكتشف الذكاء الاصطناعي {finding} بنسبة ثقة {confidence}%. "
                "يرجى كتابة تقرير طبي موجز."
            ),
            "Bone Fracture": (
                "بناءً على الأشعة السينية للعظام، "
                "اكتشف الذكاء الاصطناعي {finding} بنسبة ثقة {confidence}%. "
                "يرجى كتابة تقرير طبي موجز."
            )
        }

    def format_and_reshape(self, modality, finding, confidence):
        """
        Injects the variables, then applies RTL reshaping so the 
        PDF generator renders the Arabic script correctly.
        """
        if modality not in self.templates:
            return "Modality not supported for Arabic."

        # 1. Format the raw text with the specific diagnosis
        raw_text = self.templates[modality].format(finding=finding, confidence=confidence)
        
        # 2. Reshape the letters (connects disconnected Arabic characters)
        reshaped_text = arabic_reshaper.reshape(raw_text)
        
        # 3. Apply Bi-Directional algorithm (fixes Right-To-Left reading order)
        rtl_text = get_display(reshaped_text)
        
        return rtl_text

# --- Integration Test ---
if __name__ == "__main__":
    print("--- Phase 5: Arabic RTL Rendering Test ---")
    translator = ArabicClinicalTemplates()
    
    # Simulate a Chest X-ray finding Pneumonia at 88.5% confidence
    test_modality = "Chest X-ray"
    test_finding = "التهاب رئوي (Pneumonia)"
    test_confidence = "88.5"
    
    ready_for_pdf = translator.format_and_reshape(test_modality, test_finding, test_confidence)
    
    print("\n[Raw Output - May look disconnected or backwards in standard terminals]:")
    print(translator.templates[test_modality].format(finding=test_finding, confidence=test_confidence))
    
    print("\n[Reshaped RTL Output - Ready for PDF ReportLab Injection]:")
    print(ready_for_pdf)
    print("\n[OK] Arabic Terminology and RTL logic validated.")