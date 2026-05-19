import time
import subprocess
import datetime
import re
try:
    from pipeline.clinical_advisor import ClinicalAdvisor
except ModuleNotFoundError:
    from clinical_advisor import ClinicalAdvisor
# ---------------------------------------------------------
# CONFIGURATION — paths to real Phi-2 binary and model
# ---------------------------------------------------------
LLAMA_BINARY = "/home/jetson-nano/llama.cpp/main"
MODEL_PATH    = "/home/jetson-nano/llama.cpp/models/phi-2.Q4_0.gguf"

def kill_loops(text):
    """Cut output the moment a word sequence starts repeating."""
    words = text.split()
    for i in range(2, len(words) - 2):
        if words[i] == words[i-2] and words[i+1] == words[i-1]:
            return " ".join(words[:i]).strip()
    return text

# ---------------------------------------------------------
# URDU TEMPLATE ENGINE (Pure Nastaliq)
# ---------------------------------------------------------
URDU_MODALITY = {
    "mri":     "دماغ کے ایم آر آئی اسکین",
    "chest":   "سینے کے ایکس رے",
    "retinal": "ریٹینل فنڈس اسکین",
    "ecg":     "ای سی جی سگنل",
    "skin":    "جلد کے معائنے",
    "bone":    "ہڈی کے ایکس رے",
}

URDU_RISK = {
    "critical": "انتہائی خطرناک",
    "high":     "شدید",
    "medium":   "متوسط",
    "low":      "کم خطر",
    "tier 1":   "کم خطر",
    "tier 2":   "شدید",
    "tier 3":   "انتہائی خطرناک",
}

URDU_DIAGNOSIS = {
    "glioma": "گلیوما",
    "brain tumor": "دماغی رسولی",
    "pneumonia": "نمونیا",
    "diabetic retinopathy": "ذیابیطس ریٹینوپیتھی",
    "arrhythmia": "دل کی بے ترتیبی",
    "melanoma": "میلانوما (جلد کا کینسر)",
    "bone fracture": "ہڈی کے فریکچر",
    "fracture": "فریکچر"
}

def generate_urdu_template(modality, diagnosis, confidence, risk_tier):
    """
    Template-based Roman Urdu report — consistent and reliable.
    No LLM call needed for Urdu path.
    """
    mod_urdu  = URDU_MODALITY.get(modality.lower(), "طبی اسکین")
    risk_urdu = URDU_RISK.get(risk_tier.lower(), "شدید")
    diag_urdu = URDU_DIAGNOSIS.get(diagnosis.lower(), diagnosis) # Translates the diagnosis!

    report = (
        f"مریض کے {mod_urdu} میں {diag_urdu} کی تشخیص ہوئی ہے، "
        f"جس پر AI کا یقین {confidence:.1f} فیصد ہے۔ "
        f"مریض کی حالت {risk_urdu} ہے، "
        f"اس لیے فوری ماہر ڈاکٹر سے رجوع کرنا چاہیے۔"
    )
    return report
# ---------------------------------------------------------
# 2. Prompt Templates — Prefix Injection Format (English Only)
# ---------------------------------------------------------
PROMPTS = {
    "english": {
        "mri": (
            "### System: You are a clinical medical AI assistant. Be precise, professional, and concise. "
            "### User: Brain MRI scan analyzed. "
            "Diagnosis: {diagnosis}. "
            "AI Confidence: {confidence:.1f}%. Risk Tier: {risk_tier}. "
            "Write a 2-sentence clinical summary for the neurosurgeon about the PATIENT. "
            "### Assistant: The patient presents with"
        ),
        "chest": (
            "### System: You are a clinical medical AI assistant. Be precise and professional. "
            "### User: Chest X-ray analyzed. "
            "Diagnosis: {diagnosis}. "
            "AI Confidence: {confidence:.1f}%. "
            "Write a 2-sentence clinical summary for the physician about the PATIENT. "
            "### Assistant: The patient presents with"
        ),
    }
}

def get_prompt(modality, language, **kwargs):
    lang = language.lower()
    if lang != "english": 
        return ""
        
    mod  = modality.lower()
    lang_prompts = PROMPTS.get(lang, PROMPTS["english"])
    template = lang_prompts.get(mod, lang_prompts.get("mri", ""))

    if not template:
        template = "### Assistant: The patient presents with"

    try:
        return template.format(**kwargs)
    except KeyError:
        return template

# ---------------------------------------------------------
# 3. Real Phi-2 Subprocess Call
# ---------------------------------------------------------
def call_phi2(prompt, max_tokens=120, temperature=0.1, timeout=180):
    try:
        result = subprocess.run(
            [
                LLAMA_BINARY,
                "-m", MODEL_PATH,
                "-c", "256",
                "-n", str(max_tokens),
                "--temp", str(temperature),
                "--log-disable",
                "-p", prompt,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )

        output = result.stdout.decode("utf-8", errors="replace")
        
        marker = "### Assistant:"
        if marker in output:
            parts = output.split(marker, 1)
            report = parts[1].strip() if len(parts) > 1 else output.strip()
            
            if "The patient presents with" in prompt:
                report = "The patient presents with " + report
        else:
            report = output.strip()

        # Regex duplicate opener cleanup
        report = re.sub(r'^(The patient presents with\s+){2,}', 'The patient presents with ', report)

        # Expanded stop token cleanup
        for token in ["<|endoftext|>", "Instruct:", "\n\nInstruct", "\n\nOutput", "\n\n\n", "\nAI:", "\nNote:", "\nThe AI"]:
            if token in report:
                report = report.split(token)[0].strip()

        return report if report else "Report generation failed — empty output."

    except subprocess.TimeoutExpired:
        return "Report generation timed out. Phi-2 exceeded 180 seconds."
    except Exception as e:
        return f"Report generation error: {str(e)}"

# ---------------------------------------------------------
# 4. Main Reporter Class
# ---------------------------------------------------------
class ClinicalLLMReporter:
    def __init__(self):
        print("[SYSTEM] Initializing Phi-2 Clinical Reporter...")
        print(f"[SYSTEM] Binary : {LLAMA_BINARY}")
        print(f"[SYSTEM] Model  : {MODEL_PATH}")
        self.advisor = ClinicalAdvisor() # <--- ADD THIS LINE
    def generate_report(self, modality="mri", diagnosis="Brain Tumor", confidence=94.2, uncertainty=1.8, risk_tier="Critical", region_a=34.0, region_b=12.0, language="english"):
        print(f"\n[*] Generating report | Modality: {modality.upper()} | Language: {language.upper()} | Confidence: {confidence:.1f}%")

        start = time.time()
        
        if language.lower() == "urdu":
            print("[POST] Generating Pure Nastaliq Template...")
            final_report = generate_urdu_template(modality, diagnosis, confidence, risk_tier)
        else:
            prompt = get_prompt(
                modality=modality, language=language, diagnosis=diagnosis, 
                confidence=confidence, uncertainty=uncertainty, risk_tier=risk_tier, 
                region_a=region_a, region_b=region_b
            )
            print("[LLM] Calling real Phi-2 binary — please wait (~60-90 seconds)...")
            raw_text = call_phi2(prompt)
            raw_text = kill_loops(raw_text)
            final_report = raw_text

        elapsed = round(time.time() - start, 1)
       
        # --- NEW: Append the Patient Care Package ---
        care_package = self.advisor.get_care_package(diagnosis, confidence)
        if care_package:
            final_report += "\n" + care_package

        print("\n" + "=" * 52)
        print("  CLINICAL REPORT — " + language.upper())
        print("=" * 52)
        print(final_report.strip())
        print("=" * 52)
        print(f"[METRICS] Real generation time: {elapsed}s | Power: ~5W")
        print(f"[METRICS] Timestamp         : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return final_report

if __name__ == "__main__":
    reporter = ClinicalLLMReporter()
    
    reporter.generate_report(
        modality="mri", diagnosis="Glioma", confidence=98.0, 
        risk_tier="Critical", language="english"
    )

    reporter.generate_report(
        modality="mri", diagnosis="Glioma", confidence=98.0, 
        risk_tier="Critical", language="urdu"
    )

    reporter.generate_report(
        modality="chest", diagnosis="Pneumonia", confidence=87.3, 
        risk_tier="High", language="english"
    )
