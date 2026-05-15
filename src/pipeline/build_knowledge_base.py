import json
import os

def generate_knowledge_base():
    # Evidence-based Clinical Guidelines (WHO/NIH Cited)
    kb = {
        "Chest X-ray": {
            "Pneumonia": {
                "Precautionary Measures": "Maintain strict hand hygiene. Elevate the head of the bed to 30-45 degrees to prevent pulmonary aspiration.",
                "Foods to Eat": "High-protein foods (eggs, lean poultry) for tissue repair. Zinc-rich foods to support immune function.",
                "Foods to Avoid": "Highly processed foods and excessive refined sugars which can trigger inflammation.",
                "Natural Adjuncts": "Adequate oral hydration (2.5-3 Liters/day) to thin respiratory secretions.",
                "Citations": ["WHO: Clinical Management of Severe Acute Respiratory Infection (2022)", "NIH: Nutrition and Immune Function (2021)"]
            }
        },
        "ECG Signal": {
            "Myocardial Infarction": {
                "Precautionary Measures": "Strict bed rest during acute phase. Avoid any isometric exercises or heavy lifting. Monitor for dyspnea.",
                "Foods to Eat": "Mediterranean diet profile: rich in leafy greens, whole grains, and omega-3 fatty acids (salmon, walnuts).",
                "Foods to Avoid": "High-sodium foods (>1500mg/day), trans fats, and processed red meats.",
                "Natural Adjuncts": "Supervised cardiac rehabilitation program post-discharge.",
                "Citations": ["AHA/ACC Clinical Practice Guidelines for STEMI (2021)", "NIH: DASH Diet Protocol"]
            }
        },
        "Retinal Fundus": {
            "Diabetic Retinopathy Grade 4": {
                "Precautionary Measures": "Strict glycemic control (HbA1c < 7.0%). Immediate referral to a retinal specialist for potential photocoagulation.",
                "Foods to Eat": "Low glycemic index (GI) foods, high-fiber vegetables, and lean proteins.",
                "Foods to Avoid": "Refined carbohydrates, sugary beverages, and foods with added fructose.",
                "Natural Adjuncts": "Routine blood pressure monitoring, as hypertension exacerbates retinal vascular damage.",
                "Citations": ["WHO Global Report on Diabetes", "American Academy of Ophthalmology: Diabetic Retinopathy PPP (2023)"]
            }
        },
        "Bone Fracture": {
            "Fracture": {
                "Precautionary Measures": "Strict immobilization of the affected extremity. Elevate above heart level to reduce edema.",
                "Foods to Eat": "Calcium-rich foods (dairy, fortified plant milks) and Vitamin D sources to support osteogenesis.",
                "Foods to Avoid": "Excessive caffeine and sodium, which can increase urinary calcium excretion.",
                "Natural Adjuncts": "Supervised physical therapy after clinical union is confirmed on follow-up X-rays.",
                "Citations": ["NIH Osteoporosis and Related Bone Diseases National Resource Center", "WHO: Musculoskeletal Health"]
            }
        }
    }

    output_dir = r"D:\medical-edge-system\Final_Defense_Assets"
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, "clinical_knowledge_base.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(kb, f, indent=4, ensure_ascii=False)
        
    print(f"--- Phase 5: Clinical Knowledge Base ---")
    print(f"[SUCCESS] Evidence-based Patient Care Packages saved to:\n{file_path}")

if __name__ == "__main__":
    generate_knowledge_base()