# Official Model Card: Multi-Modal Medical Edge AI
**Clearance Target:** FDA 510(k) Class II Medical Device / DRAP
**Version:** 1.0 (Jetson TensorRT INT8 Build)

## 1. Model Details
* **Architecture:** 6 Independent Convolutional Neural Networks (CNN) with paired Autoencoder Out-of-Distribution (OOD) Guards.
* **Optimization:** NVIDIA TensorRT INT8 Quantization (Post-Training Calibration).
* **Deployment Hardware:** NVIDIA Jetson Nano (Edge Node) completely offline.
* **Clinical Modalities:** Brain MRI, Chest X-ray, Retinal Fundus, ECG Signal, Skin Lesion, Bone Fracture.

## 2. Intended Use
* **Primary Function:** Clinical Decision Support System (CDSS) and triage prioritization in resource-constrained hospital environments. 
* **Out of Scope:** This system is **NOT** an autonomous diagnostic tool. It is designed to flag abnormalities and route them to a human physician, alongside relevant SNOMED-CT codes and WHO-cited care packages.

## 3. Safety & Fallback Protocols (Crucial for Q&A)
To ensure patient safety and prevent medical malpractice, this system enforces two strict algorithmic guardrails:
1. **The "Bouncer" (OOD Detection):** Before inference, an Autoencoder evaluates the input image. If the image does not match the distribution of the target medical modality (e.g., a nurse uploads a dog photo instead of an X-ray), the system rejects the file and refuses to output a diagnosis.
2. **The 80% Confidence Threshold:** If the primary classifier's confidence drops below 80%, the system overrides the automated SNOMED-CT/ICD-10 billing code assignment. It reverts to a "Manual Review Required" state and provides the doctor with the top 3 differential diagnoses.

## 4. Performance & Metrics
*(Note: Aalyan to fill in final Jetson INT8 benchmark numbers here)*
* **Target Latency:** < 50ms per scan on Jetson Nano.
* **Precision Target:** INT8 accuracy must remain within 1.5% of the original FP32 PyTorch baseline.
* **Evaluation Metrics:** ROC-AUC, Sensitivity (Recall), and Specificity. (False Negatives are penalized heavily during training to avoid missing a critical diagnosis).

## 5. Ethical Considerations & Bias
* Models were trained on diverse, open-source datasets (e.g., NIH ChestX-ray14, HAM10000). 
* **Known Limitations:** Skin lesion detection accuracy may vary across Fitzpatrick skin types due to historical dataset imbalances. Continuous longitudinal data collection via the hospital PACS is required to fine-tune the model for the local demographic.