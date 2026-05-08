# Appendix B: System Model Cards & Hyperparameter Configurations

## B.1 Overview
This appendix provides comprehensive documentation for the Machine Learning models deployed in the Edge Medical Diagnostic System. Following industry standards for AI transparency, these model cards detail the architecture, training configurations (hyperparameters), and quantitative evaluation metrics (confusion matrices) for the primary classifiers optimized for the NVIDIA Jetson Nano 2GB.

---

## B.2 Master Architecture & Hyperparameters
To ensure uniformity and maximize memory efficiency on the edge device, all six primary diagnostic models share a standardized backbone architecture and training pipeline.

| Hyperparameter / Setting | Configuration | Justification |
| :--- | :--- | :--- |
| **Feature Extractor** | EfficientNet-B0 | Optimal balance of parameter count (5.3M) and high spatial resolution extraction. |
| **Optimizer** | Adam | Fast convergence for medical imagery with sparse gradients. |
| **Learning Rate (Initial)** | 1e-4 (0.0001) | Prevents overshooting global minima during fine-tuning of pre-trained weights. |
| **Batch Size** | 32 | Maximizes GPU memory utilization during local RTX training. |
| **Loss Function** | Categorical Cross-Entropy | Standard for multi-class medical diagnosis. |
| **Regularization** | Dropout (p=0.2) | Prevents overfitting on smaller localized datasets. |
| **Edge Quantization** | TensorRT FP16 | Reduces model size from 21.5MB to 10.8MB; required for Jetson Nano 2GB VRAM constraints. |
| **Continual Learning** | EWC (λ = 100.0) | Elastic Weight Consolidation used to prevent catastrophic forgetting during sequential training. |

---

## B.3 Primary Specialist Model Cards

### Model 1: Pneumonia Specialist (Anchor Task)
* **Domain:** Chest X-Ray (CXR)
* **Classes (2):** Normal, Pneumonia (Viral/Bacterial)
* **Intended Use:** Rapid triaging of respiratory infections in rural clinics.
* **Performance Constraints:** Must maintain >98% accuracy as the foundational anchor for the EWC Continual Learning sequence.

**Confusion Matrix:**
*(Insert your generated confusion matrix image here: e.g., `![Pneumonia CM](assets/pneumonia_cm.png)`)*

| | Predicted: Normal | Predicted: Pneumonia |
|---|---|---|
| **Actual: Normal** | True Negative (TN) | False Positive (FP) |
| **Actual: Pneumonia**| False Negative (FN) | True Positive (TP) |
*Note: The model prioritizes minimizing False Negatives (FN) to ensure no severe infections are sent home without treatment.*

---

### Model 2: Bone Fracture Specialist
* **Domain:** Musculoskeletal Radiography (X-Ray)
* **Classes (2):** Intact, Fractured
* **Intended Use:** Emergency room triage for physical trauma.

**Confusion Matrix:**
*(Insert your generated confusion matrix image here)*

---

### Model 3: Brain Tumor Specialist (MRI)
* **Domain:** Magnetic Resonance Imaging (MRI)
* **Classes (4):** Glioma, Meningioma, Pituitary Tumor, No Tumor
* **Intended Use:** Assisting neurologists in identifying abnormal masses in non-contrast scans.

**Confusion Matrix:**
*(Insert your generated confusion matrix image here)*

---

### Model 4: Retinal Disease Specialist
* **Domain:** Fundus Photography
* **Classes (5):** Normal, Mild DR, Moderate DR, Severe DR, Proliferative DR
* **Intended Use:** Diabetic Retinopathy (DR) screening in regions lacking specialized ophthalmologists.

**Confusion Matrix:**
*(Insert your generated confusion matrix image here)*

---

### Model 5: Skin Cancer Specialist
* **Domain:** Dermatoscopic Imagery
* **Classes (7):** Melanoma, Nevus, Basal Cell Carcinoma, Actinic Keratosis, Benign Keratosis, Dermatofibroma, Vascular Lesion
* **Intended Use:** Non-invasive preliminary screening for malignant dermal lesions.

**Confusion Matrix:**
*(Insert your generated confusion matrix image here)*

---

### Model 6: Cardiovascular Specialist (ECG)
* **Domain:** 12-Lead Electrocardiogram (Converted to 2D Spectrograms/Signals)
* **Classes (5):** Normal, Supraventricular Premature, Premature Ventricular, Fusion, Unclassifiable
* **Intended Use:** Real-time arrhythmia detection for intensive care monitoring.

**Confusion Matrix:**
*(Insert your generated confusion matrix image here)*

---

## B.4 Ethical Considerations & Limitations
* **Hardware Limitations:** Due to the 2GB VRAM constraint of the Jetson Nano, models were aggressively quantized to FP16. While ablation studies showed a minimal <0.5% drop in accuracy, edge cases with highly noisy X-rays may exhibit reduced confidence scores compared to cloud-based FP32 models.
* **Data Bias:** Models were trained on publicly available global datasets. Performance may vary on local demographic data (Client Drift). Future deployments will utilize the implemented Federated Learning pipeline (FedAvg) to adapt to local hospital populations safely.
* **Human-in-the-Loop:** This system is classified as a Clinical Decision Support System (CDSS). It is designed to *assist* medical professionals via the Phi-2 LLM generated reports, not to autonomously replace human medical judgment.