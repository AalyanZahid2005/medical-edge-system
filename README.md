# 🏥 Multi-Modal AI-Accelerated Medical Edge System

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![TensorRT](https://img.shields.io/badge/TensorRT-FP16%2FINT8-76B900.svg)
![Hardware](https://img.shields.io/badge/Hardware-NVIDIA_Jetson_Nano-76B900.svg)

An offline, commercial-grade Multi-Modal Medical AI Edge Server designed for the NVIDIA Jetson Nano. This system processes 6 different medical modalities through a custom 3-Stage Threaded TensorRT pipeline, offering real-time diagnostics, hardware routing, and continual learning without relying on cloud compute.

## 🚀 Core Architecture

1. **Stage 1: Preprocessing (CPU)** - Asynchronous OpenCV decoding and Out-Of-Distribution (OOD) anomaly rejection.
2. **Stage 2: Inference (GPU)** - High-speed parallel execution using TensorRT FP16/INT8 engines.
3. **Stage 3: Post-Processing (CPU)** - MC Dropout Bayesian Uncertainty, True CAM generation, Cross-Modal Transformer Fusion, and SHAP explainability.

## ✨ Advanced Features

* **6-in-1 Modality Support:** Brain MRI, Chest X-Ray, Retinal Fundus, ECG Signal, Skin Lesion, and Bone X-Ray.
* **Bayesian Uncertainty:** 20-pass MC Dropout to quantify AI confidence intervals.
* **Hardware Gated Routing:** Tiers 1-3 routing logic triggering physical GPIO hardware (Green/Red LEDs, Piezo Buzzer) and DICOM SR routing to Orthanc PACS.
* **Elastic Weight Consolidation (EWC):** Continual learning from doctor corrections without catastrophic forgetting.
* **Federated Learning (FedAvg):** Multi-node weight aggregation for privacy-preserving model updates.
* **Enterprise Security:** AES-256-GCM model encryption in RAM, JWT API authentication, and autonomous DICOM C-FIND polling.

## 👥 The Team
* **Aalyan Zahid** - Edge Architecture, Hardware, TensorRT Pipeline
* **Hamid** - ML Models, Transformer Fusion, SHAP
* **Ahmed Rasheed** - Backend API, DICOM Networking, SQLite
* **Aalyan Zahid (Node 2)** - GUI Dashboard, Flutter Mobile App, Documentation
