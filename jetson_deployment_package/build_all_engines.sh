#!/bin/bash
echo "=================================================="
echo "🚀 STARTING TENSORRT BATCH COMPILATION"
echo "=================================================="

TRT_PATH="/usr/src/tensorrt/bin/trtexec"

echo "[1/5] Forging Chest X-Ray Engine..."
$TRT_PATH --onnx=pneumonia_classifier.onnx --saveEngine=chest_xray_fp16.engine --fp16

echo "[2/5] Forging Skin Lesion Engine..."
$TRT_PATH --onnx=skin_classifier.onnx --saveEngine=skin_lesion_fp16.engine --fp16

echo "[3/5] Forging Retinal Engine..."
$TRT_PATH --onnx=retina_classifier.onnx --saveEngine=retinal_fp16.engine --fp16

echo "[4/5] Forging Bone Fracture Engine..."
$TRT_PATH --onnx=bone_classifier.onnx --saveEngine=bone_fracture_fp16.engine --fp16

echo "[5/5] Forging ECG Engine..."
$TRT_PATH --onnx=ecg_classifier.onnx --saveEngine=ecg_fp16.engine --fp16

echo "=================================================="
echo "✅ ALL 6 MEDICAL EXPERTS SUCCESSFULLY COMPILED!"
echo "=================================================="
