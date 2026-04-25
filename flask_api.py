from flask import Flask, request, jsonify
import time

app = Flask(__name__)

# ---------------------------------------------------------
# STUB ENDPOINTS FOR GUI DEVELOPMENT (DAY 4)
# ---------------------------------------------------------

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    STUB: Receives a single image from the GUI.
    Returns fake JSON simulating the 3-Stage Pipeline output.
    """
    # Mudassir (M4) can hit this endpoint right now to build the 'Scan Upload' tab
    time.sleep(1.5) # Simulate pipeline processing latency
    
    # Contract: This is the exact JSON structure the final system will use
    response = {
        "status": "success",
        "modality": "brain_mri",
        "prediction": {
            "class": "Glioma",
            "mean_confidence": 92.5,
            "uncertainty": 1.2
        },
        "ood_flag": False,
        "routing_tier": 1,
        "llm_report": "The MRI demonstrates a lesion in the temporal lobe consistent with glioma. AI confidence: 92.5%.",
        # Base64 encoded 1x1 pixel image just so the GUI image renderer doesn't crash
        "gradcam_base64": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=" 
    }
    return jsonify(response), 200

@app.route('/analyze_patient', methods=['POST'])
def analyze_patient():
    """
    STUB: Receives multiple images for Patient Fusion.
    Returns fake risk score simulating the Late-Fusion MLP.
    """
    time.sleep(2.0)
    response = {
        "status": "success",
        "fusion_risk_score": "Critical",
        "details": "Patient exhibits multi-modal Glioma and Pneumonia signatures."
    }
    return jsonify(response), 200

@app.route('/correct', methods=['POST'])
def correct_label():
    """
    STUB: Receives doctor's correction from the Active Learning Queue.
    Simulates the Elastic Weight Consolidation (EWC) fine-tuning loop.
    """
    time.sleep(3.0) # EWC takes a few seconds on the Jetson CPU
    response = {
        "status": "success",
        "message": "EWC Fine-tuning complete. Model updated without catastrophic forgetting."
    }
    return jsonify(response), 200

if __name__ == '__main__':
    print("[SYSTEM] Starting API Skeleton with Stub Endpoints on port 5000...")
    # threaded=True is required for when we introduce cuda_ctx.push() later
    app.run(host='0.0.0.0', port=5000, threaded=True)