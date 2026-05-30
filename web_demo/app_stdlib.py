#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, HTTPServer
import cgi
import json
import requests

API_URL = "http://localhost:5000/analyze"
HOST = "0.0.0.0"
PORT = 8088


HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Medical Edge AI Demo</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #0f172a;
            color: #e5e7eb;
            padding: 30px;
        }
        .card {
            background: #111827;
            padding: 25px;
            border-radius: 14px;
            max-width: 900px;
            margin: auto;
        }
        h1 { color: #38bdf8; }
        input, select, button {
            padding: 12px;
            margin: 8px 0;
            width: 100%;
            border-radius: 8px;
            border: none;
        }
        button {
            background: #2563eb;
            color: white;
            font-weight: bold;
            cursor: pointer;
        }
        pre {
            white-space: pre-wrap;
            background: #020617;
            padding: 15px;
            border-radius: 10px;
            overflow-x: auto;
        }
        .red { color: #ef4444; font-weight: bold; }
        .yellow { color: #eab308; font-weight: bold; }
        .green { color: #22c55e; font-weight: bold; }
    </style>
</head>
<body>
<div class="card">
    <h1>Medical Edge AI Server</h1>
    <p>Jetson Nano FP16 TensorRT Inference Demo</p>

    <form method="POST" enctype="multipart/form-data">
        <label>Patient ID</label>
        <input type="text" name="patient_id" value="PT-WEB-DEMO-001" required>

        <label>Modality</label>
        <select name="modality">
            <option value="brain_mri">Brain MRI - Active Validated Demo</option>
        </select>
        <p style="color:#fbbf24;">
            Safety note: Current Jetson demo build enables the validated Brain MRI TensorRT pipeline only.
            Other modalities are locked for final defense.
        </p>

        <label>Upload Medical Image</label>
        <input type="file" name="file" required>

        <button type="submit">Analyze Scan</button>
    </form>

    {result_html}
</div>
</body>
</html>
"""


def page(content=""):
    return HTML_FORM.replace("{result_html}", content)


def escape(x):
    return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def make_result_html(result):
    pred = result.get("prediction", {})
    gpio = result.get("gpio_alert_type", "")
    gpio_class = gpio.lower()

    raw_json = json.dumps(result, indent=2)

    return f"""
    <hr>
    <h2>AI Result</h2>

    <p><b>Status:</b> {escape(result.get("status"))}</p>
    <p><b>Modality:</b> {escape(result.get("modality"))}</p>
    <p><b>Prediction:</b> {escape(pred.get("class", "N/A"))}</p>
    <p><b>Confidence:</b> {float(pred.get("mean_confidence", 0)):.2f}%</p>
    <p><b>Uncertainty:</b> ±{float(pred.get("uncertainty", 0)):.2f}%</p>

    <p><b>Routing Tier:</b> {escape(result.get("routing_tier"))}</p>
    <p><b>Routing Decision:</b> {escape(result.get("routing_decision"))}</p>
    <p><b>Risk Tier:</b> {escape(result.get("risk_tier"))}</p>
    <p><b>GPIO Alert:</b> <span class="{gpio_class}">{escape(gpio)}</span></p>

    <p><b>Manual Review Required:</b> {escape(result.get("manual_review_required"))}</p>
    <p><b>Auto Report Allowed:</b> {escape(result.get("auto_report_allowed"))}</p>
    <p><b>Reason:</b> {escape(result.get("routing_reason"))}</p>

    <p><b>Precision:</b> {escape(result.get("precision"))}</p>
    <p><b>Pipeline Latency:</b> {escape(result.get("pipeline_ms"))} ms</p>

    <h3>Clinical Report</h3>
    <pre>{escape(result.get("llm_report"))}</pre>

    <h3>Raw JSON</h3>
    <pre>{escape(raw_json)}</pre>
    """


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(page().encode("utf-8"))

    def do_POST(self):
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type"),
                }
            )

            patient_id = form.getvalue("patient_id", "PT-WEB-DEMO-001")
            modality = form.getvalue("modality", "brain_mri")

            file_item = form["file"]
            file_bytes = file_item.file.read()
            filename = file_item.filename or "scan.jpg"

            files = {
                "file": (filename, file_bytes, "image/jpeg")
            }

            data = {
                "patient_id": patient_id,
                "modality": modality
            }

            r = requests.post(API_URL, files=files, data=data, timeout=180)

            try:
                result = r.json()
            except Exception:
                result = {
                    "status": "error",
                    "message": r.text
                }

            result_html = make_result_html(result)

        except Exception as e:
            result_html = f"""
            <hr>
            <h2 class="red">Error</h2>
            <pre>{escape(e)}</pre>
            """

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(page(result_html).encode("utf-8"))


if __name__ == "__main__":
    print(f"[WEB DEMO] Running on http://{HOST}:{PORT}")
    print(f"[WEB DEMO] Forwarding scans to {API_URL}")
    HTTPServer((HOST, PORT), Handler).serve_forever()
