#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import cgi
import json
import requests

API_BASE = "http://localhost:5000"
ANALYZE_URL = API_BASE + "/analyze"
HISTORY_URL = API_BASE + "/history"

HOST = "0.0.0.0"
PORT = 8088


def escape(x):
    return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def api_get_history(patient_id=None, limit=20):
    try:
        params = {"limit": str(limit)}
        if patient_id:
            params["patient_id"] = patient_id

        r = requests.get(HISTORY_URL, params=params, timeout=20)
        return r.json()
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "records": []
        }


def layout(content=""):
    return """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>NeuroEdge Hub - Medical Edge AI</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #0f172a;
            color: #e5e7eb;
            padding: 24px;
        }
        .wrap {
            max-width: 1200px;
            margin: auto;
        }
        .card {
            background: #111827;
            padding: 22px;
            border-radius: 16px;
            margin-bottom: 18px;
            border: 1px solid #1f2937;
        }
        h1 { color: #38bdf8; margin-bottom: 4px; }
        h2 { color: #93c5fd; }
        h3 { color: #bfdbfe; }
        .subtitle { color: #cbd5e1; margin-top: 0; }
        input, select, button {
            padding: 12px;
            margin: 8px 0;
            width: 100%;
            border-radius: 8px;
            border: 1px solid #334155;
            box-sizing: border-box;
        }
        input, select {
            background: #020617;
            color: #e5e7eb;
        }
        button {
            background: #2563eb;
            color: white;
            font-weight: bold;
            cursor: pointer;
            border: none;
        }
        button:hover { background: #1d4ed8; }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 18px;
        }
        .badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            background: #1e293b;
            margin: 3px;
        }
        .red { color: #ef4444; font-weight: bold; }
        .yellow { color: #eab308; font-weight: bold; }
        .green { color: #22c55e; font-weight: bold; }
        .muted { color: #94a3b8; }
        pre {
            white-space: pre-wrap;
            background: #020617;
            padding: 14px;
            border-radius: 10px;
            overflow-x: auto;
            border: 1px solid #1e293b;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        th, td {
            border-bottom: 1px solid #334155;
            padding: 9px;
            text-align: left;
            vertical-align: top;
        }
        th {
            color: #93c5fd;
            background: #020617;
        }
        a {
            color: #38bdf8;
            text-decoration: none;
        }
        .topnav {
            margin-top: 12px;
        }
        .topnav a {
            margin-right: 14px;
            font-weight: bold;
        }
        @media (max-width: 850px) {
            .grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
<div class="wrap">
    <div class="card">
        <h1>NeuroEdge Hub</h1>
        <p class="subtitle">Jetson Nano FP16 TensorRT Multi-Modal Medical AI Server</p>
        <div class="topnav">
            <a href="/">Analyze Scan</a>
            <a href="/history">Patient History</a>
        </div>
        <p>
            <span class="badge">6 TensorRT Modalities</span>
            <span class="badge">Modality Bouncer</span>
            <span class="badge">OOD Rejection</span>
            <span class="badge">Persistent SQLite Records</span>
        </p>
    </div>

    {content}
</div>
</body>
</html>""".replace("{content}", content)


def analyze_form(result_html=""):
    return layout("""
<div class="grid">
    <div class="card">
        <h2>Analyze Medical Scan</h2>

        <form method="POST" enctype="multipart/form-data">
            <label>Patient ID</label>
            <input type="text" name="patient_id" value="PT-WEB-DEMO-001" required>

            <label>Modality</label>
            <select name="modality">
                <option value="brain_mri">Brain MRI</option>
                <option value="chest_xray">Chest X-ray</option>
                <option value="bone">Bone X-ray</option>
                <option value="ecg">ECG</option>
                <option value="retinal">Retinal Fundus</option>
                <option value="skin">Skin Lesion</option>
            </select>

            <label>Upload Medical Image</label>
            <input type="file" name="file" required>

            <button type="submit">Analyze Scan</button>
        </form>

        <p class="muted">
            Safety layer: selected modality is checked by the modality bouncer before diagnosis.
            Wrong-modality or OOD scans are rejected into Tier 3 manual review.
        </p>
    </div>

    <div class="card">
        <h2>Quick History Lookup</h2>
        <form method="GET" action="/history">
            <label>Patient ID</label>
            <input type="text" name="patient_id" placeholder="e.g., PT-WEB-DEMO-001">
            <button type="submit">View Patient History</button>
        </form>
        <p><a href="/history">View latest 20 records</a></p>
    </div>
</div>

{result_html}
""".replace("{result_html}", result_html))


def make_result_html(result):
    pred = result.get("prediction", {})
    gpio = result.get("gpio_alert_type", "")
    gpio_class = str(gpio).lower()
    raw_json = json.dumps(result, indent=2)

    record_id = result.get("record_id", "N/A")
    patient_id = result.get("patient_id", "N/A")

    gradcam_html = ""
    if result.get("gradcam_base64"):
        gradcam_html = """
        <h3>Grad-CAM / Explainability</h3>
        <img style="max-width:100%; border-radius:12px;" src="data:image/png;base64,{img}">
        """.replace("{img}", escape(result.get("gradcam_base64")))

    return """
<div class="card">
    <h2>AI Result</h2>

    <p><b>Record ID:</b> {record_id}</p>
    <p><b>Patient ID:</b> {patient_id}</p>
    <p><b>Status:</b> {status}</p>
    <p><b>Modality:</b> {modality}</p>
    <p><b>Prediction:</b> {prediction}</p>
    <p><b>Confidence:</b> {confidence:.2f}%</p>
    <p><b>Uncertainty:</b> ±{uncertainty:.2f}%</p>

    <p><b>Routing Tier:</b> {routing_tier}</p>
    <p><b>Routing Decision:</b> {routing_decision}</p>
    <p><b>Risk Tier:</b> {risk_tier}</p>
    <p><b>GPIO Alert:</b> <span class="{gpio_class}">{gpio}</span></p>

    <p><b>OOD Flag:</b> {ood_flag}</p>
    <p><b>Modality Mismatch:</b> {modality_mismatch}</p>
    <p><b>Detected Modality:</b> {detected_modality}</p>

    <p><b>Manual Review Required:</b> {manual_review_required}</p>
    <p><b>Auto Report Allowed:</b> {auto_report_allowed}</p>
    <p><b>Reason:</b> {routing_reason}</p>

    <p><b>Precision:</b> {precision}</p>
    <p><b>Pipeline Latency:</b> {pipeline_ms} ms</p>

    <h3>Clinical Report</h3>
    <pre>{report}</pre>

    {gradcam_html}

    <h3>Raw JSON</h3>
    <pre>{raw_json}</pre>
</div>
""".format(
        record_id=escape(record_id),
        patient_id=escape(patient_id),
        status=escape(result.get("status")),
        modality=escape(result.get("modality")),
        prediction=escape(pred.get("class", "N/A")),
        confidence=float(pred.get("mean_confidence", 0) or 0),
        uncertainty=float(pred.get("uncertainty", 0) or 0),
        routing_tier=escape(result.get("routing_tier")),
        routing_decision=escape(result.get("routing_decision")),
        risk_tier=escape(result.get("risk_tier")),
        gpio_class=escape(gpio_class),
        gpio=escape(gpio),
        ood_flag=escape(result.get("ood_flag")),
        modality_mismatch=escape(result.get("modality_mismatch")),
        detected_modality=escape(result.get("detected_modality")),
        manual_review_required=escape(result.get("manual_review_required")),
        auto_report_allowed=escape(result.get("auto_report_allowed")),
        routing_reason=escape(result.get("routing_reason")),
        precision=escape(result.get("precision")),
        pipeline_ms=escape(result.get("pipeline_ms")),
        report=escape(result.get("llm_report")),
        gradcam_html=gradcam_html,
        raw_json=escape(raw_json),
    )


def make_history_html(patient_id=None):
    data = api_get_history(patient_id=patient_id, limit=30)
    records = data.get("records", [])

    rows = ""
    for r in records:
        pred = r.get("prediction", {})
        gpio = r.get("gpio_alert_type", "")
        gpio_class = str(gpio).lower()

        rows += """
<tr>
    <td>{record_id}</td>
    <td>{patient_id}</td>
    <td>{modality}</td>
    <td>{pred}</td>
    <td>{conf:.2f}%</td>
    <td>{tier}</td>
    <td><span class="{gpio_class}">{gpio}</span></td>
    <td>{time}</td>
</tr>
""".format(
            record_id=escape(r.get("record_id")),
            patient_id=escape(r.get("patient_id")),
            modality=escape(r.get("modality")),
            pred=escape(pred.get("class")),
            conf=float(pred.get("mean_confidence", 0) or 0),
            tier=escape(r.get("routing_tier")),
            gpio_class=escape(gpio_class),
            gpio=escape(gpio),
            time=escape(r.get("inferred_at")),
        )

        rows += """
<tr>
    <td colspan="8">
        <b>Decision:</b> {decision}<br>
        <b>Report:</b> {report}
    </td>
</tr>
""".format(
            decision=escape(r.get("routing_decision")),
            report=escape(r.get("llm_report")),
        )

    if not rows:
        rows = """
<tr>
    <td colspan="8">No records found.</td>
</tr>
"""

    title = "Patient History"
    if patient_id:
        title += " - " + escape(patient_id)

    return layout("""
<div class="card">
    <h2>{title}</h2>

    <form method="GET" action="/history">
        <label>Filter by Patient ID</label>
        <input type="text" name="patient_id" value="{patient_id_value}" placeholder="Leave blank for all records">
        <button type="submit">Search History</button>
    </form>

    <p><b>Status:</b> {status} | <b>Count:</b> {count}</p>

    <table>
        <thead>
            <tr>
                <th>Record</th>
                <th>Patient</th>
                <th>Modality</th>
                <th>Prediction</th>
                <th>Confidence</th>
                <th>Tier</th>
                <th>GPIO</th>
                <th>Time</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>

    <h3>Raw History JSON</h3>
    <pre>{raw}</pre>
</div>
""".format(
        title=title,
        patient_id_value=escape(patient_id or ""),
        status=escape(data.get("status")),
        count=escape(data.get("count", len(records))),
        rows=rows,
        raw=escape(json.dumps(data, indent=2)),
    ))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/history":
            qs = parse_qs(parsed.query)
            patient_id = None
            if "patient_id" in qs and qs["patient_id"]:
                patient_id = qs["patient_id"][0].strip() or None

            html = make_history_html(patient_id=patient_id)
        else:
            html = analyze_form()

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

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

            r = requests.post(ANALYZE_URL, files=files, data=data, timeout=240)

            try:
                result = r.json()
            except Exception:
                result = {
                    "status": "error",
                    "message": r.text
                }

            result_html = make_result_html(result)
            html = analyze_form(result_html)

        except Exception as e:
            html = analyze_form("""
<div class="card">
    <h2 class="red">Error</h2>
    <pre>{err}</pre>
</div>
""".format(err=escape(e)))

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))


if __name__ == "__main__":
    print("[WEB DEMO] Running on http://{}:{}".format(HOST, PORT))
    print("[WEB DEMO] Analyze API:", ANALYZE_URL)
    print("[WEB DEMO] History API:", HISTORY_URL)
    HTTPServer((HOST, PORT), Handler).serve_forever()
