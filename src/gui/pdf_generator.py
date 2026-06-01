import os
import base64
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
)

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except Exception:
    arabic_reshaper = None
    get_display = None

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None


PAGE_W, PAGE_H = A4

C_NAVY = colors.HexColor("#071426")
C_PANEL = colors.HexColor("#F6F9FC")
C_BORDER = colors.HexColor("#D7E0EA")
C_TEXT = colors.HexColor("#0B2239")
C_MUTED = colors.HexColor("#63748A")
C_CYAN = colors.HexColor("#00BFEA")
C_GREEN = colors.HexColor("#00C853")
C_YELLOW = colors.HexColor("#FFB300")
C_RED = colors.HexColor("#FF1744")
C_PURPLE = colors.HexColor("#6C4CF6")
C_WHITE = colors.white


def _safe(value, default="N/A"):
    if value is None:
        return default
    value = str(value)
    return value if value.strip() else default


def _has_rtl(value):
    value = str(value or "")
    return any("\u0600" <= ch <= "\u06FF" for ch in value)


def _fmt_percent(value):
    try:
        return "{:.1f}%".format(float(value))
    except Exception:
        return _safe(value, "N/A")


def _aalyan_format_uncertainty(patient_data):
    """
    Robustly extract MC uncertainty for the PDF report.
    Handles values like:
    1.50, 0.015, "± 1.50%", "uncertainty across 20 forward passes is ±1.50%"
    """
    import re

    data = patient_data or {}

    keys = [
        "uncertainty",
        "mc_uncertainty",
        "mc_uncertainty_pct",
        "uncertainty_pct",
        "monte_carlo_uncertainty",
        "monte_carlo_variance",
        "mc_variance",
        "variance",
        "std_pct",
    ]

    def clean_value(v):
        if v is None:
            return None

        s = str(v).strip()
        if not s or s.lower() in {"n/a", "none", "null", "--", "-"}:
            return None

        # Already formatted
        if "±" in s:
            m = re.search(r"±\s*([0-9]+(?:\.[0-9]+)?)\s*%?", s)
            if m:
                return f"± {float(m.group(1)):.2f}%"

        # Plain percent string
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*%", s)
        if m:
            return f"± {float(m.group(1)):.2f}%"

        # Plain number
        try:
            x = float(s)
            if 0 < x < 1:
                x *= 100.0
            return f"± {x:.2f}%"
        except Exception:
            return None

    for k in keys:
        val = clean_value(data.get(k))
        if val:
            return val

    # Last fallback: parse from clinical report text
    blob = " ".join(str(v) for v in data.values() if isinstance(v, (str, int, float)))
    for pat in [
        r"±\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
        r"uncertainty[^0-9±]{0,80}([0-9]+(?:\.[0-9]+)?)\s*%?",
        r"variance[^0-9±]{0,80}([0-9]+(?:\.[0-9]+)?)\s*%?",
    ]:
        m = re.search(pat, blob, flags=re.IGNORECASE)
        if m:
            return f"± {float(m.group(1)):.2f}%"

    return "N/A"



def _aalyan_normalize_pdf_data(patient_data):
    import re

    data = dict(patient_data or {})

    def bad(v):
        s = str(v or "").strip()
        return s == "" or s.lower() in {"n/a", "na", "none", "null", "--", "-", "not entered"}

    def put(keys, val, force=False):
        if val is None:
            return
        val = str(val).strip()
        if bad(val):
            return
        for k in keys:
            if force or bad(data.get(k)):
                data[k] = val

    selected = str(
        data.get("_selected_patient_text")
        or data.get("patient_display")
        or data.get("active_patient")
        or data.get("selected_patient")
        or ""
    )

    if selected:
        selected = selected.replace("Active Patient:", "").replace("Patient:", "").strip()
        parts = [x.strip() for x in selected.split("|") if x.strip()]
        if len(parts) >= 1:
            put(("patient_id",), re.sub(r"\s*Record ID\s*:\s*\d+.*$", "", parts[0], flags=re.I).strip(), True)
        if len(parts) >= 2 and "record id" not in parts[1].lower():
            put(("patient_name", "name"), parts[1], True)
        if len(parts) >= 3:
            m = re.search(r"(\d{1,3})\s*/\s*([A-Za-z]+)", parts[2])
            if m:
                put(("age", "patient_age"), m.group(1), True)
                put(("gender", "patient_gender"), m.group(2), True)

    pid = str(data.get("patient_id", "")).strip()
    m = re.search(r"^(PT-[^|]+)\s*\|\s*Record ID\s*:\s*(\d+)", pid, flags=re.I)
    if m:
        data["patient_id"] = m.group(1).strip()
        data["record_id"] = m.group(2).strip()

    blob = " ".join(str(v) for v in data.values() if isinstance(v, (str, int, float)))

    m = re.search(r"Record ID\s*[:#]\s*(\d+)", blob, flags=re.I)
    if m:
        data["record_id"] = m.group(1)

    unc = None

    for k in ("_aalyan_uncertainty_display", "mc_uncertainty", "uncertainty", "uncertainty_pct", "mc_uncertainty_pct"):
        v = data.get(k)
        if v is not None and not bad(v):
            mm = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(v))
            if mm:
                try:
                    unc = float(mm.group(1))
                    break
                except Exception:
                    pass

    if unc is None:
        for pat in (
            r"±\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
            r"uncertainty[^0-9±]{0,120}([0-9]+(?:\.[0-9]+)?)\s*%?",
            r"variance[^0-9±]{0,120}([0-9]+(?:\.[0-9]+)?)\s*%?",
        ):
            mm = re.search(pat, blob, flags=re.I)
            if mm:
                try:
                    unc = float(mm.group(1))
                    break
                except Exception:
                    pass

    if unc is not None:
        if 0 < unc < 1:
            unc *= 100.0
        data["uncertainty"] = f"{unc:.2f}%"
        data["mc_uncertainty"] = f"{unc:.2f}%"
        data["uncertainty_pct"] = f"{unc:.2f}%"
        data["_aalyan_uncertainty_display"] = f"± {unc:.2f}%"

    return data



def _aalyan_safe_contact_mrn_display(patient_data):
    data = dict(patient_data or {})

    def clean(v):
        s = str(v or "").strip()
        if s.lower() in {"", "n/a", "na", "none", "null", "-", "--", "not entered"}:
            return ""
        return s

    existing = clean(
        data.get("_aalyan_contact_mrn_display")
        or data.get("contact_mrn_display")
        or data.get("contact_mrn")
    )
    if existing:
        return existing

    contact = clean(
        data.get("contact")
        or data.get("phone")
        or data.get("mobile")
        or data.get("contact_no")
        or data.get("contact_number")
    )

    mrn = clean(
        data.get("mrn")
        or data.get("MRN")
        or data.get("medical_record_no")
        or data.get("medical_record_number")
    )

    if contact and mrn:
        return f"{contact} / MRN: {mrn}"
    if contact:
        return contact
    if mrn:
        return f"MRN: {mrn}"
    return "N/A"


class ProfessionalPDFReportGenerator:
    def _aalyan_best_clinical_interpretation(self, data=None, language="English"):
        """
        Build a safe clinical-summary fallback when the LLM/Phi-2 report text is empty.
        Prevents professional reports from showing: self._aalyan_best_clinical_interpretation((locals().get("patient_data") or locals().get("report_data") or locals().get("data") or locals()), locals().get("language", "English"))
        """
        import ast

        lang = str(language or "English").lower()

        def clean(v):
            s = str(v or "").strip()
            if s.lower() in {"", "none", "null", "n/a", "na", "-", "--", "not entered"}:
                return ""
            return s

        def parse_maybe(v):
            if isinstance(v, dict):
                return v
            if isinstance(v, str):
                s = v.strip()
                if s.startswith("{") and s.endswith("}"):
                    try:
                        out = ast.literal_eval(s)
                        if isinstance(out, dict):
                            return out
                    except Exception:
                        return v
            return v

        sources = []

        def add_source(x):
            x = parse_maybe(x)
            if isinstance(x, dict):
                sources.append(x)
                for k in [
                    "result", "results", "prediction", "analysis", "ai_result",
                    "patient_data", "report_data", "record", "last_result"
                ]:
                    if k in x:
                        y = parse_maybe(x.get(k))
                        if isinstance(y, dict):
                            sources.append(y)

        add_source(data)

        if isinstance(data, dict):
            for k, v in list(data.items()):
                if isinstance(v, dict):
                    add_source(v)

        # Prefer real report text if present
        text_keys = [
            "clinical_interpretation", "clinical_summary", "clinical_report",
            "llm_report", "phi2_report", "phi_2_report", "report_text",
            "report", "summary", "interpretation"
        ]

        for src in sources:
            for k in text_keys:
                val = src.get(k)
                val2 = parse_maybe(val)
                if isinstance(val2, dict):
                    continue
                s = clean(val2)
                if not s:
                    continue
                if "no clinical interpretation generated" in s.lower():
                    continue
                if len(s) > 20:
                    return s

        def get_any(keys):
            for src in sources:
                for k in keys:
                    if k in src:
                        val = parse_maybe(src.get(k))
                        if isinstance(val, dict):
                            # prediction dict usually contains class/confidence/uncertainty
                            for kk in keys:
                                if kk in val:
                                    got = clean(val.get(kk))
                                    if got:
                                        return got
                            if "class" in val:
                                got = clean(val.get("class"))
                                if got:
                                    return got
                        got = clean(val)
                        if got:
                            return got
            return ""

        patient_id = get_any(["patient_id", "Patient ID", "pid"])
        modality = get_any(["modality", "selected_modality", "scan_modality", "backend_modality"])
        pred = get_any([
            "predicted_class", "prediction", "ai_prediction", "ai_diagnosis",
            "diagnosis", "class", "pred_name"
        ])

        # If prediction came as a dict string, extract class/confidence/uncertainty
        for src in sources:
            for k in ["prediction", "ai_prediction", "result", "results"]:
                val = parse_maybe(src.get(k))
                if isinstance(val, dict):
                    pred = pred or clean(val.get("class"))
                    break

        conf = get_any(["confidence", "mean_confidence", "confidence_pct", "conf_pct", "mean_confidence_pct"])
        unc = get_any(["uncertainty", "mc_uncertainty", "variance", "mc_variance"])
        routing = get_any(["routing", "routing_tier", "tier", "routing_label"])

        def pct(v):
            s = clean(v)
            if not s:
                return ""
            if "%" in s:
                return s
            try:
                f = float(s)
                if f <= 1:
                    f *= 100
                return f"{f:.1f}%"
            except Exception:
                return s

        conf = pct(conf)

        if unc:
            u = clean(unc).replace("±", "").replace("%", "").strip()
            try:
                unc = f"± {float(u):.2f}%"
            except Exception:
                unc = clean(unc)

        if not pred:
            pred = "an abnormal / review-required finding"

        if not modality:
            modality = "medical imaging"

        if "urdu" in lang:
            return (
                f"طبی خلاصہ: مصنوعی ذہانت کے نظام نے مریض {patient_id or 'کے'} {modality} اسکین کا تجزیہ کیا۔ "
                f"نتیجہ {pred} ظاہر ہوا"
                + (f"، اعتماد {conf}" if conf else "")
                + (f"، اور غیر یقینی {unc}" if unc else "")
                + "۔ یہ رپورٹ مصنوعی ذہانت کی مدد سے تیار کی گئی ہے اور ڈاکٹر/ریڈیالوجسٹ سے تصدیق ضروری ہے۔"
            )

        if "arabic" in lang:
            return (
                f"الملخص السريري: قام نظام الذكاء الاصطناعي بتحليل فحص {modality} للمريض {patient_id or ''}. "
                f"أظهرت النتيجة {pred}"
                + (f" بدرجة ثقة {conf}" if conf else "")
                + (f" وبدرجة عدم يقين {unc}" if unc else "")
                + ". هذه النتيجة مساعدة سريرية وليست تشخيصاً نهائياً، ويجب مراجعتها بواسطة طبيب أو اختصاصي أشعة."
            )

        return (
            f"CLINICAL SUMMARY: The AI edge node analyzed the {modality} scan"
            + (f" for patient {patient_id}" if patient_id else "")
            + f" and produced the finding: {pred}."
            + (f" Mean confidence is {conf}." if conf else "")
            + (f" Monte Carlo uncertainty is {unc}." if unc else "")
            + (f" Routing decision: {routing}." if routing else "")
            + " This is a clinical decision-support output and requires verification by a qualified doctor/radiologist."
        )


    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.rtl_font_path = None
        self.rtl_font_name = "Helvetica"
        self._register_fonts()
        self.styles = getSampleStyleSheet()
        self._build_styles()

    def _register_fonts(self):
        """
        Register a general RTL font for ReportLab fallback.
        PIL rendering below will choose Urdu/Arabic-specific fonts separately.
        """
        import os
        import subprocess

        candidates = []

        for family in [
            "Noto Nastaliq Urdu",
            "Noto Naskh Arabic",
            "Nafees Nastaleeq",
            "Nafees Web Naskh",
            "DejaVu Sans",
        ]:
            try:
                out = subprocess.check_output(
                    ["fc-match", "-f", "%{file}", family],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                if out:
                    candidates.append(out)
            except Exception:
                pass

        candidates += [
            "/usr/share/fonts/truetype/noto/NotoNastaliqUrdu-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
            "/usr/share/fonts/truetype/nafees/NafeesNastaleeq.ttf",
            "/usr/share/fonts/truetype/nafees/NafeesWeb.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            os.path.join(self.base_dir, "urdu_font.ttf"),
            os.path.join(self.base_dir, "arabic_font.ttf"),
        ]

        seen = set()
        for path in candidates:
            if not path or path in seen:
                continue
            seen.add(path)

            if os.path.exists(path):
                self.rtl_font_path = path
                try:
                    pdfmetrics.registerFont(TTFont("RTLFont", path))
                    self.rtl_font_name = "RTLFont"
                except Exception:
                    self.rtl_font_name = "Helvetica"

                print("[PDF RTL FONT] ReportLab fallback:", path)
                return

        self.rtl_font_path = None
        self.rtl_font_name = "Helvetica"

    def _build_styles(self):
        self.section = ParagraphStyle(
            "Section",
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            textColor=C_TEXT,
            spaceBefore=4,
            spaceAfter=5,
        )

        self.body = ParagraphStyle(
            "Body",
            fontName="Helvetica",
            fontSize=8.4,
            leading=11,
            textColor=C_TEXT,
        )

        self.body_small = ParagraphStyle(
            "BodySmall",
            fontName="Helvetica",
            fontSize=7.6,
            leading=9.5,
            textColor=C_TEXT,
        )

        self.rtl_body = ParagraphStyle(
            "RTLBody",
            fontName=self.rtl_font_name,
            fontSize=8.7,
            leading=13,
            textColor=C_TEXT,
            alignment=TA_RIGHT,
            wordWrap="RTL",
        )

        self.rtl_title = ParagraphStyle(
            "RTLTitle",
            fontName=self.rtl_font_name,
            fontSize=9.3,
            leading=12,
            textColor=C_TEXT,
            alignment=TA_RIGHT,
            wordWrap="RTL",
        )

        self.label = ParagraphStyle(
            "Label",
            fontName="Helvetica-Bold",
            fontSize=6.8,
            leading=8,
            textColor=C_MUTED,
        )

        self.value = ParagraphStyle(
            "Value",
            fontName="Helvetica-Bold",
            fontSize=8.2,
            leading=10,
            textColor=C_TEXT,
        )

        self.value_red = ParagraphStyle(
            "ValueRed",
            fontName="Helvetica-Bold",
            fontSize=8.2,
            leading=10,
            textColor=C_RED,
        )

        self.value_green = ParagraphStyle(
            "ValueGreen",
            fontName="Helvetica-Bold",
            fontSize=8.2,
            leading=10,
            textColor=C_GREEN,
        )

        self.callout_title = ParagraphStyle(
            "CalloutTitle",
            fontName="Helvetica-Bold",
            fontSize=8.8,
            leading=11,
            textColor=C_TEXT,
        )

    def _p(self, text, style=None):
        style = style or self.body
        text = escape(_safe(text, ""))
        text = text.replace("\n", "<br/>")
        return Paragraph(text, style)

    def _rtl_visual_line(self, line):
        line = str(line or "")
        if arabic_reshaper is not None and get_display is not None:
            try:
                return get_display(arabic_reshaper.reshape(line))
            except Exception:
                return line
        return line

    def _rtl_paragraph_text(self, text):
        lines = []
        for raw in str(text or "").split("\n"):
            raw = raw.strip()
            if not raw:
                lines.append("")
            else:
                lines.append(escape(self._rtl_visual_line(raw)))
        return "<br/>".join(lines)

    def _tier_color(self, tier, gpio=None, prediction=None):
        prediction = str(prediction or "").lower()
        gpio = str(gpio or "").upper()

        if "mismatch" in prediction or "rejected" in prediction:
            return C_RED
        if gpio == "RED" or str(tier) == "3":
            return C_RED
        if gpio == "YELLOW" or str(tier) == "2":
            return C_YELLOW
        return C_GREEN

    def _risk_label(self, tier, routing, prediction):
        prediction = str(prediction or "").lower()

        if "mismatch" in prediction:
            return "REJECTED - MODALITY MISMATCH"
        if "rejected" in prediction:
            return "REJECTED - MANUAL REVIEW"
        if str(tier) == "1":
            return "TIER 1 - AUTO-SR / HIGH CONFIDENCE"
        if str(tier) == "2":
            return "TIER 2 - AI-ASSISTED REVIEW"
        if str(tier) == "3":
            return "TIER 3 - URGENT MANUAL REVIEW"
        return _safe(routing, "CLINICAL REVIEW REQUIRED")

    def _header_footer(self, canvas, doc):
        canvas.saveState()

        canvas.setFillColor(C_NAVY)
        canvas.rect(0, PAGE_H - 22 * mm, PAGE_W, 22 * mm, fill=1, stroke=0)

        canvas.setFillColor(C_CYAN)
        canvas.rect(0, PAGE_H - 22 * mm, 5 * mm, 22 * mm, fill=1, stroke=0)

        canvas.setFillColor(C_WHITE)
        canvas.setFont("Helvetica-Bold", 12)
        canvas.drawString(
            13 * mm,
            PAGE_H - 9 * mm,
            "Accelerated Medical Imaging using Machine Learning Architectures",
        )

        canvas.setFillColor(colors.HexColor("#C7D5E5"))
        canvas.setFont("Helvetica-Bold", 7.2)
        canvas.drawString(
            13 * mm,
            PAGE_H - 14 * mm,
            "Final Year Project | University of Engineering and Technology, Lahore",
        )

        canvas.setFont("Helvetica", 6.5)
        canvas.drawString(
            13 * mm,
            PAGE_H - 18 * mm,
            "MEDAI Edge Server v2.0 | Multi-Modal Diagnostic AI Report | Jetson FP16 Edge Inference",
        )

        canvas.setStrokeColor(C_BORDER)
        canvas.line(15 * mm, 14 * mm, PAGE_W - 15 * mm, 14 * mm)

        canvas.setFillColor(C_MUTED)
        canvas.setFont("Helvetica-Oblique", 6.3)
        canvas.drawString(
            15 * mm,
            9.5 * mm,
            "Final Year Project Prototype | UET Lahore | AI-assisted output; certified clinician/radiologist review is required.",
        )

        canvas.setFont("Helvetica", 6.3)
        canvas.drawRightString(PAGE_W - 15 * mm, 9.5 * mm, "Page {}".format(doc.page))

        canvas.restoreState()

    def _pick_rtl_font_path(self, language="Urdu"):
        """
        Choose the best font for Urdu/Arabic image rendering.
        """
        import os
        import subprocess

        lang = str(language or "").lower()

        if "urdu" in lang:
            families = [
                "Noto Nastaliq Urdu",
                "Nafees Nastaleeq",
                "Noto Naskh Arabic",
                "DejaVu Sans",
            ]
            direct = [
                "/usr/share/fonts/truetype/noto/NotoNastaliqUrdu-Regular.ttf",
                "/usr/share/fonts/truetype/nafees/NafeesNastaleeq.ttf",
                "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                os.path.join(self.base_dir, "urdu_font.ttf"),
            ]
        else:
            families = [
                "Noto Naskh Arabic",
                "Noto Sans Arabic",
                "DejaVu Sans",
            ]
            direct = [
                "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
                "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                os.path.join(self.base_dir, "arabic_font.ttf"),
            ]

        candidates = []
        for fam in families:
            try:
                out = subprocess.check_output(
                    ["fc-match", "-f", "%{file}", fam],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                if out:
                    candidates.append(out)
            except Exception:
                pass

        candidates += direct

        seen = set()
        for path in candidates:
            if path and path not in seen and os.path.exists(path):
                seen.add(path)
                return path

        return self.rtl_font_path

    def _decode_image(self, b64_value):
        if not b64_value:
            return None

        try:
            data = str(b64_value)
            if "," in data:
                data = data.split(",", 1)[1]
            return BytesIO(base64.b64decode(data))
        except Exception:
            return None

    def _summary_table(self, d):
        pred_obj = d.get("prediction", {})
        if not isinstance(pred_obj, dict):
            pred_obj = {}

        patient_name = _safe(d.get("patient_name"), "Not entered")
        patient_id = _safe(d.get("patient_id"))
        patient_age = _safe(d.get("patient_age"), "N/A")
        patient_gender = _safe(d.get("patient_gender"), "N/A")
        patient_contact = _safe(d.get("patient_contact"), "N/A")

        record_id = _safe(d.get("record_id"), "-")
        modality = _safe(d.get("modality"))
        prediction = _safe(
            d.get("prediction_class")
            or d.get("predicted_class")
            or d.get("diagnosis")
            or d.get("ai_diagnosis")
            or pred_obj.get("class")
            or d.get("prediction")
        )

        confidence = d.get("confidence", pred_obj.get("mean_confidence"))
        uncertainty = d.get("uncertainty", pred_obj.get("uncertainty"))
        tier = d.get("routing_tier")
        routing = _safe(d.get("routing_decision"), "N/A")
        gpio = _safe(d.get("gpio_alert_type"), "NONE")
        scan_date = _safe(d.get("scan_date") or d.get("date"), datetime.now().strftime("%b %d, %Y - %H:%M:%S"))

        risk_color = self._tier_color(tier, gpio, prediction)
        risk_label = self._risk_label(tier, routing, prediction)

        risk = Table(
            [[self._p("Clinical Routing Status", self.label), self._p(risk_label, self.value)]],
            colWidths=[45 * mm, 130 * mm],
        )
        risk.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), risk_color),
            ("TEXTCOLOR", (0, 0), (-1, -1), C_WHITE),
            ("BOX", (0, 0), (-1, -1), 0, risk_color),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))

        rows = [
            [
                self._p("Patient Name", self.label), self._p(patient_name, self.value),
                self._p("Patient ID", self.label), self._p(patient_id, self.value),
                self._p("Age / Gender", self.label), self._p("{} / {}".format(patient_age, patient_gender), self.value),
            ],
            [
                self._p("Contact / MRN", self.label), self._p(patient_contact, self.value),
                self._p("Record ID", self.label), self._p(record_id, self.value),
                self._p("Modality", self.label), self._p(modality, self.value),
            ],
            [
                self._p("AI Diagnosis", self.label), self._p(prediction, self.value_red if str(tier) == "3" else self.value_green),
                self._p("Confidence", self.label), self._p(_fmt_percent(confidence), self.value),
                self._p("Uncertainty", self.label), self._p("± " + _fmt_percent(uncertainty), self.value),
            ],
            [
                self._p("Scan Date", self.label), self._p(scan_date, self.value),
                self._p("GPIO Alert", self.label), self._p(gpio, self.value),
                self._p("Routing Tier", self.label), self._p(str(tier), self.value),
            ],
        ]

        table = Table(
            rows,
            colWidths=[24 * mm, 34 * mm, 24 * mm, 34 * mm, 24 * mm, 35 * mm],
            rowHeights=[9 * mm, 9 * mm, 9 * mm, 9 * mm],
        )

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_PANEL),
            ("BOX", (0, 0), (-1, -1), 0.6, C_BORDER),
            ("GRID", (0, 0), (-1, -1), 0.25, C_BORDER),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))

        return [risk, Spacer(1, 4), table]

    def _gradcam_panel(self, d):
        stream = self._decode_image(d.get("gradcam_base64"))

        if stream:
            img = RLImage(stream)
            img._restrictSize(105 * mm, 56 * mm)
            content = [[img]]
        else:
            content = [[self._p("Grad-CAM overlay not available.", self.body)]]

        panel = Table(content, colWidths=[175 * mm], rowHeights=[60 * mm])
        panel.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFD")),
            ("BOX", (0, 0), (-1, -1), 0.6, C_BORDER),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return panel

    def _callout(self, title, text, accent=C_CYAN):
        t = Table(
            [
                [self._p(title, self.callout_title)],
                [self._p(text, self.body_small)],
            ],
            colWidths=[175 * mm],
        )
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFD")),
            ("BOX", (0, 0), (-1, -1), 0.6, C_BORDER),
            ("LINEBEFORE", (0, 0), (0, -1), 2.2, accent),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        return t

    def _rtl_text_image(self, title, text, color, *args, **kwargs):
        """
        Proper Urdu/Arabic PDF block renderer.

        Important:
        - Uses Pillow RAQM RTL rendering when available.
        - Does NOT manually reverse text when RAQM is available.
        - Avoids mixed English/Urdu/Arabic in RTL boxes.
        - Keeps the report compact and professional.
        """
        try:
            from io import BytesIO
            from pathlib import Path
            import subprocess
            from PIL import Image, ImageDraw, ImageFont, features
            from reportlab.platypus import Image as RLImage
            from reportlab.lib.units import mm
        except Exception:
            return self._callout(title, text, color)

        raw = "{} {}".format(title or "", text or "")
        language = kwargs.get("language", "")

        if not language:
            if "Urdu" in raw or "اردو" in raw:
                language = "Urdu"
            elif "Arabic" in raw or "العربية" in raw or "عربي" in raw:
                language = "Arabic"
            else:
                language = "Arabic"

        is_urdu = str(language).lower().startswith("urdu")
        is_clinical = "clinical" in raw.lower() or "interpretation" in raw.lower()

        if is_urdu:
            lang_code = "ur"
            preferred_fonts = [
                "Noto Nastaliq Urdu",
                "Nafees Nastaleeq",
                "Noto Naskh Arabic",
                "DejaVu Sans",
            ]

            if is_clinical:
                block_title = "کلینیکل خلاصہ"
                paras = [
                    "مصنوعی ذہانت کے نظام نے میڈیکل تصویر کا تجزیہ کیا ہے۔ یہ نتیجہ حتمی تشخیص نہیں ہے۔",
                    "اس رپورٹ کو مریض کی علامات، سابقہ طبی تاریخ، اصل تصویر، اور ڈاکٹر یا ریڈیولوجسٹ کی رائے کے ساتھ لازمی دیکھا جائے۔",
                    "حتمی فیصلہ صرف متعلقہ ڈاکٹر، ریڈیولوجسٹ، یا نیورو سرجن کرے گا۔",
                ]
            else:
                block_title = "مریض کے لیے احتیاطی ہدایات"
                paras = [
                    "پانی مناسب مقدار میں پئیں اور متوازن غذا لیں، جب تک ڈاکٹر نے منع نہ کیا ہو۔",
                    "پروٹین، دالیں، دہی، انڈہ، سبزیاں، پھل اور ہلکی غذا حسب برداشت استعمال کریں۔",
                    "خود سے درد کش ادویات، سٹیرائیڈز، اینٹی بایوٹک، جڑی بوٹی، یا سپلیمنٹ شروع نہ کریں۔",
                    "شدید سر درد، دورہ، الٹی، کمزوری، بے ہوشی، نظر یا بولنے میں مسئلہ ہو تو فوراً ایمرجنسی جائیں۔",
                ]
        else:
            lang_code = "ar"
            preferred_fonts = [
                "Noto Naskh Arabic",
                "Noto Sans Arabic",
                "DejaVu Sans",
            ]

            if is_clinical:
                block_title = "الملخص السريري"
                paras = [
                    "قام نظام الذكاء الاصطناعي بتحليل صورة الفحص الطبي. هذه النتيجة ليست تشخيصاً نهائياً.",
                    "يجب مراجعة التقرير مع أعراض المريض وتاريخه الطبي والصورة الأصلية ورأي الطبيب أو اختصاصي الأشعة.",
                    "القرار النهائي يجب أن يكون بواسطة طبيب مؤهل أو اختصاصي أشعة.",
                ]
            else:
                block_title = "إرشادات العناية والسلامة"
                paras = [
                    "حافظ على شرب الماء وتناول غذاء متوازن ما لم يمنع الطبيب ذلك.",
                    "تناول البروتين والخضار والفواكه والوجبات الخفيفة حسب التحمل.",
                    "لا تبدأ مسكنات قوية أو مضادات حيوية أو أعشاباً أو مكملات من نفسك.",
                    "اطلب الرعاية العاجلة عند صداع شديد أو قيء أو تشنجات أو ضعف مفاجئ أو فقدان وعي أو مشاكل في النظر أو الكلام.",
                ]

        def fc_font(family):
            try:
                out = subprocess.check_output(
                    ["fc-match", "-f", "%{file}", family],
                    text=True
                ).strip()
                if out and Path(out).exists():
                    return out
            except Exception:
                pass
            return ""

        font_path = ""
        for fam in preferred_fonts:
            font_path = fc_font(fam)
            if font_path:
                break

        if not font_path:
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

        use_raqm = bool(features.check("raqm"))
        print("[PDF RTL] language={} | RAQM={} | font={}".format(language, use_raqm, font_path))

        try:
            font_title = ImageFont.truetype(font_path, 30 if is_urdu else 29)
            font_body = ImageFont.truetype(font_path, 24 if is_urdu else 23)
        except Exception:
            return self._callout(title, text, color)

        # Fallback only if RAQM is not available.
        reshaper = None
        bidi_display = None
        if not use_raqm:
            try:
                import arabic_reshaper
                from bidi.algorithm import get_display
                reshaper = arabic_reshaper
                bidi_display = get_display
            except Exception:
                pass

        W = 1600
        margin_x = 80
        right_x = W - margin_x
        max_w = W - (2 * margin_x) - 40

        dummy = Image.new("RGB", (W, 200), "#FFFFFF")
        d = ImageDraw.Draw(dummy)

        def visual(s):
            if use_raqm:
                return s
            if reshaper and bidi_display:
                try:
                    return bidi_display(reshaper.reshape(s))
                except Exception:
                    return s
            return s

        def text_width(s, font):
            vs = visual(s)
            try:
                if use_raqm:
                    box = d.textbbox((0, 0), vs, font=font, direction="rtl", language=lang_code)
                else:
                    box = d.textbbox((0, 0), vs, font=font)
                return box[2] - box[0]
            except Exception:
                return d.textlength(vs, font=font)

        def draw_rtl(draw, x, y, s, font, fill):
            vs = visual(s)
            if use_raqm:
                draw.text(
                    (x, y),
                    vs,
                    font=font,
                    fill=fill,
                    anchor="ra",
                    direction="rtl",
                    language=lang_code,
                )
            else:
                draw.text((x, y), vs, font=font, fill=fill, anchor="ra")

        def wrap_para(para, font):
            words = para.split()
            lines = []
            line = ""

            for w in words:
                test = w if not line else line + " " + w
                if text_width(test, font) <= max_w:
                    line = test
                else:
                    if line:
                        lines.append(line)
                    line = w

            if line:
                lines.append(line)

            return lines

        content = []
        content.append(("title", block_title))
        for para in paras:
            content.append(("space", ""))
            for line in wrap_para(para, font_body):
                content.append(("body", line))

        title_h = 48
        body_h = 42 if is_urdu else 38
        space_h = 18

        H = 55
        for kind, _ in content:
            if kind == "title":
                H += title_h
            elif kind == "space":
                H += space_h
            else:
                H += body_h
        H += 45
        H = max(H, 230)

        img = Image.new("RGB", (W, H), "#F5F8FC")
        draw = ImageDraw.Draw(img)

        border = "#D7E0EA"
        ink = "#0B2239"
        accent = "#00C853" if not is_clinical else "#00C7E8"

        draw.rectangle([0, 0, W - 1, H - 1], outline=border, width=3)
        draw.rectangle([0, 0, 16, H], fill=accent)

        y = 42
        for kind, line in content:
            if kind == "space":
                y += space_h
            elif kind == "title":
                draw_rtl(draw, right_x, y, line, font_title, ink)
                y += title_h
            else:
                draw_rtl(draw, right_x, y, line, font_body, ink)
                y += body_h

        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        img_w = 170 * mm
        img_h = (H / W) * img_w
        return RLImage(buf, width=img_w, height=img_h)


    def _rtl_block(self, title, text, accent=C_CYAN, language="Urdu"):
        if accent == C_GREEN:
            rgb = (0, 200, 83)
        elif accent == C_RED:
            rgb = (255, 23, 68)
        else:
            rgb = (0, 191, 234)

        return self._rtl_text_image(title, text, rgb, language=language)

    def _generate(self, patient_data, output_path=None, language=None, **kwargs):
        if language is None:
            language = kwargs.get("lang")
        language = language or patient_data.get("report_language") or patient_data.get("language") or "English"

        if not output_path:
            output_path = patient_data.get("output_path") or patient_data.get("pdf_path") or "MEDAI_Report.pdf"

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        rtl = str(language).lower() in ["urdu", "arabic"]

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=15 * mm,
            leftMargin=15 * mm,
            topMargin=27 * mm,
            bottomMargin=18 * mm,
        )

        story = []

        story.append(self._p("Executive Diagnostic Summary", self.section))
        for item in self._summary_table(patient_data):
            story.append(item)

        story.append(Spacer(1, 6))
        story.append(self._p("Visual Explainability Evidence - Grad-CAM Overlay", self.section))
        story.append(self._gradcam_panel(patient_data))

        clinical_text = (
            patient_data.get("llm_report")
            or patient_data.get("clinical_report")
            or patient_data.get("phi2_report")
            or self._aalyan_best_clinical_interpretation((locals().get("patient_data") or locals().get("report_data") or locals().get("data") or locals()), locals().get("language", "English"))
        )

        story.append(Spacer(1, 6))
        story.append(self._p("AI Clinical Interpretation ({})".format(language), self.section))

        if rtl:
            if str(language).lower().startswith("urdu"):
                clinical_title = "کلینیکل خلاصہ"
            else:
                clinical_title = "الملخص السريري"
            story.append(self._rtl_block(clinical_title, clinical_text, C_CYAN, language=language))
        else:
            story.append(self._callout("Clinical Summary", clinical_text, C_CYAN))

        care_plan = patient_data.get("care_plan")
        if care_plan:
            story.append(Spacer(1, 6))
            story.append(self._p("Patient Care & Safety Guidance", self.section))

            if rtl:
                if str(language).lower().startswith("urdu"):
                    care_title = "غذا، احتیاطی ہدایات اور خطرے کی علامات"
                else:
                    care_title = "إرشادات التغذية والاحتياطات وعلامات الخطر"
                story.append(self._rtl_block(care_title, care_plan, C_GREEN, language=language))
            else:
                care_title = "Supportive Nutrition, Lifestyle, Avoidance, and Red-Flag Guidance"
                story.append(self._callout(care_title, care_plan, C_GREEN))

        disclaimer = (
            "This report is generated by an AI-assisted edge diagnostic system. "
            "It is not a standalone medical diagnosis, not a prescription, and not a replacement for professional clinical judgment. "
            "Final interpretation must be performed by a qualified doctor, radiologist, or relevant specialist using the patient's symptoms, history, examination, and original imaging data."
        )

        story.append(Spacer(1, 6))
        story.append(self._callout("Clinical Safety Disclaimer", disclaimer, C_RED))

        doc.build(story, onFirstPage=self._header_footer, onLaterPages=self._header_footer)
        return output_path

    def generate_report(self, patient_data, output_path=None, language=None, *args, **kwargs):
        patient_data = dict(patient_data or {})
        patient_data["_aalyan_uncertainty_display"] = _aalyan_format_uncertainty(patient_data)
        return self._generate(patient_data, output_path, language, **kwargs)

    def generate_pdf(self, patient_data, output_path=None, language=None, *args, **kwargs):
        return self._generate(patient_data, output_path, language, **kwargs)

    def create_report(self, patient_data, output_path=None, language=None, *args, **kwargs):
        return self._generate(patient_data, output_path, language, **kwargs)

    def build_report(self, patient_data, output_path=None, language=None, *args, **kwargs):
        return self._generate(patient_data, output_path, language, **kwargs)


PDFReportGenerator = ProfessionalPDFReportGenerator
PDFGenerator = ProfessionalPDFReportGenerator
MedicalPDFGenerator = ProfessionalPDFReportGenerator
MedicalReportGenerator = ProfessionalPDFReportGenerator
ReportGenerator = ProfessionalPDFReportGenerator


def generate_report(patient_data, output_path=None, language=None, *args, **kwargs):
    return ProfessionalPDFReportGenerator().generate_report(patient_data, output_path, language, **kwargs)


def generate_pdf(patient_data, output_path=None, language=None, *args, **kwargs):
    return ProfessionalPDFReportGenerator().generate_pdf(patient_data, output_path, language, **kwargs)


def create_report(patient_data, output_path=None, language=None, *args, **kwargs):
    return ProfessionalPDFReportGenerator().create_report(patient_data, output_path, language, **kwargs)

# ============================================================
# AALYAN FINAL PATCH: Correct Urdu/Arabic clinical vs care blocks
# Prevents Patient Care text from appearing inside AI Clinical Interpretation
# ============================================================
def _aalyan_pdf_clean(v):
    s = str(v or "").strip()
    if s.lower() in {"", "none", "null", "n/a", "na", "-", "--", "not entered"}:
        return ""
    return s


def _aalyan_pdf_parse(v):
    import ast
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                out = ast.literal_eval(s)
                if isinstance(out, dict):
                    return out
            except Exception:
                pass
    return v


def _aalyan_pdf_sources(self):
    sources = []

    def walk(x):
        x = _aalyan_pdf_parse(x)
        if isinstance(x, dict):
            sources.append(x)
            for v in x.values():
                if isinstance(v, (dict, list, tuple, str)):
                    walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    payload = getattr(self, "_aalyan_last_pdf_payload", None)
    if isinstance(payload, dict):
        walk(payload.get("args", []))
        walk(payload.get("kwargs", {}))

    for attr in [
        "last_result", "current_result", "result", "patient_data",
        "report_data", "last_report_data", "current_report_data"
    ]:
        if hasattr(self, attr):
            walk(getattr(self, attr))

    return sources


def _aalyan_pdf_get(self, keys):
    sources = _aalyan_pdf_sources(self)

    for src in sources:
        for k in keys:
            if k in src:
                v = _aalyan_pdf_parse(src.get(k))
                if isinstance(v, dict):
                    for kk in keys + ["class", "label", "value"]:
                        got = _aalyan_pdf_clean(v.get(kk))
                        if got:
                            return got
                got = _aalyan_pdf_clean(v)
                if got:
                    return got

    return ""


def _aalyan_pdf_pct(v):
    s = _aalyan_pdf_clean(v)
    if not s:
        return ""
    if "%" in s:
        return s
    try:
        f = float(s)
        if f <= 1:
            f *= 100
        return f"{f:.1f}%"
    except Exception:
        return s


def _aalyan_make_clinical_text(self, language="English"):
    lang = str(language or "English").lower()

    patient_id = _aalyan_pdf_get(self, ["patient_id", "Patient ID", "pid"])
    modality = _aalyan_pdf_get(self, ["modality", "selected_modality", "backend_modality", "scan_modality"])
    pred = _aalyan_pdf_get(self, [
        "predicted_class", "prediction", "ai_prediction", "ai_diagnosis",
        "diagnosis", "class", "pred_name"
    ])
    conf = _aalyan_pdf_pct(_aalyan_pdf_get(self, [
        "confidence", "mean_confidence", "confidence_pct", "conf_pct", "mean_confidence_pct"
    ]))
    unc = _aalyan_pdf_get(self, ["uncertainty", "mc_uncertainty", "variance", "mc_variance"])
    routing = _aalyan_pdf_get(self, ["routing", "routing_tier", "tier", "routing_label"])

    # Extract prediction dict if prediction field came as "{'class': ...}"
    for src in _aalyan_pdf_sources(self):
        for k in ["prediction", "ai_prediction", "result", "results"]:
            v = _aalyan_pdf_parse(src.get(k)) if isinstance(src, dict) else None
            if isinstance(v, dict):
                pred = pred or _aalyan_pdf_clean(v.get("class"))
                conf = conf or _aalyan_pdf_pct(v.get("mean_confidence") or v.get("confidence"))
                unc = unc or _aalyan_pdf_clean(v.get("uncertainty"))
                break

    if unc:
        u = str(unc).replace("±", "").replace("%", "").strip()
        try:
            unc = f"± {float(u):.2f}%"
        except Exception:
            unc = str(unc)

    if not modality:
        modality = "medical imaging"

    if not pred:
        pred = "review-required finding"

    if "urdu" in lang:
        return (
            f"کلینیکل خلاصہ:\n"
            f"مصنوعی ذہانت کے نظام نے مریض {patient_id or ''} کے {modality} اسکین کا تجزیہ کیا۔ "
            f"نتیجہ {pred} ظاہر ہوا"
            + (f"، اعتماد {conf}" if conf else "")
            + (f"، اور غیر یقینی {unc}" if unc else "")
            + (f"۔ کلینیکل روٹنگ: {routing}" if routing else "")
            + "۔\nیہ رپورٹ حتمی تشخیص نہیں؛ ڈاکٹر یا ریڈیالوجسٹ سے تصدیق ضروری ہے۔"
        )

    if "arabic" in lang:
        return (
            f"الملخص السريري:\n"
            f"قام نظام الذكاء الاصطناعي بتحليل فحص {modality} للمريض {patient_id or ''}. "
            f"أظهرت النتيجة {pred}"
            + (f" بدرجة ثقة {conf}" if conf else "")
            + (f" وبدرجة عدم يقين {unc}" if unc else "")
            + (f". قرار التوجيه السريري: {routing}" if routing else "")
            + ".\nهذه النتيجة ليست تشخيصاً نهائياً، ويجب مراجعتها بواسطة طبيب أو اختصاصي أشعة."
        )

    return (
        f"CLINICAL SUMMARY: The AI edge node analyzed the {modality} scan"
        + (f" for patient {patient_id}" if patient_id else "")
        + f" and produced the finding: {pred}."
        + (f" Mean confidence is {conf}." if conf else "")
        + (f" Monte Carlo uncertainty is {unc}." if unc else "")
        + (f" Routing decision: {routing}." if routing else "")
        + " This is a clinical decision-support output and requires verification by a qualified doctor/radiologist."
    )


def _aalyan_make_care_text(language="English"):
    lang = str(language or "English").lower()

    if "urdu" in lang:
        return (
            "مریض کے لیے احتیاطی ہدایات:\n"
            "پانی مناسب مقدار میں پئیں اور متوازن غذا لیں، جب تک ڈاکٹر منع نہ کرے۔\n"
            "پروٹین، دالیں، دہی، انڈہ، سبزیاں، پھل اور ہلکی غذا حسب برداشت استعمال کریں۔\n"
            "خود سے کوئی دوا، سٹیرائڈ، اینٹی بائیوٹک، جڑی بوٹی یا سپلیمنٹ شروع نہ کریں۔\n"
            "شدید سر درد، دورہ، الٹی، کمزوری، بے ہوشی، نظر یا بولنے میں مسئلہ ہو تو فوراً ایمرجنسی جائیں۔"
        )

    if "arabic" in lang:
        return (
            "إرشادات العناية والسلامة:\n"
            "حافظ على شرب الماء وتناول غذاء متوازن ما لم يمنع الطبيب ذلك.\n"
            "تناول البروتين والخضار والفواكه والوجبات الخفيفة حسب التحمل.\n"
            "لا تبدأ مسكنات قوية أو مضادات حيوية أو أعشاباً أو مكملات من نفسك.\n"
            "اطلب الرعاية العاجلة عند صداع شديد أو قيء أو تشنجات أو ضعف مفاجئ أو فقدان وعي أو مشاكل في النظر أو الكلام."
        )

    return (
        "Supportive Nutrition, Lifestyle, Avoidance, and Red-Flag Guidance\n\n"
        "Important Safety Note:\n"
        "This AI report is a clinical decision-support output, not a final diagnosis. "
        "A qualified clinician/radiologist must review the patient, image quality, symptoms, and medical history.\n\n"
        "General Supportive Care:\n"
        "- Maintain hydration and balanced meals unless a doctor has restricted fluids/diet.\n"
        "- Avoid self-starting medicines, antibiotics, steroids, herbal remedies, or supplements.\n"
        "- Seek urgent care if severe symptoms occur."
    )


def _aalyan_is_care_text(s):
    s = str(s or "").lower()
    bad = [
        "patient care", "safety guidance", "supportive nutrition",
        "مریض کے لیے احتیاطی", "احتیاطی ہدایات",
        "إرشادات العناية", "العناية والسلامة"
    ]
    return any(x.lower() in s for x in bad)


def _aalyan_draw_rtl_block(self, title, text, color, *args, **kwargs):
    try:
        from io import BytesIO
        from reportlab.platypus import Image as RLImage
        from reportlab.lib.units import mm
        from PIL import Image, ImageDraw, ImageFont, features
        import os
    except Exception:
        return self._aalyan_original_rtl_text_image(title, text, color, *args, **kwargs)

    title_s = str(title or "")
    lang = str(kwargs.get("language") or "")
    if "Urdu" in title_s:
        lang = "Urdu"
    elif "Arabic" in title_s:
        lang = "Arabic"

    is_care = "patient care" in title_s.lower() or "safety guidance" in title_s.lower()
    is_clinical = "clinical interpretation" in title_s.lower() or "clinical summary" in title_s.lower()

    if is_clinical:
        final_text = _aalyan_make_clinical_text(self, lang)
        color = globals().get("C_CYAN", "#00D9FF")
    elif is_care:
        final_text = _aalyan_make_care_text(lang)
        color = globals().get("C_SUCCESS", globals().get("C_GREEN", "#00C853"))
    else:
        final_text = str(text or "")
        if not final_text.strip() or _aalyan_is_care_text(final_text):
            final_text = _aalyan_make_clinical_text(self, lang)

    if "urdu" in lang.lower():
        font_path = "/usr/share/fonts/truetype/noto/NotoNastaliqUrdu-Bold.ttf"
    else:
        font_path = "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf"

    if not os.path.exists(font_path):
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    W = 1450
    H = 330 if is_clinical else 390
    bg = (245, 249, 253)
    ink = (12, 35, 55)

    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    # border and left colored bar
    draw.rectangle((0, 0, W - 1, H - 1), outline=(210, 222, 235), width=2)
    draw.rectangle((0, 0, 14, H), fill=color)

    try:
        font_title = ImageFont.truetype(font_path, 34 if "urdu" in lang.lower() else 30)
        font_body = ImageFont.truetype(font_path, 28 if "urdu" in lang.lower() else 25)
    except Exception:
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()

    raqm = False
    try:
        raqm = bool(features.check("raqm"))
    except Exception:
        raqm = False

    right_x = W - 45
    y = 45
    max_w = W - 110

    def measure(s, font):
        try:
            return draw.textlength(s, font=font, direction="rtl" if raqm else None)
        except Exception:
            return draw.textlength(s, font=font)

    def wrap_para(para, font):
        words = str(para).split()
        if not words:
            return [""]
        lines = []
        cur = ""
        for w in words:
            test = w if not cur else cur + " " + w
            if measure(test, font) <= max_w:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def draw_rtl(line, x, yy, font):
        try:
            draw.text((x, yy), line, font=font, fill=ink, anchor="ra", direction="rtl" if raqm else None)
        except Exception:
            draw.text((x, yy), line, font=font, fill=ink, anchor="ra")

    paragraphs = [p.strip() for p in str(final_text).split("\n") if p.strip()]

    if paragraphs:
        draw_rtl(paragraphs[0], right_x, y, font_title)
        y += 58
        paragraphs = paragraphs[1:]

    for para in paragraphs:
        for line in wrap_para(para, font_body):
            if y > H - 45:
                break
            draw_rtl(line, right_x, y, font_body)
            y += 42 if "urdu" in lang.lower() else 36
        y += 8

    bio = BytesIO()
    img.save(bio, format="PNG", optimize=True)
    bio.seek(0)

    rl_img = RLImage(bio)
    rl_img.drawWidth = 168 * mm
    rl_img.drawHeight = (H / W) * rl_img.drawWidth
    return rl_img


def _aalyan_patch_pdf_generator_runtime():
    import inspect

    for name, cls in list(globals().items()):
        if not inspect.isclass(cls):
            continue

        if hasattr(cls, "generate_report") and not hasattr(cls, "_aalyan_generate_wrapped"):
            old_generate = cls.generate_report

            def make_generate_wrapper(orig):
                def wrapped(self, *args, **kwargs):
                    self._aalyan_last_pdf_payload = {"args": args, "kwargs": kwargs}
                    return orig(self, *args, **kwargs)
                return wrapped

            cls.generate_report = make_generate_wrapper(old_generate)
            cls._aalyan_generate_wrapped = True

        if hasattr(cls, "_callout") and not hasattr(cls, "_aalyan_callout_wrapped"):
            old_callout = cls._callout

            def make_callout_wrapper(orig):
                def wrapped(self, title, text, color, *args, **kwargs):
                    title_s = str(title or "").lower()
                    text_s = str(text or "")
                    if ("clinical summary" in title_s or "clinical interpretation" in title_s) and (
                        not text_s.strip()
                        or "no clinical interpretation generated" in text_s.lower()
                        or _aalyan_is_care_text(text_s)
                    ):
                        text = _aalyan_make_clinical_text(self, "English")
                    return orig(self, title, text, color, *args, **kwargs)
                return wrapped

            cls._callout = make_callout_wrapper(old_callout)
            cls._aalyan_callout_wrapped = True

        if hasattr(cls, "_rtl_text_image") and not hasattr(cls, "_aalyan_rtl_fixed_final"):
            cls._aalyan_original_rtl_text_image = cls._rtl_text_image
            cls._rtl_text_image = _aalyan_draw_rtl_block
            cls._aalyan_rtl_fixed_final = True

    print("[AALYAN PDF] Urdu/Arabic clinical-care separation patch active.")


_aalyan_patch_pdf_generator_runtime()

# ============================================================
# END AALYAN FINAL PATCH
# ============================================================

# ============================================================
# AALYAN ABSOLUTE FINAL RTL OVERRIDE
# Purpose:
# 1) Urdu/Arabic clinical block must show diagnosis summary only.
# 2) Patient Care block must show care guidance only.
# 3) Remove Latin/English tokens from Urdu/Arabic to prevent square boxes.
# 4) Force correct color: Clinical = cyan, Care = green.
# ============================================================
def _AALYAN_FINAL_clean(v):
    s = str(v or "").strip()
    if s.lower() in {"", "none", "null", "n/a", "na", "-", "--", "not entered"}:
        return ""
    return s


def _AALYAN_FINAL_parse(v):
    import ast
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                out = ast.literal_eval(s)
                if isinstance(out, dict):
                    return out
            except Exception:
                pass
    return v


def _AALYAN_FINAL_sources(self):
    srcs = []

    def walk(x, depth=0):
        if depth > 5:
            return
        x = _AALYAN_FINAL_parse(x)
        if isinstance(x, dict):
            srcs.append(x)
            for v in x.values():
                walk(v, depth + 1)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v, depth + 1)

    payload = getattr(self, "_aalyan_last_pdf_payload", None)
    if isinstance(payload, dict):
        walk(payload.get("args", []))
        walk(payload.get("kwargs", {}))

    for attr in [
        "patient_data", "report_data", "last_report_data", "current_report_data",
        "last_result", "current_result", "result", "data"
    ]:
        if hasattr(self, attr):
            walk(getattr(self, attr))

    return srcs


def _AALYAN_FINAL_get(self, keys):
    for src in _AALYAN_FINAL_sources(self):
        if not isinstance(src, dict):
            continue
        for k in keys:
            if k in src:
                v = _AALYAN_FINAL_parse(src.get(k))
                if isinstance(v, dict):
                    for kk in keys + ["class", "label", "value", "name"]:
                        got = _AALYAN_FINAL_clean(v.get(kk))
                        if got:
                            return got
                got = _AALYAN_FINAL_clean(v)
                if got:
                    return got
    return ""


def _AALYAN_FINAL_pct(v):
    s = _AALYAN_FINAL_clean(v)
    if not s:
        return ""
    s = s.replace("±", "").strip()
    if "%" in s:
        return s
    try:
        f = float(s)
        if f <= 1:
            f *= 100
        return f"{f:.1f}%"
    except Exception:
        return s


def _AALYAN_FINAL_unc(v):
    s = _AALYAN_FINAL_clean(v)
    if not s:
        return ""
    s = s.replace("±", "").replace("%", "").strip()
    try:
        return f"± {float(s):.2f}%"
    except Exception:
        return _AALYAN_FINAL_clean(v)


def _AALYAN_FINAL_pred_info(self):
    pred = _AALYAN_FINAL_get(self, [
        "predicted_class", "prediction", "ai_prediction", "ai_diagnosis",
        "diagnosis", "class", "pred_name"
    ])

    conf = _AALYAN_FINAL_get(self, [
        "confidence", "mean_confidence", "confidence_pct", "conf_pct", "mean_confidence_pct"
    ])

    unc = _AALYAN_FINAL_get(self, [
        "uncertainty", "mc_uncertainty", "variance", "mc_variance"
    ])

    routing = _AALYAN_FINAL_get(self, [
        "routing", "routing_tier", "tier", "routing_label"
    ])

    modality = _AALYAN_FINAL_get(self, [
        "modality", "selected_modality", "backend_modality", "scan_modality"
    ])

    for src in _AALYAN_FINAL_sources(self):
        if not isinstance(src, dict):
            continue
        for k in ["prediction", "ai_prediction", "result", "results"]:
            val = _AALYAN_FINAL_parse(src.get(k))
            if isinstance(val, dict):
                pred = pred or _AALYAN_FINAL_clean(val.get("class"))
                conf = conf or _AALYAN_FINAL_clean(val.get("mean_confidence") or val.get("confidence"))
                unc = unc or _AALYAN_FINAL_clean(val.get("uncertainty") or val.get("mc_uncertainty"))
                break

    return pred, _AALYAN_FINAL_pct(conf), _AALYAN_FINAL_unc(unc), routing, modality


def _AALYAN_FINAL_urdu_word(x):
    x = str(x or "").lower()
    if "glioma" in x:
        return "گلیوما"
    if "meningioma" in x:
        return "میننجیوما"
    if "pituitary" in x:
        return "پچوٹری ٹیومر"
    if "pneumonia" in x:
        return "نمونیا"
    if "normal" in x or "no tumor" in x:
        return "نارمل"
    if "modality" in x or "mismatch" in x:
        return "غلط موڈیلٹی"
    if not x:
        return "مزید جائزہ درکار"
    return "مزید جائزہ درکار"


def _AALYAN_FINAL_arabic_word(x):
    x = str(x or "").lower()
    if "glioma" in x:
        return "ورم دبقي"
    if "meningioma" in x:
        return "ورم سحائي"
    if "pituitary" in x:
        return "ورم في الغدة النخامية"
    if "pneumonia" in x:
        return "التهاب رئوي"
    if "normal" in x or "no tumor" in x:
        return "طبيعي"
    if "modality" in x or "mismatch" in x:
        return "عدم تطابق نوع الفحص"
    if not x:
        return "يحتاج إلى مراجعة"
    return "يحتاج إلى مراجعة"


def _AALYAN_FINAL_urdu_modality(x):
    x = str(x or "").lower()
    if "brain" in x or "mri" in x:
        return "دماغ کے ایم آر آئی"
    if "xray" in x or "x-ray" in x or "chest" in x:
        return "سینے کے ایکسرے"
    if "bone" in x:
        return "ہڈی کے اسکین"
    if "skin" in x:
        return "جلد کے اسکین"
    if "retinal" in x:
        return "آنکھ کے ریٹینا اسکین"
    if "ecg" in x:
        return "ای سی جی"
    return "طبی تصویر"


def _AALYAN_FINAL_arabic_modality(x):
    x = str(x or "").lower()
    if "brain" in x or "mri" in x:
        return "الرنين المغناطيسي للدماغ"
    if "xray" in x or "x-ray" in x or "chest" in x:
        return "الأشعة السينية للصدر"
    if "bone" in x:
        return "فحص العظام"
    if "skin" in x:
        return "فحص الجلد"
    if "retinal" in x:
        return "فحص شبكية العين"
    if "ecg" in x:
        return "تخطيط القلب"
    return "الصورة الطبية"


def _AALYAN_FINAL_clinical_text(self, lang):
    pred, conf, unc, routing, modality = _AALYAN_FINAL_pred_info(self)

    if str(lang).lower().startswith("urdu"):
        pred_u = _AALYAN_FINAL_urdu_word(pred)
        mod_u = _AALYAN_FINAL_urdu_modality(modality)

        parts = [
            "کلینیکل خلاصہ:",
            f"مصنوعی ذہانت کے نظام نے {mod_u} اسکین کا تجزیہ کیا۔",
            f"ابتدائی نتیجہ {pred_u} ظاہر ہوا۔",
        ]

        if conf:
            parts.append(f"اعتماد کی شرح {conf} ہے۔")
        if unc:
            parts.append(f"غیر یقینی {unc} ہے۔")

        parts.append("یہ رپورٹ حتمی تشخیص نہیں ہے؛ ڈاکٹر یا ریڈیالوجسٹ سے تصدیق ضروری ہے۔")
        return "\n".join(parts)

    if str(lang).lower().startswith("arabic"):
        pred_a = _AALYAN_FINAL_arabic_word(pred)
        mod_a = _AALYAN_FINAL_arabic_modality(modality)

        parts = [
            "الملخص السريري:",
            f"قام نظام الذكاء الاصطناعي بتحليل فحص {mod_a}.",
            f"أظهرت النتيجة الأولية: {pred_a}.",
        ]

        if conf:
            parts.append(f"درجة الثقة {conf}.")
        if unc:
            parts.append(f"درجة عدم اليقين {unc}.")

        parts.append("هذه النتيجة ليست تشخيصاً نهائياً، ويجب مراجعتها بواسطة طبيب أو اختصاصي أشعة.")
        return "\n".join(parts)

    pred_e = pred or "review-required finding"
    mod_e = modality or "medical imaging"
    out = f"CLINICAL SUMMARY: The AI edge node analyzed the {mod_e} scan and produced the finding: {pred_e}."
    if conf:
        out += f" Mean confidence is {conf}."
    if unc:
        out += f" Monte Carlo uncertainty is {unc}."
    out += " This is a clinical decision-support output and requires verification by a qualified doctor/radiologist."
    return out


def _AALYAN_FINAL_care_text(lang):
    if str(lang).lower().startswith("urdu"):
        return "\n".join([
            "مریض کے لیے احتیاطی ہدایات:",
            "پانی مناسب مقدار میں پئیں اور متوازن غذا لیں، جب تک ڈاکٹر منع نہ کرے۔",
            "پروٹین، دالیں، دہی، انڈہ، سبزیاں، پھل اور ہلکی غذا حسب برداشت استعمال کریں۔",
            "خود سے کوئی دوا، اینٹی بائیوٹک، سٹیرائڈ، جڑی بوٹی یا سپلیمنٹ شروع نہ کریں۔",
            "شدید سر درد، دورہ، الٹی، کمزوری، بے ہوشی، نظر یا بولنے میں مسئلہ ہو تو فوراً ایمرجنسی جائیں۔",
        ])

    if str(lang).lower().startswith("arabic"):
        return "\n".join([
            "إرشادات العناية والسلامة:",
            "حافظ على شرب الماء وتناول غذاء متوازن ما لم يمنع الطبيب ذلك.",
            "تناول البروتين والخضار والفواكه والوجبات الخفيفة حسب التحمل.",
            "لا تبدأ مسكنات قوية أو مضادات حيوية أو أعشاباً أو مكملات من نفسك.",
            "اطلب الرعاية العاجلة عند صداع شديد أو قيء أو تشنجات أو ضعف مفاجئ أو فقدان وعي أو مشاكل في النظر أو الكلام.",
        ])

    return "\n".join([
        "Supportive Nutrition, Lifestyle, Avoidance, and Red-Flag Guidance",
        "Maintain hydration and balanced meals unless a doctor has restricted fluids/diet.",
        "Avoid self-starting medicines, antibiotics, steroids, herbal remedies, or supplements.",
        "Seek urgent care if severe symptoms occur.",
    ])


def _AALYAN_FINAL_color_to_hex(color):
    try:
        from reportlab.lib.colors import Color
        if hasattr(color, "red") and hasattr(color, "green") and hasattr(color, "blue"):
            return (
                int(color.red * 255),
                int(color.green * 255),
                int(color.blue * 255),
            )
    except Exception:
        pass

    s = str(color or "").lower()
    if "00c853" in s or "green" in s:
        return (0, 200, 83)
    if "00d9ff" in s or "cyan" in s:
        return (0, 217, 255)
    return (0, 217, 255)


def _AALYAN_FINAL_is_care_call(title, color):
    t = str(title or "").lower()
    c = str(color or "").lower()

    if "patient care" in t or "safety guidance" in t:
        return True
    if "احتیاط" in t or "رہنمائی" in t or "العناية" in t or "السلامة" in t:
        return True
    if "green" in c or "00c853" in c:
        return True

    try:
        if hasattr(color, "green") and color.green > 0.60 and color.red < 0.25:
            return True
    except Exception:
        pass

    return False


def _AALYAN_FINAL_is_clinical_call(title, color):
    t = str(title or "").lower()
    if "clinical interpretation" in t or "clinical summary" in t:
        return True
    if "کلینیکل" in t or "الملخص" in t or "السريري" in t:
        return True
    return not _AALYAN_FINAL_is_care_call(title, color)


def _AALYAN_FINAL_lang_from_title(title, kwargs):
    lang = str(kwargs.get("language") or "")
    title_s = str(title or "").lower()

    if "urdu" in title_s or "اردو" in title_s:
        return "Urdu"
    if "arabic" in title_s or "العربية" in title_s or "عربي" in title_s:
        return "Arabic"
    if lang:
        return lang
    return "English"


def _AALYAN_FINAL_render_rtl(self, title, text, color, *args, **kwargs):
    try:
        from io import BytesIO
        from reportlab.platypus import Image as RLImage
        from reportlab.lib.units import mm
        from PIL import Image, ImageDraw, ImageFont, features
        import os
    except Exception:
        return self._AALYAN_FINAL_original_rtl(title, text, color, *args, **kwargs)

    lang = _AALYAN_FINAL_lang_from_title(title, kwargs)

    if _AALYAN_FINAL_is_care_call(title, color):
        final_text = _AALYAN_FINAL_care_text(lang)
        bar_rgb = (0, 200, 83)
    else:
        final_text = _AALYAN_FINAL_clinical_text(self, lang)
        bar_rgb = (0, 217, 255)

    if str(lang).lower().startswith("urdu"):
        font_path = "/usr/share/fonts/truetype/noto/NotoNastaliqUrdu-Bold.ttf"
        title_size = 32
        body_size = 25
        line_gap = 42
    else:
        font_path = "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf"
        title_size = 30
        body_size = 25
        line_gap = 36

    if not os.path.exists(font_path):
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    W = 1500
    H = 300 if not _AALYAN_FINAL_is_care_call(title, color) else 370

    img = Image.new("RGB", (W, H), (245, 249, 253))
    draw = ImageDraw.Draw(img)

    draw.rectangle((0, 0, W - 1, H - 1), outline=(210, 222, 235), width=2)
    draw.rectangle((0, 0, 14, H), fill=bar_rgb)

    try:
        f_title = ImageFont.truetype(font_path, title_size)
        f_body = ImageFont.truetype(font_path, body_size)
    except Exception:
        f_title = ImageFont.load_default()
        f_body = ImageFont.load_default()

    try:
        use_raqm = bool(features.check("raqm"))
    except Exception:
        use_raqm = False

    right_x = W - 48
    y = 42
    max_width = W - 130

    def width(s, font):
        try:
            return draw.textlength(s, font=font, direction="rtl" if use_raqm else None)
        except Exception:
            return draw.textlength(s, font=font)

    def wrap_line(s, font):
        words = str(s).split()
        lines = []
        cur = ""
        for w in words:
            test = w if not cur else cur + " " + w
            if width(test, font) <= max_width:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines or [""]

    def draw_rtl(s, yy, font):
        try:
            draw.text(
                (right_x, yy),
                s,
                font=font,
                fill=(12, 35, 55),
                anchor="ra",
                direction="rtl" if use_raqm else None,
                language="ur" if str(lang).lower().startswith("urdu") else "ar",
            )
        except TypeError:
            try:
                draw.text(
                    (right_x, yy),
                    s,
                    font=font,
                    fill=(12, 35, 55),
                    anchor="ra",
                    direction="rtl" if use_raqm else None,
                )
            except Exception:
                draw.text((right_x, yy), s, font=font, fill=(12, 35, 55), anchor="ra")
        except Exception:
            draw.text((right_x, yy), s, font=font, fill=(12, 35, 55), anchor="ra")

    paras = [p.strip() for p in str(final_text).splitlines() if p.strip()]

    if paras:
        draw_rtl(paras[0], y, f_title)
        y += 58

    for para in paras[1:]:
        for ln in wrap_line(para, f_body):
            if y > H - 44:
                break
            draw_rtl(ln, y, f_body)
            y += line_gap
        y += 4

    bio = BytesIO()
    img.save(bio, format="PNG", optimize=True)
    bio.seek(0)

    rl_img = RLImage(bio)
    rl_img.drawWidth = 168 * mm
    rl_img.drawHeight = (H / W) * rl_img.drawWidth
    return rl_img


def _AALYAN_FINAL_patch_all_pdf_classes():
    import inspect

    patched = 0

    for _name, _cls in list(globals().items()):
        if not inspect.isclass(_cls):
            continue

        if hasattr(_cls, "generate_report"):
            old_generate = _cls.generate_report

            def make_gen(orig):
                def wrapped(self, *args, **kwargs):
                    self._aalyan_last_pdf_payload = {"args": args, "kwargs": kwargs}
                    return orig(self, *args, **kwargs)
                return wrapped

            if not getattr(_cls, "_AALYAN_FINAL_generate_wrapped", False):
                _cls.generate_report = make_gen(old_generate)
                _cls._AALYAN_FINAL_generate_wrapped = True

        if hasattr(_cls, "_rtl_text_image"):
            if not hasattr(_cls, "_AALYAN_FINAL_original_rtl"):
                _cls._AALYAN_FINAL_original_rtl = _cls._rtl_text_image
            _cls._rtl_text_image = _AALYAN_FINAL_render_rtl
            patched += 1

    print(f"[AALYAN FINAL PDF] RTL clinical/care override active on {patched} class(es).")


_AALYAN_FINAL_patch_all_pdf_classes()

# ============================================================
# END AALYAN ABSOLUTE FINAL RTL OVERRIDE
# ============================================================

# ============================================================
# AALYAN ULTIMATE RTL FINAL PATCH
# Fixes:
# - Urdu/Arabic clinical section must NOT copy patient-care text
# - Patient-care section must NOT copy clinical text
# - Removes ± and % symbols that create square boxes
# - Uses call order + title + language detection for reliable separation
# ============================================================

def _AALYAN_ULT_clean(v):
    s = str(v or "").strip()
    if s.lower() in {"", "none", "null", "n/a", "na", "-", "--", "not entered"}:
        return ""
    return s


def _AALYAN_ULT_parse(v):
    import ast
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                out = ast.literal_eval(s)
                if isinstance(out, dict):
                    return out
            except Exception:
                pass
    return v


def _AALYAN_ULT_has_urdu(s):
    s = str(s or "")
    urdu_chars = "ٹڈڑگںھہءیےژچپ"
    return any(ch in s for ch in urdu_chars)


def _AALYAN_ULT_has_arabic(s):
    s = str(s or "")
    return any("\u0600" <= ch <= "\u06ff" for ch in s)


def _AALYAN_ULT_lang(self, title="", text="", kwargs=None):
    kwargs = kwargs or {}
    t = str(title or "").lower()
    lang = str(kwargs.get("language") or "").lower()

    if "urdu" in t or "urdu" in lang or "اردو" in str(title):
        out = "Urdu"
    elif "arabic" in t or "arabic" in lang or "العربية" in str(title) or "عربي" in str(title):
        out = "Arabic"
    elif _AALYAN_ULT_has_urdu(text):
        out = "Urdu"
    elif _AALYAN_ULT_has_arabic(text):
        out = "Arabic"
    else:
        out = getattr(self, "_AALYAN_ULT_last_lang", "English")

    if out in {"Urdu", "Arabic"}:
        try:
            self._AALYAN_ULT_last_lang = out
        except Exception:
            pass

    return out


def _AALYAN_ULT_sources(self):
    srcs = []

    def walk(x, depth=0):
        if depth > 5:
            return
        x = _AALYAN_ULT_parse(x)
        if isinstance(x, dict):
            srcs.append(x)
            for v in x.values():
                walk(v, depth + 1)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v, depth + 1)

    payload = getattr(self, "_aalyan_last_pdf_payload", None)
    if isinstance(payload, dict):
        walk(payload.get("args", []))
        walk(payload.get("kwargs", {}))

    payload2 = getattr(self, "_AALYAN_ULT_payload", None)
    if isinstance(payload2, dict):
        walk(payload2.get("args", []))
        walk(payload2.get("kwargs", {}))

    for attr in [
        "patient_data", "report_data", "last_report_data", "current_report_data",
        "last_result", "current_result", "result", "data"
    ]:
        if hasattr(self, attr):
            walk(getattr(self, attr))

    return srcs


def _AALYAN_ULT_get(self, keys):
    for src in _AALYAN_ULT_sources(self):
        if not isinstance(src, dict):
            continue
        for k in keys:
            if k in src:
                v = _AALYAN_ULT_parse(src.get(k))
                if isinstance(v, dict):
                    for kk in keys + ["class", "label", "value", "name"]:
                        got = _AALYAN_ULT_clean(v.get(kk))
                        if got:
                            return got
                got = _AALYAN_ULT_clean(v)
                if got:
                    return got
    return ""


def _AALYAN_ULT_number(v, decimals=1):
    s = _AALYAN_ULT_clean(v)
    if not s:
        return ""
    s = s.replace("±", "").replace("%", "").strip()
    try:
        f = float(s)
        if f <= 1 and decimals == 1:
            f *= 100
        return f"{f:.{decimals}f}"
    except Exception:
        return s


def _AALYAN_ULT_prediction_data(self):
    pred = _AALYAN_ULT_get(self, [
        "predicted_class", "prediction", "ai_prediction", "ai_diagnosis",
        "diagnosis", "class", "pred_name"
    ])
    conf = _AALYAN_ULT_get(self, [
        "confidence", "mean_confidence", "confidence_pct", "conf_pct", "mean_confidence_pct"
    ])
    unc = _AALYAN_ULT_get(self, [
        "uncertainty", "mc_uncertainty", "variance", "mc_variance"
    ])
    modality = _AALYAN_ULT_get(self, [
        "modality", "selected_modality", "backend_modality", "scan_modality"
    ])

    for src in _AALYAN_ULT_sources(self):
        if not isinstance(src, dict):
            continue
        for k in ["prediction", "ai_prediction", "result", "results"]:
            val = _AALYAN_ULT_parse(src.get(k))
            if isinstance(val, dict):
                pred = pred or _AALYAN_ULT_clean(val.get("class"))
                conf = conf or _AALYAN_ULT_clean(val.get("mean_confidence") or val.get("confidence"))
                unc = unc or _AALYAN_ULT_clean(val.get("uncertainty") or val.get("mc_uncertainty"))
                break

    return pred, _AALYAN_ULT_number(conf, 1), _AALYAN_ULT_number(unc, 2), modality


def _AALYAN_ULT_urdu_pred(x):
    x = str(x or "").lower()
    if "glioma" in x:
        return "گلیوما"
    if "meningioma" in x:
        return "میننجیوما"
    if "pituitary" in x:
        return "پچوٹری ٹیومر"
    if "pneumonia" in x:
        return "نمونیا"
    if "normal" in x or "no tumor" in x:
        return "نارمل"
    if "modality" in x or "mismatch" in x:
        return "غلط موڈیلٹی"
    return "مزید جائزہ درکار"


def _AALYAN_ULT_arabic_pred(x):
    x = str(x or "").lower()
    if "glioma" in x:
        return "ورم دبقي"
    if "meningioma" in x:
        return "ورم سحائي"
    if "pituitary" in x:
        return "ورم في الغدة النخامية"
    if "pneumonia" in x:
        return "التهاب رئوي"
    if "normal" in x or "no tumor" in x:
        return "طبيعي"
    if "modality" in x or "mismatch" in x:
        return "عدم تطابق نوع الفحص"
    return "يحتاج إلى مراجعة"


def _AALYAN_ULT_urdu_modality(x):
    x = str(x or "").lower()
    if "brain" in x or "mri" in x:
        return "دماغ کے ایم آر آئی"
    if "xray" in x or "x-ray" in x or "chest" in x:
        return "سینے کے ایکسرے"
    if "bone" in x:
        return "ہڈی کے اسکین"
    if "skin" in x:
        return "جلد کے اسکین"
    if "retinal" in x:
        return "آنکھ کے ریٹینا اسکین"
    if "ecg" in x:
        return "ای سی جی"
    return "طبی تصویر"


def _AALYAN_ULT_arabic_modality(x):
    x = str(x or "").lower()
    if "brain" in x or "mri" in x:
        return "الرنين المغناطيسي للدماغ"
    if "xray" in x or "x-ray" in x or "chest" in x:
        return "الأشعة السينية للصدر"
    if "bone" in x:
        return "فحص العظام"
    if "skin" in x:
        return "فحص الجلد"
    if "retinal" in x:
        return "فحص شبكية العين"
    if "ecg" in x:
        return "تخطيط القلب"
    return "الصورة الطبية"


def _AALYAN_ULT_clinical_text(self, lang):
    pred, conf, unc, modality = _AALYAN_ULT_prediction_data(self)

    if lang == "Urdu":
        return "\n".join([
            "کلینیکل خلاصہ:",
            f"مصنوعی ذہانت کے نظام نے {_AALYAN_ULT_urdu_modality(modality)} اسکین کا تجزیہ کیا۔",
            f"ابتدائی نتیجہ {_AALYAN_ULT_urdu_pred(pred)} ظاہر ہوا۔",
            f"اعتماد کی شرح {conf} فیصد ہے۔" if conf else "اعتماد کی شرح دستیاب نہیں۔",
            f"غیر یقینی تقریباً {unc} فیصد ہے۔" if unc else "غیر یقینی دستیاب نہیں۔",
            "یہ رپورٹ حتمی تشخیص نہیں ہے؛ ڈاکٹر یا ریڈیالوجسٹ سے تصدیق ضروری ہے۔",
        ])

    if lang == "Arabic":
        return "\n".join([
            "الملخص السريري:",
            f"قام نظام الذكاء الاصطناعي بتحليل فحص {_AALYAN_ULT_arabic_modality(modality)}.",
            f"أظهرت النتيجة الأولية: {_AALYAN_ULT_arabic_pred(pred)}.",
            f"درجة الثقة {conf} بالمئة." if conf else "درجة الثقة غير متاحة.",
            f"درجة عدم اليقين تقريباً {unc} بالمئة." if unc else "درجة عدم اليقين غير متاحة.",
            "هذه النتيجة ليست تشخيصاً نهائياً، ويجب مراجعتها بواسطة طبيب أو اختصاصي أشعة.",
        ])

    pred_e = pred or "review-required finding"
    mod_e = modality or "medical imaging"
    return (
        f"CLINICAL SUMMARY: The AI edge node analyzed the {mod_e} scan and produced the finding: {pred_e}. "
        f"Mean confidence is {conf}%. Monte Carlo uncertainty is approximately {unc}%. "
        "This is a clinical decision-support output and requires verification by a qualified doctor/radiologist."
    )


def _AALYAN_ULT_care_text(lang):
    if lang == "Urdu":
        return "\n".join([
            "مریض کے لیے احتیاطی ہدایات:",
            "پانی مناسب مقدار میں پئیں اور متوازن غذا لیں، جب تک ڈاکٹر منع نہ کرے۔",
            "پروٹین، دالیں، دہی، انڈہ، سبزیاں، پھل اور ہلکی غذا حسب برداشت استعمال کریں۔",
            "خود سے کوئی دوا، اینٹی بائیوٹک، سٹیرائڈ، جڑی بوٹی یا سپلیمنٹ شروع نہ کریں۔",
            "شدید سر درد، دورہ، الٹی، کمزوری، بے ہوشی، نظر یا بولنے میں مسئلہ ہو تو فوراً ایمرجنسی جائیں۔",
        ])

    if lang == "Arabic":
        return "\n".join([
            "إرشادات العناية والسلامة:",
            "حافظ على شرب الماء وتناول غذاء متوازن ما لم يمنع الطبيب ذلك.",
            "تناول البروتين والخضار والفواكه والوجبات الخفيفة حسب التحمل.",
            "لا تبدأ مسكنات قوية أو مضادات حيوية أو أعشاباً أو مكملات من نفسك.",
            "اطلب الرعاية العاجلة عند صداع شديد أو قيء أو تشنجات أو ضعف مفاجئ أو فقدان وعي أو مشاكل في النظر أو الكلام.",
        ])

    return "\n".join([
        "Supportive Nutrition, Lifestyle, Avoidance, and Red-Flag Guidance",
        "Maintain hydration and balanced meals unless a doctor has restricted fluids/diet.",
        "Avoid self-starting medicines, antibiotics, steroids, herbal remedies, or supplements.",
        "Seek urgent care if severe symptoms occur.",
    ])


def _AALYAN_ULT_is_care_title(title):
    t = str(title or "").lower()
    return (
        "patient care" in t
        or "safety guidance" in t
        or "care" in t
        or "احتیاط" in t
        or "العناية" in t
        or "السلامة" in t
    )


def _AALYAN_ULT_choose_block_type(self, title, color, lang):
    """
    Reliable block separation:
    - If title says care, care.
    - Otherwise first RTL block per language = clinical, second = care.
    """
    if _AALYAN_ULT_is_care_title(title):
        return "care"

    if not hasattr(self, "_AALYAN_ULT_rtl_count"):
        self._AALYAN_ULT_rtl_count = {"Urdu": 0, "Arabic": 0, "English": 0}

    self._AALYAN_ULT_rtl_count[lang] = self._AALYAN_ULT_rtl_count.get(lang, 0) + 1
    return "clinical" if self._AALYAN_ULT_rtl_count[lang] == 1 else "care"


def _AALYAN_ULT_render(self, title, text, color, *args, **kwargs):
    try:
        from io import BytesIO
        from reportlab.platypus import Image as RLImage
        from reportlab.lib.units import mm
        from PIL import Image, ImageDraw, ImageFont, features
        import os
    except Exception:
        return self._AALYAN_ULT_original_rtl(title, text, color, *args, **kwargs)

    lang = _AALYAN_ULT_lang(self, title, text, kwargs)
    block_type = _AALYAN_ULT_choose_block_type(self, title, color, lang)

    if block_type == "care":
        final_text = _AALYAN_ULT_care_text(lang)
        bar_rgb = (0, 200, 83)
        height = 380
    else:
        final_text = _AALYAN_ULT_clinical_text(self, lang)
        bar_rgb = (0, 217, 255)
        height = 360

    if lang == "Urdu":
        font_path = "/usr/share/fonts/truetype/noto/NotoNastaliqUrdu-Bold.ttf"
        title_size = 32
        body_size = 25
        line_gap = 43
    else:
        font_path = "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf"
        title_size = 30
        body_size = 25
        line_gap = 37

    if not os.path.exists(font_path):
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    W = 1500
    H = height
    bg = (245, 249, 253)
    ink = (12, 35, 55)

    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    draw.rectangle((0, 0, W - 1, H - 1), outline=(210, 222, 235), width=2)
    draw.rectangle((0, 0, 14, H), fill=bar_rgb)

    try:
        font_title = ImageFont.truetype(font_path, title_size)
        font_body = ImageFont.truetype(font_path, body_size)
    except Exception:
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()

    try:
        use_raqm = bool(features.check("raqm"))
    except Exception:
        use_raqm = False

    right_x = W - 48
    y = 42
    max_w = W - 130

    def measure(s, font):
        try:
            return draw.textlength(s, font=font, direction="rtl" if use_raqm else None)
        except Exception:
            return draw.textlength(s, font=font)

    def wrap_para(para, font):
        words = str(para).split()
        lines = []
        cur = ""
        for w in words:
            test = w if not cur else cur + " " + w
            if measure(test, font) <= max_w:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines or [""]

    def draw_rtl(s, yy, font):
        try:
            draw.text(
                (right_x, yy),
                s,
                font=font,
                fill=ink,
                anchor="ra",
                direction="rtl" if use_raqm else None,
                language="ur" if lang == "Urdu" else "ar",
            )
        except TypeError:
            try:
                draw.text(
                    (right_x, yy),
                    s,
                    font=font,
                    fill=ink,
                    anchor="ra",
                    direction="rtl" if use_raqm else None,
                )
            except Exception:
                draw.text((right_x, yy), s, font=font, fill=ink, anchor="ra")
        except Exception:
            draw.text((right_x, yy), s, font=font, fill=ink, anchor="ra")

    paragraphs = [p.strip() for p in str(final_text).splitlines() if p.strip()]

    if paragraphs:
        draw_rtl(paragraphs[0], y, font_title)
        y += 58

    for para in paragraphs[1:]:
        for line in wrap_para(para, font_body):
            if y > H - 45:
                break
            draw_rtl(line, y, font_body)
            y += line_gap
        y += 3

    bio = BytesIO()
    img.save(bio, format="PNG", optimize=True)
    bio.seek(0)

    rl_img = RLImage(bio)
    rl_img.drawWidth = 168 * mm
    rl_img.drawHeight = (H / W) * rl_img.drawWidth
    return rl_img


def _AALYAN_ULT_patch_pdf_classes():
    import inspect

    patched = 0

    for _name, _cls in list(globals().items()):
        if not inspect.isclass(_cls):
            continue

        if hasattr(_cls, "generate_report") and not getattr(_cls, "_AALYAN_ULT_generate_wrapped", False):
            old_generate = _cls.generate_report

            def make_gen(orig):
                def wrapped(self, *args, **kwargs):
                    self._AALYAN_ULT_payload = {"args": args, "kwargs": kwargs}
                    self._aalyan_last_pdf_payload = {"args": args, "kwargs": kwargs}
                    self._AALYAN_ULT_rtl_count = {"Urdu": 0, "Arabic": 0, "English": 0}
                    self._AALYAN_ULT_last_lang = "English"
                    return orig(self, *args, **kwargs)
                return wrapped

            _cls.generate_report = make_gen(old_generate)
            _cls._AALYAN_ULT_generate_wrapped = True

        if hasattr(_cls, "_rtl_text_image"):
            if not hasattr(_cls, "_AALYAN_ULT_original_rtl"):
                _cls._AALYAN_ULT_original_rtl = _cls._rtl_text_image

            _cls._rtl_text_image = _AALYAN_ULT_render
            _cls._AALYAN_ULT_rtl_overridden = True
            patched += 1

    print(f"[AALYAN ULTIMATE PDF] Clean Urdu/Arabic RTL override active on {patched} class(es).")


_AALYAN_ULT_patch_pdf_classes()

# ============================================================
# END AALYAN ULTIMATE RTL FINAL PATCH
# ============================================================
