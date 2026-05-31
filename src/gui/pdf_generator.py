import os
import base64
import tempfile
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus import Image as RLImage
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# RTL Text Parsing Libraries
import arabic_reshaper
from bidi.algorithm import get_display

class MedicalReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        
        # Register a single, stable font for all RTL text
        self.has_rtl_font = False
        font_path = os.path.join(os.path.dirname(__file__), 'arabic_font.ttf')
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont('RTLFont', font_path))
            self.has_rtl_font = True

        # Corporate Medical Colors
        self.c_primary = colors.HexColor("#0A2540")   
        self.c_accent = colors.HexColor("#00D4FF")    
        self.c_danger = colors.HexColor("#D32F2F")    
        self.c_bg = colors.HexColor("#F8FAFC")        

        # Base Styles
        self.header_style = ParagraphStyle(
            'HeaderStyle', parent=self.styles['Heading1'], fontName='Helvetica-Bold',
            fontSize=22, textColor=self.c_primary, spaceAfter=20
        )
        self.sub_header = ParagraphStyle(
            'SubHeader', parent=self.styles['Heading2'], fontName='Helvetica-Bold',
            fontSize=14, textColor=self.c_primary, spaceBefore=15, spaceAfter=10
        )

    def custom_rtl_wrap(self, text, lang, font_name, font_size, max_width):
        """
        Calculates pixel width of words to manually wrap text BEFORE applying BIDI.
        This completely bypasses ReportLab's broken RTL multi-line wrapping.
        """
        # 1. Reshape the text to connect letters (Logical Order)
        reshaper = arabic_reshaper.ArabicReshaper(
            configuration={
                'language': 'Urdu' if "Urdu" in lang else 'Arabic',
                'delete_harakat': True
            }
        )
        reshaped_text = reshaper.reshape(text)
        
        final_bidi_lines = []
        paragraphs = reshaped_text.split('\n')
        
        for para in paragraphs:
            words = para.split(' ')
            current_line = []
            
            for word in words:
                # Test adding the next word to the current line
                test_line = ' '.join(current_line + [word]) if current_line else word
                w = pdfmetrics.stringWidth(test_line, font_name, font_size)
                
                # If it fits the box, keep it on this line
                if w <= max_width:
                    current_line.append(word)
                else:
                    # Line is full. Apply BIDI flip to this specific line and save it.
                    if current_line:
                        line_str = ' '.join(current_line)
                        bidi_line = get_display(line_str)
                        clean = bidi_line.replace('\u202a', '').replace('\u202b', '').replace('\u202c', '').replace('\u200e', '').replace('\u200f', '')
                        final_bidi_lines.append(clean)
                    current_line = [word] # Start next line
                    
            # Catch any remaining words at the end of the paragraph
            if current_line:
                line_str = ' '.join(current_line)
                bidi_line = get_display(line_str)
                clean = bidi_line.replace('\u202a', '').replace('\u202b', '').replace('\u202c', '').replace('\u200e', '').replace('\u200f', '')
                final_bidi_lines.append(clean)
                
        return final_bidi_lines

    def generate_report(self, patient_data, output_path, lang="English"):
        doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        story = []

        # --- HEADER SECTION ---
        story.append(Paragraph("<b>MEDAI</b> | Edge Server v2.0", self.header_style))
        story.append(Paragraph("<font size=10 color=grey>MULTI-MODAL DIAGNOSTIC AI REPORT</font>", self.styles['Normal']))
        story.append(Spacer(1, 15))
        story.append(HRFlowable(width="100%", thickness=2, color=self.c_primary))
        story.append(Spacer(1, 20))

        # --- METADATA ---
        date_str = datetime.now().strftime("%B %d, %Y - %H:%M:%S")
        meta_data = [
            [Paragraph("<b>Patient ID:</b>"), patient_data.get("patient_id", "UNKNOWN"), Paragraph("<b>Scan Date:</b>"), date_str],
            [Paragraph("<b>Modality:</b>"), patient_data.get("modality", "N/A"), Paragraph("<b>MC Confidence:</b>"), f"{patient_data.get('confidence', 0)}%"],
            [Paragraph("<b>AI Diagnosis:</b>"), patient_data.get("prediction", "N/A"), Paragraph("<b>Routing Tier:</b>"), patient_data.get("routing_tier", "N/A")]
        ]
        
        t = Table(meta_data, colWidths=[90, 150, 100, 170])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), self.c_bg),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
            ('BOX', (0,0), (-1,-1), 1, self.c_primary),
            ('PADDING', (0,0), (-1,-1), 8),
            ('TEXTCOLOR', (1,2), (1,2), self.c_danger)
        ]))
        story.append(t)
        story.append(Spacer(1, 25))

        # --- VISUAL EXPLAINABILITY ---
        story.append(Paragraph("Visual Explainability (Grad-CAM Overlay)", self.sub_header))

        gradcam_b64 = patient_data.get("gradcam_base64")
        gradcam_path = patient_data.get("gradcam_path")

        if gradcam_b64:
            try:
                raw = base64.b64decode(gradcam_b64)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tmp.write(raw)
                tmp.close()
                gradcam_path = tmp.name
            except Exception:
                gradcam_path = None

        if gradcam_path and os.path.exists(gradcam_path):
            try:
                img = RLImage(gradcam_path, width=360, height=270)
                img_table = Table([[img]], colWidths=[530], rowHeights=[285])
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke),
                    ('BOX', (0,0), (-1,-1), 1, colors.grey),
                ]))
                story.append(img_table)
            except Exception:
                placeholder = Table([[Paragraph("<b>[ GRAD-CAM IMAGE COULD NOT BE EMBEDDED ]</b>", self.styles['Normal'])]], colWidths=[530], rowHeights=[180])
                placeholder.setStyle(TableStyle([
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke),
                    ('BOX', (0,0), (-1,-1), 1, colors.grey),
                ]))
                story.append(placeholder)
        else:
            placeholder = Table([[Paragraph("<b>[ NO GRAD-CAM AVAILABLE: RUN LIVE ANALYSIS FIRST ]</b>", self.styles['Normal'])]], colWidths=[530], rowHeights=[180])
            placeholder.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke),
                ('BOX', (0,0), (-1,-1), 1, colors.grey),
            ]))
            story.append(placeholder)

        story.append(Spacer(1, 25))

        # --- DYNAMIC FONT & ALIGNMENT ---
        current_font = 'RTLFont' if (self.has_rtl_font and ("Urdu" in lang or "Arabic" in lang)) else 'Helvetica'
        text_alignment = 2 if current_font == 'RTLFont' else 0 
        
        llm_style = ParagraphStyle(
            'LLMStyle', parent=self.styles['Normal'], fontName=current_font,
            fontSize=14, leading=26, alignment=text_alignment
        )

        # --- CLINICAL INTERPRETATION ---
        display_lang = lang.split(' ')[0] 
        story.append(Paragraph(f"Phi-2 Clinical Interpretation ({display_lang})", self.sub_header))
        
        raw_text = patient_data.get("llm_text", "No clinical text generated.")
        
        if "Urdu" in lang or "Arabic" in lang:
            # Generate the manually wrapped lines based on a 480px width box
            bidi_lines = self.custom_rtl_wrap(raw_text, lang, current_font, 14, max_width=480)
            
            # Join the lines with a PDF line-break so ReportLab treats it as one solid block
            final_text = "<br/>".join(bidi_lines)
            text_paragraph = Paragraph(final_text, llm_style)
        else:
            text_paragraph = Paragraph(raw_text, llm_style)

        # Wrap everything in a single clean border box
        text_box = Table([[text_paragraph]], colWidths=[530])
        text_box.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), self.c_bg),
            ('BOX', (0,0), (-1,-1), 1, colors.lightgrey),
            ('PADDING', (0,0), (-1,-1), 12),
            ('VALIGN', (0,0), (-1,-1), 'TOP')
        ]))
        
        story.append(text_box)

        # --- PATIENT CARE PACKAGE (EVIDENCE-BASED GUIDELINES) ---
        care_package = patient_data.get("care_package")
        if care_package:
            story.append(Spacer(1, 25))
            story.append(Paragraph("Patient Care Package (Evidence-Based Clinical Support)", self.sub_header))
            
            care_data = [
                [Paragraph("<b>Therapeutic Nutrition:</b>", self.styles['Normal']), Paragraph(care_package.get("nutrition", "N/A"), self.styles['Normal'])],
                [Paragraph("<b>Precautionary Measures:</b>", self.styles['Normal']), Paragraph(care_package.get("precautions", "N/A"), self.styles['Normal'])],
                [Paragraph("<b>Clinical Reference:</b>", self.styles['Normal']), Paragraph(care_package.get("reference", "WHO/NIH Guidelines"), self.styles['Normal'])]
            ]
            
            # colWidths structured to equal exactly 530 (130 + 400) to align with other boxes
            care_table = Table(care_data, colWidths=[130, 400])
            care_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke), # Light background for the header column
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('GRID', (0,0), (-1,-1), 1, colors.lightgrey),
                ('PADDING', (0,0), (-1,-1), 10),
            ]))
            story.append(care_table)

        # --- FOOTER ---
        story.append(Spacer(1, 40))

        story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
        story.append(Spacer(1, 5))
        story.append(Paragraph("<font size=8 color=grey><i>Generated autonomously by MEDAI Edge Server. This report is AI-assisted and must be reviewed by a certified radiologist.</i></font>", self.styles['Normal']))

        doc.build(story)
        return output_path