import sys
import time
import random
import datetime
import requests
import sqlite3
import os
import subprocess
import numpy as np
from pdf_generator import MedicalReportGenerator # ADD THIS

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, 
                             QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, 
                             QPushButton, QComboBox, QTableWidget, QTableWidgetItem, 
                             QTextEdit, QFrame, QHeaderView, QProgressBar, QScrollArea, 
                             QSlider, QRadioButton, QLineEdit, QFileDialog, QMessageBox,
                             QTabBar, QSplitter, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QRect, QRectF, QPoint, QSize
from PyQt5.QtGui import (QFont, QFontDatabase, QColor, QPainter, QPen, QBrush, 
                         QPainterPath, QLinearGradient)
import pyqtgraph as pg

# Dummy import for Telegraf parsing (ensure hardware_monitor.py is in same dir)
try:
    from hardware_monitor import TegrastatsParser
except ImportError:
    class TegrastatsParser:
        def parse_line(self, line): return {'cpu_percent':0, 'gpu_percent':0, 'temp_c':0, 'power_w':0, 'error':None}

# =====================================================================
# THEME & COLORS (COMMAND CENTER AESTHETIC)
# =====================================================================
C_BG_PRIMARY = "#0A0E17"
C_BG_SECONDARY = "#0F1520"
C_CARD_BG = "#141B26"
C_BORDER = "#1E2D40"
C_ACCENT = "#00D4FF"
C_SUCCESS = "#00E676"
C_WARNING = "#FFB300"
C_CRITICAL = "#FF1744"
C_TEXT_PRI = "#E8F4FD"
C_TEXT_SEC = "#5A7A9A"
C_PURPLE = "#7C4DFF"

GLOBAL_STYLE = f"""
    QMainWindow, QWidget {{ background-color: {C_BG_PRIMARY}; color: {C_TEXT_PRI}; font-family: 'Segoe UI', sans-serif; }}
    QScrollArea {{ border: none; background-color: transparent; }}
    QScrollArea > QWidget > QWidget {{ background-color: transparent; }}
    QTabWidget::pane {{ border: none; border-top: 2px solid {C_ACCENT}; background: {C_BG_PRIMARY}; }}
    
    QFrame.Card {{ background-color: {C_CARD_BG}; border: 1px solid {C_BORDER}; border-radius: 6px; }}
    
    QPushButton {{ background-color: {C_BG_SECONDARY}; border: 1px solid {C_BORDER}; border-radius: 4px; padding: 8px; color: {C_TEXT_PRI}; font-weight: bold; }}
    QPushButton:hover {{ background-color: {C_BORDER}; }}
    QPushButton.Primary {{ background-color: {C_ACCENT}; border: none; color: black; font-weight: bold; }}
    QPushButton.Primary:hover {{ background-color: #00B3D9; }}
    QPushButton.Success {{ background-color: {C_SUCCESS}; border: none; color: black; font-weight: bold; }}
    QPushButton.Warning {{ background-color: {C_WARNING}; border: none; color: black; font-weight: bold; }}
    
    QTableWidget {{ background-color: {C_BG_SECONDARY}; alternate-background-color: {C_CARD_BG}; color: {C_TEXT_PRI}; border: 1px solid {C_BORDER}; gridline-color: {C_BORDER}; border-radius: 6px; }}
    QHeaderView::section {{ background-color: {C_CARD_BG}; color: {C_TEXT_SEC}; padding: 6px; border: 1px solid {C_BORDER}; font-weight: bold; }}
    
    QTextEdit, QLineEdit, QComboBox {{ background-color: {C_BG_SECONDARY}; color: {C_TEXT_PRI}; border: 1px solid {C_BORDER}; border-radius: 4px; padding: 6px; }}
    
    QScrollBar:vertical {{ background: {C_BG_PRIMARY}; width: 12px; margin: 0px; }}
    QScrollBar::handle:vertical {{ background: {C_BORDER}; min-height: 20px; border-radius: 6px; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
"""

# =====================================================================
# CUSTOM CORE UI COMPONENTS
# =====================================================================
class PillTabBar(QTabBar):
    def __init__(self):
        super().__init__()
        self.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.setDrawBase(False)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        for i in range(self.count()):
            rect = self.tabRect(i)
            rect.adjust(4, 4, -4, -4)  # Margin for pills
            
            if self.currentIndex() == i:
                painter.setBrush(QBrush(QColor(C_ACCENT)))
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(rect, rect.height()//2, rect.height()//2)
                painter.setPen(QColor("#000000"))
            else:
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor(C_ACCENT), 1))
                painter.drawRoundedRect(rect, rect.height()//2, rect.height()//2)
                painter.setPen(QColor(C_ACCENT))
                
            painter.drawText(rect, Qt.AlignCenter, self.tabText(i))

class CommandHeader(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(55)
        self.hw_label = QLabel("GPU: --%  |  Temp: --°C  |  Pwr: --W")
        self.hw_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.hw_label.setStyleSheet("background: transparent; color: #E8F4FD;")
        self.init_ui()

    def init_ui(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(15, 0, 15, 0)
        
        # Left
        left_box = QHBoxLayout()
        left_box.setSpacing(5)
        l1 = QLabel("MEDAI")
        l1.setFont(QFont("Segoe UI", 20, QFont.Bold))
        l1.setStyleSheet(f"color: {C_ACCENT}; background: transparent;")
        l2 = QLabel("Edge v2.0")
        l2.setFont(QFont("Segoe UI", 12, QFont.Bold))
        l2.setStyleSheet(f"color: {C_TEXT_SEC}; background: transparent; padding-top: 6px;")
        left_box.addWidget(l1)
        left_box.addWidget(l2)
        
        # Center
        c = QLabel("NVIDIA Jetson Nano 4GB  |  Multi-Modal Diagnostic AI")
        c.setFont(QFont("Segoe UI", 10))
        c.setStyleSheet(f"color: {C_TEXT_SEC}; background: transparent;")
        c.setAlignment(Qt.AlignCenter)
        
        lay.addLayout(left_box)
        lay.addWidget(c, stretch=1)
        lay.addWidget(self.hw_label, alignment=Qt.AlignRight)

    def paintEvent(self, event):
        painter = QPainter(self)
        grad = QLinearGradient(0, 0, self.width(), 0)
        grad.setColorAt(0, QColor(C_BG_SECONDARY))
        grad.setColorAt(1, QColor(C_BG_PRIMARY))
        painter.fillRect(self.rect(), grad)
        painter.setPen(QPen(QColor(C_ACCENT), 1))
        painter.drawLine(0, self.height()-1, self.width(), self.height()-1)

    def update_hw(self, gpu, temp, pwr):
        color = C_SUCCESS if temp < 50 else (C_WARNING if temp < 65 else C_CRITICAL)
        self.hw_label.setText(f"GPU: {gpu}%  |  Temp: <span style='color:{color}'>{temp:.1f}°C</span>  |  Pwr: {pwr:.1f}W")

class ConfidenceBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(12)
        self.value = 0

    def set_value(self, val):
        self.value = val
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        
        painter.setBrush(QColor(C_BG_SECONDARY))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, 6, 6)
        
        if self.value > 0:
            fill_width = int(rect.width() * (self.value / 100.0))
            fill_rect = QRect(0, 0, fill_width, rect.height())
            grad = QLinearGradient(0, 0, rect.width(), 0)
            grad.setColorAt(0.0, QColor(C_CRITICAL))
            grad.setColorAt(0.5, QColor(C_WARNING))
            grad.setColorAt(1.0, QColor(C_SUCCESS))
            painter.setBrush(grad)
            painter.drawRoundedRect(fill_rect, 6, 6)

class ModalityCard(QFrame):
    def __init__(self, name):
        super().__init__()
        self.setProperty('class', 'Card')
        self.setMinimumHeight(60)
        self.name = name
        self.is_selected = False
        
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        self.lbl = QLabel(name)
        self.lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.lbl)
        self.update_style()

    def mousePressEvent(self, event):
        self.is_selected = True
        self.update_style()
        # Trigger unselect on siblings handled in parent logic

    def update_style(self):
        if self.is_selected:
            self.setStyleSheet(f"QFrame {{ background: {C_BG_SECONDARY}; border: 1px solid {C_BORDER}; border-left: 4px solid {C_ACCENT}; border-radius: 6px; }}")
            self.lbl.setStyleSheet(f"color: {C_ACCENT};")
        else:
            self.setStyleSheet(f"QFrame {{ background: {C_CARD_BG}; border: 1px solid {C_BORDER}; border-radius: 6px; }}")
            self.lbl.setStyleSheet(f"color: {C_TEXT_PRI};")

class PillButtonGroup(QWidget):
    def __init__(self, labels):
        super().__init__()
        self.setFixedHeight(36)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0,0,0,0)
        lay.setSpacing(0)
        
        self.btns = []
        for i, text in enumerate(labels):
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setFixedHeight(36)
            btn.clicked.connect(lambda checked, b=btn: self.handle_click(b))
            self.btns.append(btn)
            lay.addWidget(btn)
            
        self.btns[0].setChecked(True)
        self.update_styles()

    def handle_click(self, clicked_btn):
        for btn in self.btns:
            btn.setChecked(btn == clicked_btn)
        self.update_styles()

    def update_styles(self):
        for i, btn in enumerate(self.btns):
            rad = "18px 0px 0px 18px" if i == 0 else ("0px 18px 18px 0px" if i == len(self.btns)-1 else "0px")
            border = "border-right: none;" if i < len(self.btns)-1 else ""
            if btn.isChecked():
                btn.setStyleSheet(f"background: {C_ACCENT}; color: black; border: 1px solid {C_ACCENT}; border-radius: {rad}; font-weight: bold;")
            else:
                btn.setStyleSheet(f"background: transparent; color: {C_TEXT_PRI}; border: 1px solid {C_BORDER}; {border} border-radius: {rad};")

# =====================================================================
# EXISTING WIDGETS (REFACTORED FOR AESTHETIC)
# =====================================================================
class CircularGauge(QWidget):
    def __init__(self, title, unit, color):
        super().__init__()
        self.title = title
        self.unit = unit
        self.base_color = color
        self.current_color = color
        self.value = 0
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(160, 160)
        self.setMaximumSize(220, 220)

    def set_value(self, val):
        self.value = val
        if "Temp" in self.title:
            if val < 50: self.current_color = C_SUCCESS
            elif val < 65: self.current_color = C_WARNING
            else: self.current_color = C_CRITICAL
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        size = min(rect.width(), rect.height()) - 20
        x = (rect.width() - size) / 2
        y = (rect.height() - size) / 2
        bounding_rect = QRectF(x, y, size, size)

        pen_bg = QPen(QColor(C_CARD_BG), 14, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen_bg)
        painter.drawArc(bounding_rect, 225 * 16, -270 * 16)

        pen_val = QPen(QColor(self.current_color), 14, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen_val)
        span_angle = int(-270 * (min(100, max(0, self.value)) / 100.0) * 16)
        painter.drawArc(bounding_rect, 225 * 16, span_angle)

        painter.setPen(QColor(C_TEXT_PRI))
        painter.setFont(QFont("Segoe UI", 20, QFont.Bold))
        painter.drawText(rect, Qt.AlignCenter, str(int(self.value)))
        
        painter.setPen(QColor(C_TEXT_SEC))
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(QRect(rect.x(), rect.y() + size//3 + 10, rect.width(), rect.height()), Qt.AlignCenter, self.unit)
        painter.drawText(QRect(rect.x(), rect.y() - size//2.5, rect.width(), rect.height()), Qt.AlignCenter, self.title)

class SemiCircularGauge(QWidget):
    def __init__(self):
        super().__init__()
        self.value = 85
        self.setMinimumSize(320, 200)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        size = min(rect.width(), rect.height() * 2) - 40
        x = (rect.width() - size) / 2
        y = rect.height() - (size / 2) - 20
        bounding_rect = QRectF(x, y, size, size)

        pen_width = 30
        arcs = [(C_SUCCESS, 180, -45), (C_WARNING, 135, -45), ("#E67E22", 90, -45), (C_CRITICAL, 45, -45)]
        for color, start, span in arcs:
            painter.setPen(QPen(QColor(color), pen_width, Qt.SolidLine, Qt.FlatCap))
            painter.drawArc(bounding_rect, start * 16, span * 16)

        painter.translate(rect.width() / 2, y + size/2)
        painter.rotate(180 - (self.value / 100.0 * 180))
        
        path = QPainterPath()
        path.moveTo(-12, 0); path.lineTo(12, 0); path.lineTo(0, -size/2 + 25)
        painter.setBrush(QBrush(QColor(C_TEXT_PRI)))
        painter.setPen(Qt.NoPen)
        painter.drawPath(path)
        painter.drawEllipse(QPoint(0,0), 16, 16)
        
        painter.resetTransform()
        painter.setPen(QColor(C_TEXT_PRI))
        painter.setFont(QFont("Segoe UI", 18, QFont.Bold))
        painter.drawText(QRect(0, int(y + size/2 + 10), rect.width(), 30), Qt.AlignCenter, "CRITICAL RISK")

class TimelineWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(90)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        mid_y = h // 2 - 10
        
        painter.setPen(QPen(QColor(C_BORDER), 4))
        painter.drawLine(60, mid_y, w - 60, mid_y)
        
        visits = [("Jan 12", C_SUCCESS), ("Feb 05", C_SUCCESS), ("Mar 22", C_WARNING), ("May 05", C_CRITICAL)]
        spacing = (w - 120) / max(1, len(visits) - 1)
        
        for i, (date, color) in enumerate(visits):
            cx = int(60 + i * spacing)
            painter.setBrush(QBrush(QColor(color)))
            painter.setPen(QPen(QColor(C_BG_PRIMARY), 4))
            painter.drawEllipse(QPoint(cx, mid_y), 12, 12)
            
            painter.setPen(QColor(C_TEXT_SEC))
            painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
            painter.drawText(cx - 40, mid_y + 25, 80, 20, Qt.AlignCenter, date)

class CameraOverlayLabel(QLabel):
    def __init__(self, text):
        super().__init__(text)
        self.show_bbox = True
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background-color: {C_BG_SECONDARY}; color: {C_TEXT_SEC}; border: 1px solid {C_BORDER}; border-radius: 8px;")

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.show_bbox:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QPen(QColor(C_WARNING), 3, Qt.DashLine))
            cx, cy = self.width()//2, self.height()//2
            rw, rh = 250, 300
            painter.drawRect(cx - rw//2, cy - rh//2, rw, rh)
            painter.fillRect(cx - rw//2, cy - rh//2 - 25, 120, 25, QColor(C_WARNING))
            painter.setPen(QColor("#000"))
            painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
            painter.drawText(cx - rw//2 + 5, cy - rh//2 - 7, "Lesion: 88%")

class TypewriterTextEdit(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMinimumHeight(120)
        self.setStyleSheet(f"background-color: {C_BG_SECONDARY}; color: {C_TEXT_PRI}; border: 1px solid {C_BORDER}; padding: 10px; font-family: monospace; font-size: 11pt;")
        self.full_text = ""
        self.current_idx = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self._add_char)

    def start_typing(self, text):
        self.clear()
        self.full_text = text
        self.current_idx = 0
        self.timer.start(20)

    def _add_char(self):
        if self.current_idx < len(self.full_text):
            self.insertPlainText(self.full_text[self.current_idx])
            self.current_idx += 1
            self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        else:
            self.timer.stop()

# =====================================================================
# THREADS
# =====================================================================
class HardwareWorker(QThread):
    data_updated = pyqtSignal(dict)
    def run(self):
        parser = TegrastatsParser()
        ssh_command = ["ssh", "jetson-nano@192.168.1.100", "tegrastats"]
        try:
            process = subprocess.Popen(ssh_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            while True:
                line = process.stdout.readline()
                if line:
                    metrics = parser.parse_line(line)
                    if not metrics.get("error"):
                        self.data_updated.emit({
                            'gpu': metrics['gpu_percent'], 'cpu': metrics['cpu_percent'],
                            'temp': metrics['temp_c'], 'power': metrics['power_w']
                        })
                else:
                    time.sleep(1)
        except Exception as e:
            print(f"Failed SSH tegrastats, falling back to mock: {e}")
            t, p = 40.0, 5.0
            while True:
                t = max(35, min(75, t + random.uniform(-1.5, 2.0)))
                p = max(3.0, min(10.0, p + random.uniform(-0.5, 0.5)))
                self.data_updated.emit({'gpu': random.randint(80,95), 'cpu': random.randint(20,40), 'temp': t, 'power': p})
                time.sleep(1)

# =====================================================================
# SPLASH SCREEN
# =====================================================================
class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setFixedSize(600, 350)
        self.setStyleSheet(f"background-color: {C_BG_PRIMARY}; border: 2px solid {C_ACCENT}; border-radius: 12px;")
        
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        
        t = QLabel("MEDAI Edge Server v2.0")
        t.setFont(QFont("Segoe UI", 26, QFont.Bold))
        t.setStyleSheet(f"color: {C_ACCENT}; border: none;")
        t.setAlignment(Qt.AlignCenter)
        
        sub = QLabel("ACCELERATED MEDICAL IMAGING USING ML ARCHITECTURES \n Supervised by DR. NAVEED NAWAZ \nUET Lahore - Final Year Project")
        sub.setFont(QFont("Segoe UI", 10))
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(f"color: {C_TEXT_SEC}; border: none;")
        
        self.info = QLabel("Initializing TensorRT Engines...")
        self.info.setAlignment(Qt.AlignCenter)
        self.info.setStyleSheet("border: none; margin-top: 40px; font-weight: bold;")
        
        self.prog = QProgressBar()
        self.prog.setRange(0, 100)
        self.prog.setTextVisible(False)
        self.prog.setFixedHeight(4)
        self.prog.setStyleSheet(f"border: none; background: {C_BORDER};")
        
        lay.addWidget(t)
        lay.addWidget(sub)
        lay.addWidget(self.info)
        lay.addWidget(self.prog)
        
        self.val = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(30)

    def tick(self):
        self.val += 1
        self.prog.setValue(self.val)
        msg = {20: "Loading PyTorch FP16 Models...", 50: "Warming CUDA Context...", 80: "Mounting SQLite DB..."}
        if self.val in msg: self.info.setText(msg[self.val])
        if self.val >= 100:
            self.timer.stop()
            self.close()
            self.main_window = MedicalEdgeDashboard()
            self.main_window.show()

# =====================================================================
# MAIN WINDOW
# =====================================================================
class MedicalEdgeDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MEDAI Command Center")
        self.setMinimumSize(1280, 800)
        self.setStyleSheet(GLOBAL_STYLE)
        
        pg.setConfigOption('background', C_CARD_BG)
        pg.setConfigOption('foreground', C_TEXT_PRI)
        self.hist_temp, self.hist_pow = np.full(60, 40.0), np.full(60, 5.0)

        self.central = QWidget()
        self.setCentralWidget(self.central)
        self.main_layout = QVBoxLayout(self.central)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        self.header = CommandHeader()
        self.main_layout.addWidget(self.header)
        
        self.tabs = QTabWidget()
        self.tabs.setTabBar(PillTabBar())
        self.main_layout.addWidget(self.tabs)
        
        self.build_tab1_scan()
        self.build_tab2_fusion()
        self.build_tab3_alq()
        self.build_tab4_camera()
        self.build_tab5_hw()
        self.build_tab6_federated()
        self.build_tab7_security()
        
        self.build_statusbar()

        self.hw_thread = HardwareWorker()
        self.hw_thread.data_updated.connect(self.update_hw_ui)
        self.hw_thread.start()

        # Effect Timers
        self.pulse_state = False
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self.do_image_pulse)
        
        self.blink_state = False
        self.blink_count = 0
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.do_alert_blink)

    def generate_pdf_report(self):
        # 1. Create the reports directory in the project root if it doesn't exist
        report_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'reports')
        os.makedirs(report_dir, exist_ok=True)
        
        # 2. Generate a unique filename using a timestamp
        filename = f"Report_PAT-001_{int(time.time())}.pdf"
        output_path = os.path.join(report_dir, filename)
        
        # 3. Dummy data dictionary matching Phase 2 Day 3 requirements
        dummy_data = {
            "patient_id": "PAT-001",
            "modality": "Brain MRI",
            "prediction": "Glioma (Malignant)",
            "confidence": "98.8",
            "routing_tier": "TIER 1 (Urgent Review)",
            "llm_text": "AI Analysis Report: The scan reveals a hyperintense mass in the right temporal lobe, highly consistent with a high-grade Glioma. Bayesian uncertainty is exceptionally low (±1.2%), indicating high model confidence. No out-of-distribution artifacts detected. Immediate routing to oncology recommended."
        }
        
        # 4. Generate the PDF
        try:
            generator = MedicalReportGenerator()
            generator.generate_report(dummy_data, output_path)
            QMessageBox.information(self, "Success", f"PDF Report generated successfully!\n\nSaved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate PDF:\n{str(e)}")

    def _wrap_scroll(self, layout):
        inner = QWidget()
        inner.setLayout(layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        return scroll

    def _create_card(self, title, val, c_color=None, min_h=60):
        f = QFrame(); f.setProperty('class', 'Card')
        f.setMinimumHeight(min_h)
        if c_color: f.setStyleSheet(f"QFrame {{ border-left: 3px solid {c_color}; }}")
        l = QHBoxLayout(f); l.setContentsMargins(12,8,12,8)
        lbl_t = QLabel(title); lbl_t.setFont(QFont("Segoe UI", 9)); lbl_t.setStyleSheet(f"color: {C_TEXT_SEC};")
        lbl_v = QLabel(val); lbl_v.setFont(QFont("Segoe UI", 14, QFont.Bold)); lbl_v.setAlignment(Qt.AlignRight)
        if c_color: lbl_v.setStyleSheet(f"color: {c_color};")
        l.addWidget(lbl_t); l.addWidget(lbl_v)
        return f

    # --- TAB 1: SCAN ANALYSIS ---
    def build_tab1_scan(self):
        lay = QHBoxLayout()
        lay.setContentsMargins(15, 15, 15, 15)
        lay.setSpacing(15)
        
        # Left
        left = QVBoxLayout()
        left.setSpacing(10)
        self.img_view = QLabel("DROP DICOM HERE")
        self.img_view.setAlignment(Qt.AlignCenter)
        self.img_view.setMinimumSize(380, 380)
        self.img_view.setStyleSheet(f"background: #000; border: 1px solid {C_BORDER}; border-radius: 12px; color: {C_TEXT_SEC};")
        
        btn_grp = PillButtonGroup(["Original", "GradCAM", "SHAP"])
        
        cl = QHBoxLayout()
        cl.addWidget(QLabel("Mean Confidence:"))
        self.conf_bar = ConfidenceBar()
        self.conf_bar.set_value(0)
        cl.addWidget(self.conf_bar, stretch=1)
        self.conf_lbl = QLabel("0%")
        self.conf_lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        cl.addWidget(self.conf_lbl)
        
        left.addWidget(self.img_view, stretch=1)
        left.addWidget(btn_grp)
        left.addLayout(cl)
        
        # Right
        right = QVBoxLayout()
        right.setSpacing(8)
        right.setAlignment(Qt.AlignTop)
        
        grid = QGridLayout()
        grid.setSpacing(8)
        self.mod_cards = []
        mods = [("🧠 Brain MRI",0,0), ("🫁 Chest X-Ray",0,1), ("👁 Retinal",1,0), ("❤️ ECG",1,1), ("🦷 Skin",2,0), ("🦴 Bone",2,1)]
        for n, r, c in mods:
            mc = ModalityCard(n)
            if r==0 and c==0: mc.is_selected = True; mc.update_style()
            self.mod_cards.append(mc)
            grid.addWidget(mc, r, c)
        
        btn_up = QPushButton("Drop DICOM here or click to browse")
        btn_up.setMinimumHeight(50)
        btn_up.setStyleSheet(f"background: {C_CARD_BG}; border: 1px dashed {C_ACCENT}; color: {C_ACCENT};")
        btn_up.clicked.connect(self.upload_and_analyze_scan)
        
        btn_an = QPushButton("Run Pipeline (Analyze)")
        btn_an.setMinimumHeight(50)
        btn_an.setStyleSheet(f"background: {C_ACCENT}; color: #000; font-size: 12pt; font-weight: bold;")
        btn_an.clicked.connect(self.trigger_analysis_mock)
        
        self.res_class = self._create_card("Predicted Class:", "Pending...", min_h=42)
        self.res_unc = self._create_card("MC Uncertainty:", "--", min_h=42)
        self.res_tier = self._create_card("Routing Tier:", "--", min_h=42)
        self.res_ood = self._create_card("OOD Status:", "--", min_h=42)
        
        self.typewriter = TypewriterTextEdit()
        btn_pdf = QPushButton("Generate PDF Report")
        btn_pdf.setProperty('class', 'Primary')
        btn_pdf.setMinimumHeight(44)
        btn_pdf.clicked.connect(self.generate_pdf_report) # THIS WIRES IT UP

        right.addLayout(grid)
        right.addWidget(btn_up)
        right.addWidget(btn_an)
        right.addWidget(self.res_class)
        right.addWidget(self.res_unc)
        right.addWidget(self.res_tier)
        right.addWidget(self.res_ood)
        right.addWidget(QLabel("Phi-2 Clinical Report:"))
        right.addWidget(self.typewriter)
        right.addWidget(btn_pdf)
        
        lay.addLayout(left, stretch=6)
        lay.addLayout(right, stretch=4)
        self.tabs.addTab(self._wrap_scroll(lay), "Scan")

    def do_image_pulse(self):
        self.pulse_state = not self.pulse_state
        bc = C_ACCENT if self.pulse_state else C_BORDER
        self.img_view.setStyleSheet(f"background: #000; border: 2px solid {bc}; border-radius: 12px; color: {C_TEXT_PRI};")

    def do_alert_blink(self):
        if self.blink_count >= 12: # ~10s at 800ms
            self.blink_timer.stop()
            self.res_tier.setStyleSheet(f"QFrame {{ background: {C_CARD_BG}; border: 1px solid {C_BORDER}; border-left: 3px solid {C_CRITICAL}; }}")
            return
        self.blink_state = not self.blink_state
        bg = "#4A0000" if self.blink_state else C_CARD_BG
        self.res_tier.setStyleSheet(f"QFrame {{ background: {bg}; border: 1px solid {C_CRITICAL}; border-left: 3px solid {C_CRITICAL}; }}")
        self.blink_count += 1

    def trigger_analysis_mock(self):
        self.conf_bar.set_value(98); self.conf_lbl.setText("98%")
        self.res_class.layout().itemAt(1).widget().setText("Glioma (Malignant)")
        self.res_class.layout().itemAt(1).widget().setStyleSheet(f"color: {C_CRITICAL}; font-size: 14pt; font-weight: bold;")
        self.res_class.setStyleSheet(f"QFrame {{ border-left: 3px solid {C_CRITICAL}; }}")
        
        self.res_unc.layout().itemAt(1).widget().setText("± 1.2% (Low)")
        self.res_unc.layout().itemAt(1).widget().setStyleSheet(f"color: {C_SUCCESS}; font-size: 14pt; font-weight: bold;")
        self.res_unc.setStyleSheet(f"QFrame {{ border-left: 3px solid {C_SUCCESS}; }}")
        
        self.res_tier.layout().itemAt(1).widget().setText("TIER 1 (Urgent)")
        self.res_tier.layout().itemAt(1).widget().setStyleSheet(f"color: {C_CRITICAL}; font-size: 14pt; font-weight: bold;")
        
        self.res_ood.layout().itemAt(1).widget().setText("CLEAR")
        self.res_ood.layout().itemAt(1).widget().setStyleSheet(f"color: {C_SUCCESS}; font-size: 14pt; font-weight: bold;")
        self.res_ood.setStyleSheet(f"QFrame {{ border-left: 3px solid {C_SUCCESS}; }}")
        
        self.blink_count = 0; self.blink_timer.start(800)
        self.typewriter.start_typing("AI Analysis Report:\nThe scan reveals a hyperintense mass in the right temporal lobe, highly consistent with a high-grade Glioma. Bayesian uncertainty is exceptionally low (±1.2%), indicating high model confidence. No out-of-distribution artifacts detected. Immediate routing to oncology recommended.")

    # --- TAB 2: FUSION ---
    def build_tab2_fusion(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(15,15,15,15)
        lay.setSpacing(15)
        
        sh = QHBoxLayout()
        src = QLineEdit(); src.setPlaceholderText("Search Patient ID (e.g. PAT-9042)"); src.setMinimumHeight(40)
        bs = QPushButton("Fetch"); bs.setProperty('class', 'Primary'); bs.setMinimumHeight(40)
        sh.addWidget(src); sh.addWidget(bs)
        lay.addLayout(sh)
        
        lay.addWidget(SemiCircularGauge(), alignment=Qt.AlignCenter)
        
        hc = QHBoxLayout(); hc.setSpacing(8)
        for m, d, w, c in [("🧠 Brain MRI", "Glioma", "85", C_CRITICAL), ("🫁 Chest X-Ray", "Normal", "10", C_SUCCESS), ("❤️ ECG", "Arrhythmia", "5", C_WARNING)]:
            f = QFrame(); f.setProperty('class', 'Card'); f.setMinimumHeight(100)
            fl = QVBoxLayout(f)
            fl.addWidget(QLabel(f"<b>{m}</b>"))
            dl = QLabel(d); dl.setStyleSheet(f"color: {c}; font-size: 14pt; font-weight: bold;")
            fl.addWidget(dl)
            pb = QProgressBar(); pb.setFixedHeight(6); pb.setValue(int(w)); pb.setTextVisible(False)
            pb.setStyleSheet(f"QProgressBar {{ background: {C_BG_PRIMARY}; border: none; }} QProgressBar::chunk {{ background: {C_ACCENT}; border-radius: 3px; }}")
            al = QHBoxLayout(); al.addWidget(QLabel("Attention:")); al.addWidget(pb)
            fl.addLayout(al)
            hc.addWidget(f)
        lay.addLayout(hc)
        
        lay.addWidget(TimelineWidget())
        tb = QLabel("TREND: DISEASE PROGRESSION DETECTED")
        tb.setFixedHeight(36)
        tb.setStyleSheet(f"background: {C_CRITICAL}; color: white; border-radius: 18px; font-weight: bold;")
        tb.setAlignment(Qt.AlignCenter)
        lay.addWidget(tb)
        
        lay.setAlignment(Qt.AlignTop)
        self.tabs.addTab(self._wrap_scroll(lay), "Fusion")

    # --- TAB 3: ALQ ---
    def build_tab3_alq(self):
        top_lay = QVBoxLayout()
        top_lay.setContentsMargins(15,15,15,15); top_lay.setSpacing(15)
        
        hs = QHBoxLayout(); hs.setSpacing(8)
        hs.addWidget(self._create_card("Pending Review", "12", C_WARNING, 80))
        hs.addWidget(self._create_card("Corrected Today", "3", C_SUCCESS, 80))
        hs.addWidget(self._create_card("Model Updated", "2 hrs ago", C_PURPLE, 80))
        top_lay.addLayout(hs)
        
        self.alq_table = QTableWidget(0, 6)
        self.alq_table.setHorizontalHeaderLabels(["Patient ID", "Date", "Modality", "AI Prediction", "Uncertainty", "Action"])
        self.alq_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.alq_table.verticalHeader().setVisible(False)
        self.alq_table.setAlternatingRowColors(True)
        self.alq_table.setWordWrap(False)
        top_lay.addWidget(self.alq_table)
        
        top_w = QWidget(); top_w.setLayout(top_lay)
        
        cp = QFrame(); cp.setProperty('class', 'Card'); cp.setMinimumHeight(120)
        cpl = QVBoxLayout(cp); cpl.setContentsMargins(15,15,15,15)
        self.lbl_correction_target = QLabel("<b>EWC Correction Panel (Select a Patient)</b>")
        cpl.addWidget(self.lbl_correction_target)
        
        hcp = QHBoxLayout()
        hcp.addWidget(QLabel("True Label:"))
        self.correction_dropdown = QComboBox()
        self.correction_dropdown.addItems(["Normal", "Pneumonia", "COVID-19", "Melanoma"])
        hcp.addWidget(self.correction_dropdown)
        hcp.addWidget(QLabel("EWC Lambda:"))
        hcp.addWidget(QSlider(Qt.Horizontal))
        self.btn_ewc = QPushButton("Submit Correction + Run EWC")
        self.btn_ewc.setProperty('class', 'Warning')
        self.btn_ewc.setMinimumHeight(36)
        self.btn_ewc.clicked.connect(self.mock_ewc)
        hcp.addWidget(self.btn_ewc)
        cpl.addLayout(hcp)
        
        self.prog_ewc = QProgressBar()
        self.prog_ewc.setValue(0); self.prog_ewc.setFixedHeight(6)
        cpl.addWidget(self.prog_ewc)
        
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(top_w)
        splitter.addWidget(cp)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        
        ml = QVBoxLayout(); ml.setContentsMargins(0,0,0,0)
        ml.addWidget(splitter)
        self.tabs.addTab(self._wrap_scroll(ml), "Learning")
        
        QTimer.singleShot(100, self.refresh_alq_table)

    def mock_ewc(self):
        self.prog_ewc.setValue(100)
        self.btn_ewc.setText("✓ Success: EWC Complete")
        self.btn_ewc.setStyleSheet(f"background: {C_PURPLE}; color: white;")

# --- TAB 4: CAMERA ---
    # --- TAB 4: CAMERA ---
    def build_tab4_camera(self):
        tab = QWidget()                       # <--- NEW
        lay = QHBoxLayout(tab)                # <--- CHANGED
        lay.setContentsMargins(15,15,15,15); lay.setSpacing(15)
        
        self.cam_lbl = CameraOverlayLabel("[ USB/CSI OFFLINE ]")
        self.cam_lbl.setMinimumSize(400, 350)
        lay.addWidget(self.cam_lbl, stretch=3)
        
        r = QVBoxLayout(); r.setSpacing(10)
        rw = QWidget(); rw.setMinimumWidth(220); rw.setMaximumWidth(280); rw.setLayout(r)
        
        r.addWidget(QLabel("<b>Camera Target:</b>"))
        cb = QComboBox(); cb.addItems(["Skin Lesion (HAM10000)", "Retinal Fundus"])
        r.addWidget(cb)
        
        r.addWidget(self._create_card("Live Inference", "MELANOMA\n88% | 14fps", C_CRITICAL, 80))
        
        bs = QPushButton("Start Live Stream"); bs.setProperty('class', 'Success'); bs.setMinimumHeight(48)
        bc = QPushButton("📸 Capture Frame"); bc.setMinimumHeight(44)
        r.addWidget(bs); r.addWidget(bc)
        r.addStretch()
        
        lay.addWidget(rw, stretch=1)
        self.tabs.addTab(tab, "Camera")       # <--- CHANGED

    # --- TAB 5: HARDWARE ---
    def build_tab5_hw(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(15,15,15,15); lay.setSpacing(15)
        
        hg = QHBoxLayout(); hg.setSpacing(15)
        self.g_gpu = CircularGauge("GPU Usage", "%", C_ACCENT)
        self.g_cpu = CircularGauge("CPU Usage", "%", C_SUCCESS)
        self.g_temp = CircularGauge("Core Temp", "°C", C_SUCCESS)
        self.g_pow = CircularGauge("Power (10W max)", "W", C_ACCENT)
        hg.addWidget(self.g_gpu); hg.addWidget(self.g_cpu); hg.addWidget(self.g_temp); hg.addWidget(self.g_pow)
        lay.addLayout(hg)
        
        self.lbl_warn = QLabel("⚠ THERMAL WARNING")
        self.lbl_warn.setFixedHeight(36)
        self.lbl_warn.setStyleSheet(f"background: {C_WARNING}; color: black; font-weight: bold; border-radius: 4px;")
        self.lbl_warn.setAlignment(Qt.AlignCenter)
        self.lbl_warn.hide()
        lay.addWidget(self.lbl_warn)
        
        hp = QHBoxLayout(); hp.setSpacing(15)
        self.plt_temp = pg.PlotWidget(title="Temp (60s)")
        self.plt_temp.setMinimumHeight(200)
        self.plt_temp.setYRange(30, 80)
        self.plt_temp.showGrid(x=True, y=True, alpha=0.2)
        self.crv_temp = self.plt_temp.plot(pen=pg.mkPen(color=C_CRITICAL, width=2))
        thr = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(color=C_WARNING, style=Qt.DashLine))
        thr.setPos(70); self.plt_temp.addItem(thr)
        hp.addWidget(self.plt_temp)
        
        self.plt_pow = pg.PlotWidget(title="Power (W)")
        self.plt_pow.setMinimumHeight(200)
        self.plt_pow.setYRange(0, 15)
        self.plt_pow.showGrid(x=True, y=True, alpha=0.2)
        self.crv_pow = self.plt_pow.plot(pen=pg.mkPen(color=C_ACCENT, width=2))
        hp.addWidget(self.plt_pow)
        
        lay.addLayout(hp)
        lay.addWidget(QLabel("Red dashed line = 70°C Thermal Throttle Limit"), alignment=Qt.AlignCenter)
        
        self.tabs.addTab(self._wrap_scroll(lay), "Hardware")

    # --- TAB 6: FEDERATED ---
    def build_tab6_federated(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(15,15,15,15); lay.setSpacing(15)
        
        hh = QHBoxLayout(); hh.setSpacing(10)
        for h, a in [("Hospital A", "92.4%"), ("Hospital B", "89.1%"), ("Hospital C", "94.0%")]:
            f = QFrame(); f.setProperty('class', 'Card'); f.setMinimumHeight(110)
            fl = QVBoxLayout(f)
            fl.addWidget(QLabel(f"<b>{h}</b>"))
            fl.addWidget(QLabel("🟢 Online"))
            fl.addWidget(QLabel(f"Acc: <span style='color:{C_SUCCESS}'><b>{a}</b></span>"))
            fl.addWidget(QLabel("Samples: 1,204"))
            hh.addWidget(f)
        lay.addLayout(hh)
        
        bs = QPushButton("Synchronize Global Network (FedAvg)")
        bs.setMinimumHeight(50)
        bs.setStyleSheet(f"background: {C_WARNING}; color: black; font-size: 11pt; font-weight: bold;")
        lay.addWidget(bs)
        
        tbl = QTableWidget(3, 5)
        tbl.setHorizontalHeaderLabels(["Round", "Hosp A", "Hosp B", "Hosp C", "Global"])
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tbl.verticalHeader().setVisible(False)
        tbl.setMinimumHeight(180)
        for r, row in enumerate([("3","92.4%","89.1%","94.0%","93.8%"), ("2","91.0%","88.5%","92.1%","91.5%"), ("1","88.2%","85.0%","89.0%","88.0%")]):
            for c, val in enumerate(row): tbl.setItem(r, c, QTableWidgetItem(val))
        lay.addWidget(tbl)
        
        t = QTextEdit()
        t.setReadOnly(True)
        t.setMinimumHeight(120); t.setMaximumHeight(180)
        t.setStyleSheet(f"background: #000; color: {C_SUCCESS}; font-family: monospace; border: 1px solid {C_BORDER};")
        t.setText("> Initializing Federated Server...\n> 3 nodes connected.\n> Awaiting Round 4 aggregation...")
        lay.addWidget(t)
        
        lay.setAlignment(Qt.AlignTop)
        self.tabs.addTab(self._wrap_scroll(lay), "Federated")

    # --- TAB 7: SECURITY ---
    def build_tab7_security(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(15,15,15,15); lay.setSpacing(15)
        
        grid = QGridLayout(); grid.setSpacing(10)
        for i, (t, d, c) in enumerate([("AES-256 RAM Encrypt", "Key: Jetson Serial", C_SUCCESS), ("JWT Auth", "Token valid", C_SUCCESS), 
                                       ("TLS / HTTPS", "Cert Valid", C_SUCCESS), ("DICOM C-FIND", "Queue: 4", C_ACCENT)]):
            f = self._create_card(t, "ACTIVE", c, 100)
            f.layout().addWidget(QLabel(d))
            grid.addWidget(f, i//2, i%2)
        lay.addLayout(grid)
        
        t = QTextEdit()
        t.setReadOnly(True)
        t.setMinimumHeight(200)
        t.setStyleSheet("background: #000; color: #E8F4FD; font-family: monospace; border: 1px solid #1E2D40;")
        t.setText("[INFO] TLS Handshake OK\n[WARN] GPU Temp rising\n[CRITICAL] TIER 3 Route triggered.")
        lay.addWidget(t)
        
        lay.setAlignment(Qt.AlignTop)
        self.tabs.addTab(self._wrap_scroll(lay), "Security")

    def build_statusbar(self):
        bar = QWidget()
        bar.setFixedHeight(28)
        bar.setStyleSheet(f"background: {C_CARD_BG}; border-top: 1px solid {C_BORDER};")
        lay = QHBoxLayout(bar); lay.setContentsMargins(10, 0, 10, 0)
        
        def badge(txt, c):
            l = QLabel(txt); l.setStyleSheet(f"background:{c}; color:black; padding:2px 8px; border-radius:8px; font-weight:bold; font-size:8pt;")
            return l
            
        lay.addWidget(QLabel("System:"))
        lay.addWidget(badge("ONLINE", C_SUCCESS))
        lay.addWidget(QLabel(" | Infer:"))
        lay.addWidget(badge("READY", C_SUCCESS))
        lay.addStretch()
        self.lbl_clock = QLabel()
        self.lbl_clock.setFont(QFont("Courier", 9))
        lay.addWidget(self.lbl_clock)
        self.main_layout.addWidget(bar)
        
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(lambda: self.lbl_clock.setText(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        self.clock_timer.start(1000)

    # --- FUNCTIONAL LOGIC ---
    def update_hw_ui(self, data):
        self.header.update_hw(data['gpu'], data['temp'], data['power'])
        self.g_gpu.set_value(data['gpu'])
        self.g_cpu.set_value(data['cpu'])
        self.g_temp.set_value(data['temp'])
        self.g_pow.set_value((data['power']/10.0)*100); self.g_pow.value = data['power']

        self.hist_temp[:-1] = self.hist_temp[1:]; self.hist_temp[-1] = data['temp']
        self.crv_temp.setData(self.hist_temp)
        self.hist_pow[:-1] = self.hist_pow[1:]; self.hist_pow[-1] = data['power']
        self.crv_pow.setData(self.hist_pow)
        
        if data['temp'] > 65:
            self.lbl_warn.show()
            if data['temp'] > 70:
                self.lbl_warn.setStyleSheet(f"background: {C_CRITICAL}; color: white; font-weight: bold; border-radius: 4px;")
                self.lbl_warn.setText("⚠ CRITICAL: Thermal Throttling Active (>70°C)")
        else: self.lbl_warn.hide()

    def upload_and_analyze_scan(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Medical Scan", "", "Medical Images (*.png *.jpg *.jpeg *.dcm);;All Files (*)", options=options)
        if not file_path: return
        
        filename = os.path.basename(file_path)
        self.img_view.setText(f"Loaded:\n{filename}")
        self.img_view.setStyleSheet(f"background: #000; border: 2px solid {C_ACCENT}; border-radius: 12px; color: {C_TEXT_PRI};")
        self.pulse_timer.start(1000)

        api_url = "http://192.168.1.100:5000/analyze"
        try:
            with open(file_path, 'rb') as img_file:
                resp = requests.post(api_url, files={'file': (filename, img_file, 'application/octet-stream')}, timeout=2)
            if resp.status_code == 200: QMessageBox.information(self, "Success", "Scan sent to Jetson.")
            else: QMessageBox.warning(self, "API Error", f"Status: {resp.status_code}")
        except requests.exceptions.RequestException as e:
            pass # Suppressing popup for layout test without actual Nano connection
            print(f"Flask API unreachable: {e}")

    def refresh_alq_table(self):
        # MOCK DB LOAD for pure UI testing - integrates db logic without crashing if no sqlite file
        self.alq_table.setRowCount(0)
        data = [("PAT-001", "2026-05-05", "Chest X-Ray", "Pneumonia", 14.2), ("PAT-042", "2026-05-04", "Skin", "Melanoma", 8.1)]
        for r, row in enumerate(data):
            self.alq_table.insertRow(r)
            for c, val in enumerate(row):
                if c == 4:
                    ub = QLabel(f"{val:.1f}%")
                    col = C_CRITICAL if val > 10.0 else C_WARNING
                    ub.setStyleSheet(f"background:{col}; color:black; border-radius:10px; padding:2px; font-weight:bold;")
                    ub.setAlignment(Qt.AlignCenter)
                    self.alq_table.setCellWidget(r, 4, ub)
                else:
                    item = QTableWidgetItem(str(val))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.alq_table.setItem(r, c, item)
            b = QPushButton("Review"); b.setProperty('class', 'Primary')
            b.clicked.connect(lambda chk, pid=row[0]: self.load_correction_panel(pid))
            self.alq_table.setCellWidget(r, 5, b)

    def load_correction_panel(self, pid):
        self.lbl_correction_target.setText(f"<b>EWC Correction Panel (Selected: {pid})</b>")
        self.prog_ewc.setValue(0)
        self.btn_ewc.setText("Submit Correction + Run EWC")
        self.btn_ewc.setStyleSheet(f"background: {C_WARNING}; color: black;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    splash = SplashScreen()
    splash.show()
    sys.exit(app.exec_())

# REDESIGN SUMMARY:
# - Visually replaced standard tabs with custom painted PillTabBar and command center palette.
# - Preserved all 10 specific functions/methods, all thread logic, and HW parsing completely intact.
# - Added low-RAM effects: 1s pulse on loaded image, 800ms blink on critical tier routing.
# - Wrapped dense tabs in QScrollArea and used QSplitter in Learning tab to prevent absolutely all clipping.
# - Jetson limits respected: No web engines, no opacity loops, strictly used QPainter and QTimers.
