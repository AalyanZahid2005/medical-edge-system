import sys
import base64
import time
import random
import datetime
import requests
import sqlite3
import cv2
import os
import subprocess
import numpy as np
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtGui import QImage, QPixmap
from pdf_generator import MedicalReportGenerator

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, 
                             QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, 
                             QPushButton, QComboBox, QTableWidget, QTableWidgetItem, 
                             QTextEdit, QFrame, QHeaderView, QProgressBar, QScrollArea, 
                             QSlider, QRadioButton, QLineEdit, QFileDialog, QMessageBox,
                             QTabBar, QSplitter, QSizePolicy, QGroupBox, QCheckBox) # <--- Added QCheckBox
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QRect, QRectF, QPoint, QSize
from PyQt5.QtGui import (QFont, QFontDatabase, QColor, QPainter, QPen, QBrush, 
                         QPainterPath, QLinearGradient, QPixmap)

import pyqtgraph as pg

# Matplotlib integration for Clinical Validation Tab
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        
        # Match the MEDAI Command Center Dark Theme
        self.fig.patch.set_facecolor('#141B26') 
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#0F1520') 
        
        # Style the axes and text
        self.axes.tick_params(colors='#E8F4FD') 
        self.axes.xaxis.label.set_color('#E8F4FD')
        self.axes.yaxis.label.set_color('#E8F4FD')
        self.axes.title.set_color('#E8F4FD')
        
        for spine in ['bottom', 'top', 'left', 'right']:
            self.axes.spines[spine].set_color('#1E2D40')
            
        super(MplCanvas, self).__init__(self.fig)

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
            rect.adjust(4, 4, -4, -4)
            
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

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        
        # Match the MEDAI Command Center Dark Theme
        self.fig.patch.set_facecolor('#141B26') 
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#0F1520') 
        
        # Style the axes and text for dark mode
        self.axes.tick_params(colors='#E8F4FD') 
        self.axes.xaxis.label.set_color('#E8F4FD')
        self.axes.yaxis.label.set_color('#E8F4FD')
        self.axes.title.set_color('#E8F4FD')
        
        for spine in ['bottom', 'top', 'left', 'right']:
            self.axes.spines[spine].set_color('#1E2D40')
            
        super(MplCanvas, self).__init__(self.fig)

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
        painter.drawText(
            QRect(
                int(rect.x()),
                int(rect.y() - (size / 2.5)),
                int(rect.width()),
                int(rect.height())
            ),
            Qt.AlignCenter,
            self.title
        )

class SemiCircularGauge(QWidget):
    def __init__(self):
        super().__init__()
        self.value = 85
        self.setMinimumSize(320, 240) # Increased base height
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed) # Prevent vertical squishing

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()
        
        # Add padding so the arc isn't cut off at the edges
        size = min(rect.width(), rect.height() * 2) - 60
        x = (rect.width() - size) / 2
        y = rect.height() - (size / 2) - 40 # Pull it up from the bottom edge
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
        
        # Safely draw text below the arc without clipping
        painter.drawText(QRect(0, int(y + size/2 + 10), rect.width(), 40), Qt.AlignCenter, "CRITICAL RISK")

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

class RiskTrafficLight(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(100, 260)
        self.setMaximumSize(120, 300)
        self.current_state = 0 

    def set_risk_level(self, level):
        self.current_state = level
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        housing_rect = self.rect().adjusted(10, 10, -10, -10)
        painter.setBrush(QBrush(QColor("#1A1A24")))
        painter.setPen(QPen(QColor("#00E5FF"), 2))
        painter.drawRoundedRect(housing_rect, 15, 15)

        c_red_on = QColor("#FF3B30"); c_red_off = QColor("#4A1515")
        c_yel_on = QColor("#FFCC00"); c_yel_off = QColor("#4A3B05")
        c_grn_on = QColor("#34C759"); c_grn_off = QColor("#103D1B")

        w = self.width()
        cx = w // 2
        r = 24  
        spacing = 65
        start_y = 40

        painter.setBrush(QBrush(c_red_on if self.current_state == 3 else c_red_off))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(cx - r, start_y, r*2, r*2)

        painter.setBrush(QBrush(c_yel_on if self.current_state == 2 else c_yel_off))
        painter.drawEllipse(cx - r, start_y + spacing, r*2, r*2)

        painter.setBrush(QBrush(c_grn_on if self.current_state == 1 else c_grn_off))
        painter.drawEllipse(cx - r, start_y + spacing * 2, r*2, r*2)

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



class AnalysisWorker(QThread):
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, file_path, target_lang, modality="brain_mri", patient_id="PT-M4-GUI-001"):
        super().__init__()
        self.file_path = file_path
        self.target_lang = target_lang
        self.modality = modality
        self.patient_id = patient_id
        self.api_url = "http://10.42.0.51:5000/analyze"

    def run(self):
        try:
            with open(self.file_path, "rb") as img_file:
                files = {
                    "file": (
                        os.path.basename(self.file_path),
                        img_file,
                        "application/octet-stream"
                    )
                }

                data = {
                    "language": self.target_lang,
                    "modality": self.modality,
                    "patient_id": self.patient_id
                }

                resp = requests.post(
                    self.api_url,
                    files=files,
                    data=data,
                    timeout=300
                )

            if resp.status_code == 200:
                self.finished_signal.emit(resp.json())
            else:
                self.error_signal.emit(
                    "Jetson API Error: HTTP {}\n{}".format(resp.status_code, resp.text)
                )

        except requests.exceptions.RequestException as e:
            self.error_signal.emit(
                "Network Error: Could not reach Jetson Nano at {}.\n{}".format(
                    self.api_url,
                    str(e)
                )
            )


class HardwareWorker(QThread):
    data_updated = pyqtSignal(dict)
    def run(self):
        parser = TegrastatsParser()
        ssh_command = ["ssh", "jetson-nano@10.42.0.51", "sudo", "tegrastats"]
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

class EWCWorkerThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)

    def __init__(self, patient_id, old_pred, new_label):
        super().__init__()
        self.patient_id = patient_id
        self.old_pred = old_pred
        self.new_label = new_label
        self.api_url = "http://10.42.0.51:5000/correct"

    def run(self):
        payload = {
            "patient_id": self.patient_id,
            "old_prediction": self.old_pred,
            "new_label": self.new_label
        }
        
        try:
            resp = requests.post(self.api_url, json=payload, timeout=5)
            
            if resp.status_code == 200:
                for i in range(101):
                    time.sleep(0.12)
                    self.progress.emit(i)
                
                self.finished.emit(True, "EWC Training and TRT Engine Recompilation Successful.")
            else:
                self.finished.emit(False, f"Jetson API Error: HTTP {resp.status_code}")
                
        except requests.exceptions.RequestException as e:
            self.finished.emit(False, f"Network Error: Could not reach Jetson Nano.\n{str(e)}")

class CameraStreamThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    error_signal = pyqtSignal(str)

    def run(self):
        stream_url = "http://10.42.0.51:5000/stream"
        try:
            cap = cv2.VideoCapture(stream_url)
            while True:
                ret, cv_img = cap.read()
                if ret:
                    # Convert OpenCV BGR format to PyQt RGB format
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    h, w, ch = cv_img.shape
                    bytes_per_line = ch * w
                    qt_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.change_pixmap_signal.emit(qt_img)
                else:
                    time.sleep(0.1) # Wait for stream to initialize
        except Exception as e:
            self.error_signal.emit(str(e))
# =====================================================================
# SPLASH SCREEN
# =====================================================================
class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setFixedSize(600, 420) 
        self.setStyleSheet(f"background-color: {C_BG_PRIMARY}; border: 2px solid {C_ACCENT}; border-radius: 12px;")
        
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        
        # --- ROBUST LOGO LOADER ---
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setStyleSheet("border: none; background: transparent;")
        
        # Try finding the image in src/gui first, then fallback to the root folder
        base_dir = os.path.dirname(__file__)
        logo_path = os.path.join(base_dir, 'uet_logo.png')
        if not os.path.exists(logo_path):
            logo_path = os.path.join(base_dir, '../../uet_logo.png')
            
        pixmap = QPixmap(logo_path)
        if not pixmap.isNull():
            self.logo_label.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.logo_label.setText("[ UET LOGO MISSING ]\nSave as 'uet_logo.png'")
            self.logo_label.setStyleSheet(f"color: {C_WARNING}; border: 1px dashed {C_WARNING}; font-size: 10px;")
            self.logo_label.setFixedSize(120, 80)
        # --------------------------
        
        t = QLabel("MEDICAL AI SERVER")
        t.setFont(QFont("Segoe UI", 26, QFont.Bold))
        t.setStyleSheet(f"color: {C_ACCENT}; border: none;")
        t.setAlignment(Qt.AlignCenter)
        
        sub = QLabel("ACCELERATED MEDICAL IMAGING USING ML ARCHITECTURES \n Supervised by Dr. NAVEED NAWAZ \n Co-Supervised by Dr. SULEMAN QAZI \nUET Lahore - Final Year Project")        
        sub.setFont(QFont("Segoe UI", 10))
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet(f"color: {C_TEXT_SEC}; border: none;")
        
        self.info = QLabel("Initializing TensorRT Engines...")
        self.info.setAlignment(Qt.AlignCenter)
        self.info.setStyleSheet("border: none; margin-top: 30px; font-weight: bold;")
        
        self.prog = QProgressBar()
        self.prog.setRange(0, 100)
        self.prog.setTextVisible(False)
        self.prog.setFixedHeight(4)
        self.prog.setStyleSheet(f"border: none; background: {C_BORDER};")
        
        lay.addWidget(self.logo_label)
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
        self.build_tab8_validation()
        self.build_tab9_billing()
        self.build_tab10_analytics()

        self.build_statusbar()

        self.hw_thread = HardwareWorker()
        self.hw_thread.data_updated.connect(self.update_hw_ui)
        self.hw_thread.start()

        self.pulse_state = False
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self.do_image_pulse)
        
        self.blink_state = False
        self.blink_count = 0
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.do_alert_blink)

    def start_camera_stream(self):
        self.btn_start_stream.setText("Connecting to Jetson...")
        self.btn_start_stream.setEnabled(False)
        self.cam_thread = CameraStreamThread()
        self.cam_thread.change_pixmap_signal.connect(self.update_camera_feed)
        self.cam_thread.error_signal.connect(lambda e: self.cam_lbl.setText(f"[ STREAM ERROR ]\n{e}"))
        self.cam_thread.start()

    def update_camera_feed(self, qt_img):
        self.btn_start_stream.setText("Stream Active")
        self.btn_start_stream.setStyleSheet(f"background: {C_SUCCESS}; color: black;")
        scaled_img = qt_img.scaled(self.cam_lbl.width(), self.cam_lbl.height(), Qt.KeepAspectRatio)
        self.cam_lbl.setPixmap(QPixmap.fromImage(scaled_img))
    
    # --- TAB 10: ANALYTICS DASHBOARD ---
    def build_tab10_analytics(self):
        lay = QVBoxLayout()
        lay.addWidget(self._simulation_banner())
        lay.setContentsMargins(15, 15, 15, 15)
        lay.setSpacing(15)

        # Top Summary Metrics
        hc = QHBoxLayout(); hc.setSpacing(10)
        hc.addWidget(self._create_card("Total Scans Today", "1,204", C_SUCCESS, 80))
        hc.addWidget(self._create_card("Avg Inference Latency", "42 ms", C_ACCENT, 80))
        hc.addWidget(self._create_card("Network Uptime", "99.99%", C_WARNING, 80))
        lay.addLayout(hc)

        # Live Real-Time Chart
        chart_box = QFrame()
        chart_box.setProperty('class', 'Card')
        chart_lay = QVBoxLayout(chart_box)
        lbl_title = QLabel("<b>Live Edge Server Performance (Scans per Minute vs Latency)</b>")
        lbl_title.setStyleSheet("font-size: 16px;")
        chart_lay.addWidget(lbl_title)

        self.plt_analytics = pg.PlotWidget()
        self.plt_analytics.setMinimumHeight(300)
        self.plt_analytics.setYRange(0, 100)
        self.plt_analytics.showGrid(x=True, y=True, alpha=0.2)
        
        # Two data streams for the live chart
        self.hist_latency = np.full(100, 42.0)
        self.hist_throughput = np.full(100, 20.0)
        
        self.crv_latency = self.plt_analytics.plot(pen=pg.mkPen(color=C_CRITICAL, width=2), name="Latency (ms)")
        self.crv_throughput = self.plt_analytics.plot(pen=pg.mkPen(color=C_ACCENT, width=2), name="Throughput (Scans/Min)")
        
        # Add legend
        self.plt_analytics.addLegend()
        
        chart_lay.addWidget(self.plt_analytics)
        lay.addWidget(chart_box)

        # Modality Distribution (Mock Progress Bars)
        dist_box = QGroupBox("Modality Workload Distribution (24h)")
        dist_box.setStyleSheet(f"QGroupBox {{ color: {C_ACCENT}; font-weight: bold; border: 1px solid {C_BORDER}; margin-top: 10px; }}")
        dist_lay = QVBoxLayout()
        
        for modality, percentage in [("Chest X-Ray", 45), ("Brain MRI", 25), ("Skin Lesion", 15), ("ECG", 10), ("Retinal", 5)]:
            row_lay = QHBoxLayout()
            row_lay.addWidget(QLabel(modality), stretch=1)
            bar = QProgressBar()
            bar.setValue(percentage)
            bar.setFixedHeight(8)
            bar.setTextVisible(False)
            bar.setStyleSheet(f"QProgressBar {{ background: {C_BG_PRIMARY}; border: none; }} QProgressBar::chunk {{ background: {C_SUCCESS}; border-radius: 4px; }}")
            row_lay.addWidget(bar, stretch=4)
            row_lay.addWidget(QLabel(f"{percentage}%"), stretch=1)
            dist_lay.addLayout(row_lay)
            
        dist_box.setLayout(dist_lay)
        lay.addWidget(dist_box)

        lay.setAlignment(Qt.AlignTop)
        self.tabs.addTab(self._wrap_scroll(lay), "Analytics")

        # Timer to update the live analytics chart
        self.analytics_timer = QTimer()
        self.analytics_timer.timeout.connect(self.update_analytics_chart)
        self.analytics_timer.start(1000) # Update every 1 second

    def update_analytics_chart(self):
        # Shift data arrays to the left
        self.hist_latency[:-1] = self.hist_latency[1:]
        self.hist_throughput[:-1] = self.hist_throughput[1:]
        
        # Inject new random live data (Simulating Edge Server Load)
        self.hist_latency[-1] = max(30, min(80, self.hist_latency[-2] + random.uniform(-5.0, 5.0)))
        self.hist_throughput[-1] = max(10, min(50, self.hist_throughput[-2] + random.uniform(-2.0, 2.0)))
        
        self.crv_latency.setData(self.hist_latency)
        self.crv_throughput.setData(self.hist_throughput)

    def export_fhir(self):
            self.btn_fhir.setText("Pushing to Server...")
            self.btn_fhir.setEnabled(False)
            QApplication.processEvents()
            time.sleep(0.8) # Simulate network latency to EMR server
            
            # Generate valid HL7 FHIR v4.0 JSON Payload
            fhir_payload = """{
    "resourceType": "DiagnosticReport",
    "status": "final",
    "code": {
        "coding": [{
        "system": "http://snomed.info/sct",
        "code": "126952004",
        "display": "Glioma"
        }]
    },
    "subject": { "reference": "Patient/PAT-001" },
    "effectiveDateTime": "%s",
    "conclusion": "High-grade Glioma detected. Bayesian uncertainty ±1.2%%.",
    "presentedForm": [{
        "contentType": "application/pdf",
        "title": "MEDAI Auto-Generated AI Report"
    }]
    }""" % datetime.datetime.now().isoformat()

            QMessageBox.information(
                self, 
                "FHIR Export Successful", 
                f"Successfully pushed DiagnosticReport to Hospital EMR (Epic/Cerner) via REST API.\n\nPayload Preview:\n{fhir_payload}"
            )
            
            self.btn_fhir.setText("Push to EMR (HL7 FHIR)")
            self.btn_fhir.setEnabled(True)

    def build_tab8_validation(self):
            lay = QVBoxLayout()
            lay.addWidget(self._simulation_banner())
            lay.setContentsMargins(15, 15, 15, 15)
            lay.setSpacing(15)

            # 1. Top Summary Cards
            hc = QHBoxLayout(); hc.setSpacing(10)
            hc.addWidget(self._create_card("Mean AUC-ROC", "0.954", C_SUCCESS, 80))
            hc.addWidget(self._create_card("System Specificity", "96.2%", C_ACCENT, 80))
            hc.addWidget(self._create_card("False Positive Rate", "3.8%", C_WARNING, 80))
            lay.addLayout(hc)

            # 2. Matplotlib ROC Chart Container
            chart_box = QFrame()
            chart_box.setProperty('class', 'Card')
            chart_lay = QVBoxLayout(chart_box)
            lbl_title = QLabel("<b>Multi-Modal Receiver Operating Characteristic (ROC)</b>")
            lbl_title.setStyleSheet("font-size: 16px;")
            chart_lay.addWidget(lbl_title)

            self.roc_canvas = MplCanvas(self, width=6, height=4, dpi=100)
            chart_lay.addWidget(self.roc_canvas)

            # 3. Generate Mathematical ROC Curves
            fpr = np.linspace(0, 1, 100)
            mods = [
                ("Brain MRI (Glioma)", 0.98, C_CRITICAL),
                ("Chest X-Ray", 0.92, C_SUCCESS),
                ("Retinal Fundus", 0.95, C_WARNING),
                ("ECG Signal", 0.96, C_ACCENT)
            ]

            for name, auc, color in mods:
                # Mathematical generation of smooth ROC curve based on target AUC
                k = auc / (1.001 - auc)
                tpr = 1 - (1 - fpr)**k
                self.roc_canvas.axes.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})', color=color, linewidth=2)
                
                # Annotate optimal clinical threshold (Youden's J statistic approximation)
                optimal_idx = np.argmax(tpr - fpr)
                self.roc_canvas.axes.plot(fpr[optimal_idx], tpr[optimal_idx], marker='o', markersize=6, color="white")
                self.roc_canvas.axes.annotate(f'Op. Thresh', 
                                            (fpr[optimal_idx], tpr[optimal_idx]), 
                                            textcoords="offset points", 
                                            xytext=(10,-10), 
                                            ha='left', color=color, fontsize=8)

            # Add random guess line and styling
            self.roc_canvas.axes.plot([0, 1], [0, 1], linestyle='--', color=C_TEXT_SEC) 
            self.roc_canvas.axes.set_xlabel('False Positive Rate (1 - Specificity)')
            self.roc_canvas.axes.set_ylabel('True Positive Rate (Sensitivity)')
            self.roc_canvas.axes.grid(True, color=C_BORDER, linestyle='--', alpha=0.6)
            
            # Style the legend to match dark mode
            legend = self.roc_canvas.axes.legend(facecolor='#141B26', edgecolor=C_BORDER, labelcolor=C_TEXT_PRI)
            
            lay.addWidget(chart_box)

            # 4. Clinical Metrics Table
            tbl = QTableWidget(4, 5)
            tbl.setHorizontalHeaderLabels(["Modality", "Sensitivity", "Specificity", "PPV", "NPV"])
            tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            tbl.verticalHeader().setVisible(False)
            tbl.setAlternatingRowColors(True)
            tbl.setMinimumHeight(170)

            metrics = [
                ("Brain MRI", "97.5%", "98.1%", "96.0%", "98.8%"),
                ("Chest X-Ray", "91.2%", "93.0%", "89.5%", "94.2%"),
                ("Retinal Fundus", "94.0%", "96.5%", "93.1%", "97.0%"),
                ("ECG Signal", "95.8%", "97.0%", "94.5%", "97.8%")
            ]

            for r, row in enumerate(metrics):
                for c, val in enumerate(row):
                    item = QTableWidgetItem(val)
                    item.setTextAlignment(Qt.AlignCenter)
                    tbl.setItem(r, c, item)

            lay.addWidget(tbl)
            
            lay.setAlignment(Qt.AlignTop)
            
            # Add the completed tab to your main interface
            self.tabs.addTab(self._wrap_scroll(lay), "Validation")

    def generate_pdf_report(self):
        try:
            if not hasattr(self, "last_result") or not isinstance(self.last_result, dict):
                QMessageBox.warning(
                    self,
                    "No Analysis Result",
                    "Run a live scan analysis first. PDF reports require a real Jetson result."
                )
                return

            result = self.last_result
            pred_obj = result.get("prediction", {})
            if not isinstance(pred_obj, dict):
                pred_obj = {}

            patient_id = result.get("patient_id")
            if not patient_id:
                if hasattr(self, "input_patient_id"):
                    patient_id = self.input_patient_id.text().strip()
            if not patient_id:
                patient_id = "PT-M4-GUI-001"

            record_id = result.get("record_id", "NA")
            modality = result.get("modality", "unknown")
            prediction = pred_obj.get("class") or result.get("prediction") or result.get("predicted_class") or "UNKNOWN"

            if result.get("modality_mismatch"):
                prediction = "REJECTED: MODALITY MISMATCH"
            elif result.get("ood_flag") or result.get("status") == "rejected":
                prediction = "REJECTED: OOD / MANUAL REVIEW"

            try:
                confidence = float(pred_obj.get("mean_confidence", result.get("confidence", 0.0)) or 0.0)
            except Exception:
                confidence = 0.0

            routing_tier_num = result.get("routing_tier", "N/A")
            routing_decision = result.get("routing_decision", "Tier {}".format(routing_tier_num))
            llm_text = result.get("llm_report") or result.get("phi2_report") or "No clinical report generated."
            gradcam_b64 = result.get("gradcam_base64")

            try:
                uncertainty = float(pred_obj.get("uncertainty", result.get("uncertainty", 0.0)) or 0.0)
            except Exception:
                uncertainty = 0.0

            def localized_report(lang_name):
                if lang_name == "Urdu":
                    return (
                        "طبی خلاصہ: مصنوعی ذہانت کے ایج نظام نے مریض {patient} کے {modality} اسکین کا تجزیہ کیا۔ "
                        "نتیجہ {prediction} ظاہر ہوا، اعتماد {confidence:.1f} فیصد ہے، اور غیر یقینی ±{uncertainty:.2f} فیصد ہے۔ "
                        "کلینیکل روٹنگ: {routing}. یہ رپورٹ AI کی مدد سے تیار کی گئی ہے اور ڈاکٹر/ریڈیالوجسٹ سے تصدیق ضروری ہے۔"
                    ).format(
                        patient=patient_id,
                        modality=modality,
                        prediction=prediction,
                        confidence=confidence,
                        uncertainty=uncertainty,
                        routing=routing_decision
                    )

                if lang_name == "Arabic":
                    return (
                        "الملخص السريري: قام نظام الذكاء الاصطناعي على جهاز الحافة بتحليل فحص {modality} للمريض {patient}. "
                        "كانت النتيجة {prediction}، بنسبة ثقة {confidence:.1f}%، مع عدم يقين ±{uncertainty:.2f}%. "
                        "قرار التوجيه السريري: {routing}. هذا التقرير مدعوم بالذكاء الاصطناعي ويجب مراجعته من قبل الطبيب أو أخصائي الأشعة."
                    ).format(
                        patient=patient_id,
                        modality=modality,
                        prediction=prediction,
                        confidence=confidence,
                        uncertainty=uncertainty,
                        routing=routing_decision
                    )

                return llm_text

            selected_langs = []
            if hasattr(self, "chk_en") and self.chk_en.isChecked():
                selected_langs.append("English")
            if hasattr(self, "chk_ur") and self.chk_ur.isChecked():
                selected_langs.append("Urdu")
            if hasattr(self, "chk_ar") and self.chk_ar.isChecked():
                selected_langs.append("Arabic")

            if not selected_langs:
                selected_langs = ["English"]

            reports_dir = os.path.join(os.getcwd(), "reports")
            os.makedirs(reports_dir, exist_ok=True)

            safe_patient = str(patient_id).replace("/", "_").replace(" ", "_")
            base_filename = "Report_{}_REC{}_{}".format(
                safe_patient,
                record_id,
                int(time.time())
            )

            generated_paths = []

            for lang in selected_langs:
                generator = MedicalReportGenerator()
                out_path = os.path.join(
                    reports_dir,
                    "{}_{}.pdf".format(base_filename, lang.replace(" ", "_"))
                )

                patient_data = {
                    "patient_id": "{} | Record ID: {}".format(patient_id, record_id),
                    "modality": modality,
                    "prediction": prediction,
                    "confidence": "{:.1f}".format(confidence),
                    "routing_tier": routing_decision,
                    "llm_text": localized_report(lang),
                    "gradcam_base64": gradcam_b64,
                }

                generator.generate_report(patient_data, out_path, lang=lang)
                generated_paths.append(out_path)

            self.lbl_pdf_status.setText(
                "✓ Real Jetson PDF generated:\n" + "\n".join(generated_paths)
            )
            self.lbl_pdf_status.setStyleSheet("color: {}; font-family: Consolas;".format(C_SUCCESS))

            QMessageBox.information(
                self,
                "PDF Reports Generated",
                "Generated {} real report(s) from latest Jetson result.".format(len(generated_paths))
            )

        except Exception as e:
            self.lbl_pdf_status.setText("PDF generation failed: {}".format(str(e)))
            self.lbl_pdf_status.setStyleSheet("color: {};".format(C_CRITICAL))
            QMessageBox.critical(self, "PDF Error", str(e))


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
            
            right = QVBoxLayout()
            right.setSpacing(8)
            right.setAlignment(Qt.AlignTop)
            
            grid = QGridLayout()
            grid.setSpacing(8)
            self.mod_cards = []
            self.selected_modality = "brain_mri"

            mods = [
                ("Brain MRI", "brain_mri", 0, 0),
                ("Chest X-Ray", "chest_xray", 0, 1),
                ("Retinal", "retinal", 1, 0),
                ("ECG", "ecg", 1, 1),
                ("Skin", "skin", 2, 0),
                ("Bone", "bone", 2, 1)
            ]

            for n, key, r, c in mods:
                mc = ModalityCard(n)
                mc.modality_key = key
                mc.mousePressEvent = lambda event, card=mc: self.select_modality_card(card)

                if key == "brain_mri":
                    mc.is_selected = True
                    mc.update_style()

                self.mod_cards.append(mc)
                grid.addWidget(mc, r, c)
            
            btn_up = QPushButton("Drop DICOM here or click to browse")
            btn_up.setMinimumHeight(50)
            btn_up.setStyleSheet(f"background: {C_CARD_BG}; border: 1px dashed {C_ACCENT}; color: {C_ACCENT};")
            btn_up.clicked.connect(self.upload_and_analyze_scan)
            
            btn_an = QPushButton("Run Pipeline (Analyze)")
            btn_an.setMinimumHeight(50)
            btn_an.setStyleSheet(f"background: {C_ACCENT}; color: #000; font-size: 12pt; font-weight: bold;")
            self.btn_an = btn_an
            btn_an.clicked.connect(self.upload_and_analyze_scan)
            
            self.res_class = self._create_card("Predicted Class:", "Pending...", min_h=70)
            self.res_unc = self._create_card("MC Uncertainty:", "--", min_h=70)
            self.res_tier = self._create_card("Routing Tier:", "--", min_h=70)
            self.res_ood = self._create_card("OOD Status:", "--", min_h=70)
            
            self.typewriter = TypewriterTextEdit()
            
            # --- NEW PHASE 5 UI: LANGUAGES & EMR EXPORT ---
            lang_layout = QHBoxLayout()
            
            chk_vbox = QVBoxLayout()
            chk_vbox.setSpacing(2)
            self.chk_en = QCheckBox("English")
            self.chk_en.setChecked(True) 
            self.chk_ur = QCheckBox("Urdu (اردو)")
            self.chk_ar = QCheckBox("Arabic (العربية)")
            
            chk_vbox.addWidget(self.chk_en)
            chk_vbox.addWidget(self.chk_ur)
            chk_vbox.addWidget(self.chk_ar)
            
            export_vbox = QVBoxLayout()
            export_vbox.setSpacing(5)
            
            btn_pdf = QPushButton("Generate PDF Report(s)")
            btn_pdf.setProperty('class', 'Primary')
            btn_pdf.setMinimumHeight(44)
            btn_pdf.clicked.connect(self.generate_pdf_report)
            
            self.btn_fhir = QPushButton("Push to EMR (HL7 FHIR)")
            self.btn_fhir.setProperty('class', 'Success')
            self.btn_fhir.setMinimumHeight(44)
            self.btn_fhir.clicked.connect(self.export_fhir)
            
            export_vbox.addWidget(btn_pdf)
            export_vbox.addWidget(self.btn_fhir)
            
            lang_layout.addLayout(chk_vbox, stretch=1)
            lang_layout.addLayout(export_vbox, stretch=2)
            
            self.lbl_pdf_status = QLabel("")
            self.lbl_pdf_status.setWordWrap(True)
            self.lbl_pdf_status.setMinimumHeight(40)
            # -------------------------------------------------

            patient_row = QHBoxLayout()
            patient_row.addWidget(QLabel("Patient ID:"))
            self.input_patient_id = QLineEdit("PT-GUI-001")
            self.input_patient_id.setPlaceholderText("e.g., PT-GUI-001")
            patient_row.addWidget(self.input_patient_id)

            self.combo_modality = QComboBox()
            self.combo_modality.addItem("Brain MRI", "brain_mri")
            self.combo_modality.addItem("Chest X-Ray", "chest_xray")
            self.combo_modality.addItem("Bone X-Ray", "bone")
            self.combo_modality.addItem("ECG", "ecg")
            self.combo_modality.addItem("Retinal Fundus", "retinal")
            self.combo_modality.addItem("Skin Lesion", "skin")

            right.addLayout(patient_row)
            right.addWidget(QLabel("Backend Modality:"))
            right.addWidget(self.combo_modality)
            right.addWidget(QLabel("Select Modality:"))
            right.addLayout(grid)
            right.addWidget(btn_up)
            right.addWidget(btn_an)
            right.addWidget(self.res_class)
            right.addWidget(self.res_unc)
            right.addWidget(self.res_tier)
            right.addWidget(self.res_ood)
            right.addWidget(QLabel("Phi-2 Clinical Report:"))
            right.addWidget(self.typewriter)
            right.addLayout(lang_layout)
            right.addWidget(self.lbl_pdf_status) 
            
            lay.addLayout(left, stretch=6)
            lay.addLayout(right, stretch=4)
            self.tabs.addTab(self._wrap_scroll(lay), "Scan")


    def select_modality_card(self, selected_card):
        for card in self.mod_cards:
            card.is_selected = (card is selected_card)
            card.update_style()

        self.selected_modality = getattr(selected_card, "modality_key", "brain_mri")

    def _set_card_value(self, card, text, color=None):
        value_widget = card.layout().itemAt(1).widget()
        value_widget.setText(str(text))

        if color:
            value_widget.setStyleSheet(
                f"color: {color}; font-size: 11pt; font-weight: bold;"
            )
            card.setStyleSheet(
                f"QFrame {{ background: {C_CARD_BG}; border: 1px solid {C_BORDER}; border-left: 3px solid {color}; }}"
            )

    def _display_base64_image(self, b64_text):
        try:
            raw = base64.b64decode(b64_text)
            pix = QPixmap()
            if pix.loadFromData(raw):
                self.current_gradcam_pixmap = pix
                scaled = pix.scaled(
                    self.img_view.width(),
                    self.img_view.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.img_view.setPixmap(scaled)
                return True
        except Exception as e:
            print("GradCAM display error:", e)
        return False


    def on_analysis_complete(self, data):
        self.last_result = data
        self.pulse_timer.stop()

        self.img_view.setStyleSheet(
            "background: #000; border: 2px solid {}; border-radius: 12px; color: {};".format(
                C_SUCCESS,
                C_TEXT_PRI
            )
        )

        pred_obj = data.get("prediction", {})
        if not isinstance(pred_obj, dict):
            pred_obj = {}

        pred = pred_obj.get("class") or data.get("prediction") or data.get("predicted_class") or "UNKNOWN"

        try:
            conf_val = float(pred_obj.get("mean_confidence", data.get("confidence", 0.0)) or 0.0)
        except Exception:
            conf_val = 0.0

        try:
            unc_val = float(pred_obj.get("uncertainty", data.get("uncertainty", 0.0)) or 0.0)
        except Exception:
            unc_val = 0.0

        try:
            tier_num = int(data.get("routing_tier", 3))
        except Exception:
            tier_num = 3

        patient_id = data.get("patient_id", "PT-M4-GUI-001")
        record_id = data.get("record_id", "N/A")
        modality = data.get("modality", "unknown")
        report = data.get("llm_report") or data.get("phi2_report") or "No report generated."
        routing_decision = data.get("routing_decision", "Tier {}".format(tier_num))
        risk_tier = data.get("risk_tier", "N/A")
        gpio = data.get("gpio_alert_type", "NONE")

        self.current_patient_id = patient_id

        self.conf_bar.set_value(conf_val)
        self.conf_lbl.setText("{:.1f}%".format(conf_val))

        if tier_num == 1:
            color = C_SUCCESS
            risk_level = 1
        elif tier_num == 2:
            color = C_WARNING
            risk_level = 2
        else:
            color = C_CRITICAL
            risk_level = 3

        try:
            self.res_class.layout().itemAt(1).widget().setText(
                "{}\nRecord ID: {} | Patient: {}\nModality: {}".format(
                    pred,
                    record_id,
                    patient_id,
                    modality
                )
            )
            self.res_class.layout().itemAt(1).widget().setStyleSheet(
                "color: {}; font-size: 11pt; font-weight: bold;".format(color)
            )
            self.res_class.setStyleSheet(
                "QFrame {{ background: {}; border: 1px solid {}; border-left: 3px solid {}; }}".format(
                    C_CARD_BG,
                    C_BORDER,
                    color
                )
            )

            self.res_unc.layout().itemAt(1).widget().setText("± {:.2f}%".format(unc_val))
            self.res_unc.layout().itemAt(1).widget().setStyleSheet(
                "color: {}; font-size: 14pt; font-weight: bold;".format(C_TEXT_PRI)
            )

            self.res_tier.layout().itemAt(1).widget().setText(
                "{}\n{}\nGPIO: {}".format(routing_decision, risk_tier, gpio)
            )
            self.res_tier.layout().itemAt(1).widget().setStyleSheet(
                "color: {}; font-size: 11pt; font-weight: bold;".format(color)
            )
            self.res_tier.setStyleSheet(
                "QFrame {{ background: {}; border: 1px solid {}; border-left: 3px solid {}; }}".format(
                    C_CARD_BG,
                    C_BORDER,
                    color
                )
            )

            if data.get("modality_mismatch"):
                ood_text = "MODALITY MISMATCH\nDetected: {}".format(data.get("detected_modality", "unknown"))
                ood_color = C_CRITICAL
            elif data.get("ood_flag") or data.get("status") == "rejected":
                ood_text = "REJECTED / OOD"
                ood_color = C_CRITICAL
            else:
                ood_text = "CLEAR"
                ood_color = C_SUCCESS

            self.res_ood.layout().itemAt(1).widget().setText(ood_text)
            self.res_ood.layout().itemAt(1).widget().setStyleSheet(
                "color: {}; font-size: 12pt; font-weight: bold;".format(ood_color)
            )
            self.res_ood.setStyleSheet(
                "QFrame {{ background: {}; border: 1px solid {}; border-left: 3px solid {}; }}".format(
                    C_CARD_BG,
                    C_BORDER,
                    ood_color
                )
            )

        except Exception as e:
            print("Result card update error:", e)

        if hasattr(self, "traffic_light"):
            self.traffic_light.set_risk_level(risk_level)
            self.lbl_risk_title.setText(routing_decision)
            self.lbl_risk_title.setStyleSheet(
                "font-size: 24px; font-weight: bold; color: {};".format(color)
            )
            self.lbl_risk_desc.setText(
                data.get("routing_reason", "Analysis complete using latest Jetson API routing.")
            )

        if risk_level == 3:
            self.blink_count = 0
            self.blink_timer.start(800)

        gradcam_b64 = data.get("gradcam_base64")
        if gradcam_b64:
            try:
                raw = base64.b64decode(gradcam_b64)
                pix = QPixmap()
                if pix.loadFromData(raw):
                    self.img_view.setPixmap(
                        pix.scaled(
                            self.img_view.width(),
                            self.img_view.height(),
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation
                        )
                    )
            except Exception as e:
                print("GradCAM display error:", e)

        # Final visible override for modality mismatch / rejected wrong-modality cases.
        # This runs after normal card updates so the GUI cannot misleadingly show Brain MRI diagnosis
        # when backend has blocked the classifier.
        if data.get("modality_mismatch") or pred == "MODALITY_MISMATCH":
            selected_modality = data.get("selected_modality", data.get("modality", "unknown"))
            detected_modality = data.get("detected_modality", "unknown")

            mismatch_text = (
                "REJECTED: WRONG MODALITY\n"
                "Selected: {} | Detected: {}\n"
                "Classifier blocked | Record ID: {}"
            ).format(selected_modality, detected_modality, record_id)

            try:
                self.res_class.layout().itemAt(1).widget().setText(mismatch_text)
                self.res_class.layout().itemAt(1).widget().setStyleSheet(
                    "color: {}; font-size: 11pt; font-weight: bold;".format(C_CRITICAL)
                )
                self.res_class.setStyleSheet(
                    "QFrame {{ background: {}; border: 1px solid {}; border-left: 3px solid {}; }}".format(
                        C_CARD_BG,
                        C_BORDER,
                        C_CRITICAL
                    )
                )

                self.res_tier.layout().itemAt(1).widget().setText(
                    "Tier 3: Modality Mismatch / Manual Review\nRejected / Wrong Modality\nGPIO: RED"
                )
                self.res_tier.layout().itemAt(1).widget().setStyleSheet(
                    "color: {}; font-size: 11pt; font-weight: bold;".format(C_CRITICAL)
                )
                self.res_tier.setStyleSheet(
                    "QFrame {{ background: {}; border: 1px solid {}; border-left: 3px solid {}; }}".format(
                        C_CARD_BG,
                        C_BORDER,
                        C_CRITICAL
                    )
                )

                self.res_ood.layout().itemAt(1).widget().setText(
                    "MODALITY MISMATCH\nSelected: {}\nDetected: {}\nClassifier blocked".format(
                        selected_modality,
                        detected_modality
                    )
                )
                self.res_ood.layout().itemAt(1).widget().setStyleSheet(
                    "color: {}; font-size: 11pt; font-weight: bold;".format(C_CRITICAL)
                )
                self.res_ood.setStyleSheet(
                    "QFrame {{ background: {}; border: 1px solid {}; border-left: 3px solid {}; }}".format(
                        C_CARD_BG,
                        C_BORDER,
                        C_CRITICAL
                    )
                )

                self.conf_bar.set_value(0)
                self.conf_lbl.setText("0.0%")

            except Exception as e:
                print("Mismatch override display error:", e)


        self.typewriter.start_typing(report)


    def on_analysis_error(self, err_msg):
        self.pulse_timer.stop()
        self.res_class.layout().itemAt(1).widget().setText("CONNECTION FAILED")
        self.res_class.layout().itemAt(1).widget().setStyleSheet(f"color: {C_CRITICAL}; font-size: 11pt; font-weight: bold;")
        self.typewriter.setText(err_msg)
        QMessageBox.critical(self, "Jetson Connection Error", err_msg)
            
    def export_fhir(self):
        if not hasattr(self, 'current_patient_id'):
            QMessageBox.warning(self, "No Patient", "Please analyze a scan first before exporting to EMR.")
            return

        self.btn_fhir.setText("Pushing to Server...")
        self.btn_fhir.setEnabled(False)
        QApplication.processEvents()
        
        try:
            # Tell the Jetson to execute fhir_bridge.py logic for this patient
            resp = requests.post(f"http://10.42.0.51:5000/fhir_export", json={"patient_id": self.current_patient_id}, timeout=10)
            
            if resp.status_code in [200, 201]:
                QMessageBox.information(self, "FHIR Export Successful", f"Jetson successfully pushed DiagnosticReport to Hospital EMR.\n\nServer Response: {resp.status_code}")
            else:
                QMessageBox.warning(self, "FHIR Export Failed", f"Jetson returned error code {resp.status_code}:\n{resp.text}")
                
        except Exception as e:
            QMessageBox.critical(self, "Network Error", f"Could not reach Jetson Nano.\n{str(e)}")
            
        self.btn_fhir.setText("Push to EMR (HL7 FHIR)")
        self.btn_fhir.setEnabled(True)

    def do_image_pulse(self):
        self.pulse_state = not self.pulse_state
        bc = C_ACCENT if self.pulse_state else C_BORDER
        self.img_view.setStyleSheet(f"background: #000; border: 2px solid {bc}; border-radius: 12px; color: {C_TEXT_PRI};")

    def do_alert_blink(self):
        if self.blink_count >= 12: 
            self.blink_timer.stop()
            self.res_tier.setStyleSheet(f"QFrame {{ background: {C_CARD_BG}; border: 1px solid {C_BORDER}; border-left: 3px solid {C_CRITICAL}; }}")
            return
        self.blink_state = not self.blink_state
        bg = "#4A0000" if self.blink_state else C_CARD_BG
        self.res_tier.setStyleSheet(f"QFrame {{ background: {bg}; border: 1px solid {C_CRITICAL}; border-left: 3px solid {C_CRITICAL}; }}")
        self.blink_count += 1

    def trigger_analysis_mock(self):
        self.conf_bar.set_value(98); self.conf_lbl.setText("98%")

        self.res_class.layout().itemAt(1).widget().setText("Glioma (Malignant)\n[SNOMED: 126952004]") 
        self.res_class.layout().itemAt(1).widget().setStyleSheet(f"color: {C_CRITICAL}; font-size: 13pt; font-weight: bold;")
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
        lay.addWidget(self._simulation_banner())
        lay.setContentsMargins(15,15,15,15)
        lay.setSpacing(15)
        
        sh = QHBoxLayout()
        src = QLineEdit(); src.setPlaceholderText("Search Patient ID (e.g. PAT-9042)"); src.setMinimumHeight(40)
        bs = QPushButton("Fetch"); bs.setProperty('class', 'Primary'); bs.setMinimumHeight(40)
        sh.addWidget(src); sh.addWidget(bs)
        lay.addLayout(sh)
        
        lay.addWidget(SemiCircularGauge(), alignment=Qt.AlignCenter)
        
        hc = QHBoxLayout(); hc.setSpacing(8)
        for m, d, w, c in [("Brain MRI", "Glioma", "85", C_CRITICAL), ("Chest X-Ray", "Normal", "10", C_SUCCESS), ("ECG", "Arrhythmia", "5", C_WARNING)]:
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
        
        risk_group = QGroupBox("Clinical Routing and Risk Assessment")
        risk_group.setStyleSheet("QGroupBox { color: #00E5FF; font-weight: bold; border: 1px solid #333; margin-top: 10px; }")
        risk_layout = QHBoxLayout()
        
        self.traffic_light = RiskTrafficLight()
        risk_layout.addWidget(self.traffic_light, alignment=Qt.AlignCenter)
        
        text_panel = QVBoxLayout()
        self.lbl_risk_title = QLabel("Awaiting Scan...")
        self.lbl_risk_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #FFF;")
        
        self.lbl_risk_desc = QLabel("Upload a multi-modal scan sequence to calculate Bayesian uncertainty and clinical routing tier.")
        self.lbl_risk_desc.setWordWrap(True)
        self.lbl_risk_desc.setStyleSheet("color: #AAA; font-size: 14px;")
        
        text_panel.addWidget(self.lbl_risk_title)
        text_panel.addWidget(self.lbl_risk_desc)
        text_panel.addStretch()
        
        risk_layout.addLayout(text_panel)
        risk_layout.setStretch(0, 1)
        risk_layout.setStretch(1, 3)
        
        risk_group.setLayout(risk_layout)
        lay.addWidget(risk_group)
        
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
        patient_id = "PAT-001"                     
        old_pred = "AI Prediction"            
        new_label = self.correction_dropdown.currentText()
        
        self.btn_ewc.setEnabled(False)
        self.btn_ewc.setText("Recompiling TRT Engine...")
        self.prog_ewc.setValue(0)
        
        self.ewc_thread = EWCWorkerThread(patient_id, old_pred, new_label)
        self.ewc_thread.progress.connect(self.prog_ewc.setValue)
        self.ewc_thread.finished.connect(self.on_ewc_finished)
        self.ewc_thread.start()

    def on_ewc_finished(self, success, message):
        self.btn_ewc.setEnabled(True)
        self.btn_ewc.setText("Submit Correction + Run EWC")
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if success:
            self.prog_ewc.setValue(100) 
            QMessageBox.information(self, "Continuous Learning Updated", f"{message}\n\nCorrection logged at: {timestamp}")
        else:
            self.prog_ewc.setValue(0)
            QMessageBox.critical(self, "EWC Update Failed", f"{message}\n\nFailed at: {timestamp}")

# --- TAB 4: CAMERA ---
    def build_tab4_camera(self):
        tab = QWidget()                       
        lay = QHBoxLayout(tab)                
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
        
        # --- NEW WIRED CAMERA BUTTON ---
        self.btn_start_stream = QPushButton("Start Live Stream")
        self.btn_start_stream.setProperty('class', 'Success')
        self.btn_start_stream.setMinimumHeight(48)
        self.btn_start_stream.clicked.connect(self.start_camera_stream)
        
        bc = QPushButton("📸 Capture Frame"); bc.setMinimumHeight(44)
        r.addWidget(self.btn_start_stream); r.addWidget(bc)
        # -------------------------------
        
        r.addStretch()
        
        lay.addWidget(rw, stretch=1)
        self.tabs.addTab(tab, "Camera")     

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
        lay.addWidget(self._simulation_banner())
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
        lay.addWidget(self._simulation_banner())
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

    # --- TAB 8: CLINICAL VALIDATION ---
    def build_tab8_validation(self):
        lay = QVBoxLayout()
        lay.setContentsMargins(15, 15, 15, 15)
        lay.setSpacing(15)

        # 1. Top Summary Cards
        hc = QHBoxLayout(); hc.setSpacing(10)
        hc.addWidget(self._create_card("Mean AUC-ROC", "0.954", C_SUCCESS, 80))
        hc.addWidget(self._create_card("System Specificity", "96.2%", C_ACCENT, 80))
        hc.addWidget(self._create_card("False Positive Rate", "3.8%", C_WARNING, 80))
        lay.addLayout(hc)

        # 2. Matplotlib ROC Chart Container
        chart_box = QFrame()
        chart_box.setProperty('class', 'Card')
        chart_lay = QVBoxLayout(chart_box)
        lbl_title = QLabel("<b>Multi-Modal Receiver Operating Characteristic (ROC)</b>")
        lbl_title.setStyleSheet("font-size: 16px;")
        chart_lay.addWidget(lbl_title)

        self.roc_canvas = MplCanvas(self, width=6, height=4, dpi=100)
        chart_lay.addWidget(self.roc_canvas)

        # 3. Generate Mathematical ROC Curves
        fpr = np.linspace(0, 1, 100)
        mods = [
            ("Brain MRI (Glioma)", 0.98, C_CRITICAL),
            ("Chest X-Ray", 0.92, C_SUCCESS),
            ("Retinal Fundus", 0.95, C_WARNING),
            ("ECG Signal", 0.96, C_ACCENT),
            ("Skin Lesion", 0.91, C_PURPLE),
            ("Bone Fracture", 0.94, "#FF9100") 
        ]

        for name, auc, color in mods:
            # Math trick to generate a smooth curve based on a target AUC value
            k = auc / (1.001 - auc)
            tpr = 1 - (1 - fpr)**k
            self.roc_canvas.axes.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})', color=color, linewidth=2)

        # Add random guess line and styling
        self.roc_canvas.axes.plot([0, 1], [0, 1], linestyle='--', color=C_TEXT_SEC) 
        self.roc_canvas.axes.set_xlabel('False Positive Rate (1 - Specificity)')
        self.roc_canvas.axes.set_ylabel('True Positive Rate (Sensitivity)')
        self.roc_canvas.axes.grid(True, color=C_BORDER, linestyle='--', alpha=0.6)
        
        # Style the legend to match dark mode
        legend = self.roc_canvas.axes.legend(facecolor='#141B26', edgecolor=C_BORDER, labelcolor=C_TEXT_PRI)
        
        lay.addWidget(chart_box)

        # 4. Clinical Metrics Table
        tbl = QTableWidget(6, 5)
        tbl.setHorizontalHeaderLabels(["Modality", "Sensitivity", "Specificity", "PPV", "NPV"])
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tbl.verticalHeader().setVisible(False)
        tbl.setAlternatingRowColors(True)
        tbl.setMinimumHeight(220)

        metrics = [
            ("Brain MRI", "97.5%", "98.1%", "96.0%", "98.8%"),
            ("Chest X-Ray", "91.2%", "93.0%", "89.5%", "94.2%"),
            ("Retinal Fundus", "94.0%", "96.5%", "93.1%", "97.0%"),
            ("ECG Signal", "95.8%", "97.0%", "94.5%", "97.8%"),
            ("Skin Lesion", "90.5%", "92.4%", "88.0%", "93.5%"),
            ("Bone Fracture", "93.2%", "95.1%", "91.8%", "96.0%")
        ]

        for r, row in enumerate(metrics):
            for c, val in enumerate(row):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                tbl.setItem(r, c, item)

        lay.addWidget(tbl)
        
        lay.setAlignment(Qt.AlignTop)
        self.tabs.addTab(self._wrap_scroll(lay), "Validation")

    def _simulation_banner(self, text="Simulation / Research Dashboard - not live patient data"):
        lbl = QLabel("SIMULATION MODE: " + text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(
            "background: #3b2f00; color: #ffcc00; border: 1px solid #ffcc00; "
            "padding: 8px; border-radius: 6px; font-weight: bold;"
        )
        return lbl

    def check_jetson_health(self, silent=False):
        """
        Checks Jetson API before analysis.
        If Jetson is offline, disables Run Pipeline and shows a clear status.
        """
        self.last_health_error = ""

        try:
            r = requests.get("http://10.42.0.51:5000/health", timeout=4)
            data = r.json()

            ok = (
                r.status_code == 200
                and data.get("status") == "ok"
                and data.get("db") == "connected"
            )

            if ok:
                if hasattr(self, "btn_an"):
                    self.btn_an.setEnabled(True)
                    self.btn_an.setText("Run Pipeline (Analyze)")

                if hasattr(self, "lbl_jetson_health"):
                    self.lbl_jetson_health.setText(
                        "Jetson API: ONLINE | DB: {} | Precision: {}".format(
                            data.get("db", "unknown"),
                            data.get("precision", "unknown")
                        )
                    )
                    self.lbl_jetson_health.setStyleSheet(
                        "color: {}; font-weight: bold;".format(C_SUCCESS)
                    )

                return True

            self.last_health_error = "Jetson health returned unexpected response: {}".format(data)

        except Exception as e:
            self.last_health_error = str(e)

        if hasattr(self, "btn_an"):
            self.btn_an.setEnabled(False)
            self.btn_an.setText("Jetson Offline - Cannot Analyze")

        if hasattr(self, "lbl_jetson_health"):
            self.lbl_jetson_health.setText("Jetson API: OFFLINE / ERROR")
            self.lbl_jetson_health.setStyleSheet(
                "color: {}; font-weight: bold;".format(C_CRITICAL)
            )

        if not silent:
            QMessageBox.critical(
                self,
                "Jetson API Offline",
                "Cannot reach Jetson Nano API at http://10.42.0.51:5000/health\n\n"
                "Error: {}\n\n"
                "Start/verify the Jetson Docker container before running analysis.".format(
                    self.last_health_error
                )
            )

        return False

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

        QTimer.singleShot(1000, self.check_jetson_health)

# --- TAB 9: ICD-10 BILLING & ANALYTICS ---
    def build_tab9_billing(self):
        lay = QVBoxLayout()
        lay.addWidget(self._simulation_banner())
        lay.setContentsMargins(15, 15, 15, 15)
        lay.setSpacing(15)

        # 1. Top Financial Summary Cards
        hc = QHBoxLayout(); hc.setSpacing(10)
        hc.addWidget(self._create_card("Scans Processed (24h)", "142", C_ACCENT, 80))
        hc.addWidget(self._create_card("Auto-Coded (ICD-10)", "138", C_SUCCESS, 80))
        hc.addWidget(self._create_card("Admin Hours Saved", "11.5 hrs", C_WARNING, 80))
        lay.addLayout(hc)

        # 2. Daily Billing Ledger Table
        lbl_title = QLabel("<b>Automated ICD-10-CM Daily Ledger</b>")
        lbl_title.setStyleSheet("font-size: 16px; margin-top: 10px;")
        lay.addWidget(lbl_title)

        tbl = QTableWidget(6, 6)
        tbl.setHorizontalHeaderLabels(["Timestamp", "Patient ID", "Modality", "AI Diagnosis", "ICD-10 Code", "Status"])
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tbl.verticalHeader().setVisible(False)
        tbl.setAlternatingRowColors(True)
        tbl.setMinimumHeight(250)

        # Mock billing data mapping AI outputs to CMS 2025 ICD-10 codes
        billing_data = [
            ("09:30 AM", "PAT-001", "Brain MRI", "Glioma", "C71.9", "Exported (FHIR)"),
            ("09:15 AM", "PAT-084", "Chest X-Ray", "Pneumonia", "J18.9", "Exported (FHIR)"),
            ("08:50 AM", "PAT-112", "Skin Lesion", "Melanoma", "C43.9", "Exported (FHIR)"),
            ("08:22 AM", "PAT-045", "Retinal", "Diabetic Retinopathy", "E11.3592", "Exported (FHIR)"),
            ("08:10 AM", "PAT-099", "ECG", "Arrhythmia", "I49.9", "Exported (FHIR)"),
            ("07:55 AM", "PAT-201", "Bone X-Ray", "Fracture (Femur)", "S72.90XA", "Pending Review")
        ]

        for r, row in enumerate(billing_data):
            for c, val in enumerate(row):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                # Highlight the ICD-10 code column
                if c == 4:
                    item.setForeground(QBrush(QColor(C_ACCENT)))
                    item.setFont(QFont("Segoe UI", 10, QFont.Bold))
                # Highlight Status
                if c == 5:
                    color = C_SUCCESS if "Exported" in val else C_WARNING
                    item.setForeground(QBrush(QColor(color)))
                tbl.setItem(r, c, item)

        lay.addWidget(tbl)

        # 3. Export to CSV Button
        btn_csv = QPushButton("Download Daily Billing Report (CSV)")
        btn_csv.setProperty('class', 'Primary')
        btn_csv.setMinimumHeight(44)
        btn_csv.clicked.connect(lambda: QMessageBox.information(self, "Export", "Billing ledger exported to /reports/billing_summary.csv"))
        lay.addWidget(btn_csv, alignment=Qt.AlignRight)

        lay.setAlignment(Qt.AlignTop)
        self.tabs.addTab(self._wrap_scroll(lay), "Billing")

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

    def select_image_file(self):
        """
        Opens file picker and stores selected image path.
        This is separate from analysis so the GUI does not stay stuck at Pending.
        """
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Medical Image",
                os.getcwd(),
                "Medical Images (*.jpg *.jpeg *.png *.bmp *.dcm);;All Files (*)"
            )

            if not file_path:
                return False

            self.current_file_path = file_path
            filename = os.path.basename(file_path)

            pix = QPixmap(file_path)
            if not pix.isNull():
                self.img_view.setPixmap(
                    pix.scaled(
                        self.img_view.width(),
                        self.img_view.height(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                )
            else:
                self.img_view.setText("Loaded:\n{}".format(filename))

            self.img_view.setStyleSheet(
                "background: #000; border: 2px solid {}; border-radius: 12px; color: {};".format(
                    C_ACCENT,
                    C_TEXT_PRI
                )
            )

            try:
                self.res_class.layout().itemAt(1).widget().setText("Ready to analyze")
                self.res_unc.layout().itemAt(1).widget().setText("--")
                self.res_tier.layout().itemAt(1).widget().setText("--")
                self.res_ood.layout().itemAt(1).widget().setText("--")
                self.conf_bar.set_value(0)
                self.conf_lbl.setText("0%")
            except Exception:
                pass

            return True

        except Exception as e:
            QMessageBox.critical(self, "File Selection Error", str(e))
            return False


    def upload_and_analyze_scan(self):
        """
        Full-proof analysis launcher:
        - checks Jetson health
        - opens file picker if no file selected
        - validates patient ID and modality
        - sends real request to Jetson
        """
        try:
            if not self.check_jetson_health(silent=True):
                QMessageBox.critical(
                    self,
                    "Jetson API Offline",
                    "Jetson Nano API is not reachable. Analysis is disabled until the API is online.\n\n"
                    "Error: {}".format(getattr(self, "last_health_error", "Unknown error"))
                )
                return

            if not hasattr(self, "current_file_path") or not self.current_file_path:
                selected = self.select_image_file()
                if not selected:
                    QMessageBox.warning(
                        self,
                        "No Image Selected",
                        "Please select a medical image first."
                    )
                    return

            if not os.path.exists(self.current_file_path):
                QMessageBox.warning(
                    self,
                    "Image Missing",
                    "Selected image file was not found. Please select it again."
                )
                self.current_file_path = None
                return

            patient_id = "PT-M4-GUI-001"
            if hasattr(self, "input_patient_id"):
                patient_id = self.input_patient_id.text().strip() or "PT-M4-GUI-001"

            modality = "brain_mri"
            if hasattr(self, "combo_modality"):
                modality = self.combo_modality.currentData() or "brain_mri"

            selected_lang = "English"

            try:
                self.res_class.layout().itemAt(1).widget().setText("Running...")
                self.res_unc.layout().itemAt(1).widget().setText("--")
                self.res_tier.layout().itemAt(1).widget().setText("--")
                self.res_ood.layout().itemAt(1).widget().setText("--")
            except Exception:
                pass

            if hasattr(self, "btn_an"):
                self.btn_an.setEnabled(False)
                self.btn_an.setText("Analyzing on Jetson...")

            self.analysis_thread = AnalysisWorker(
                self.current_file_path,
                selected_lang,
                modality,
                patient_id
            )
            self.analysis_thread.finished_signal.connect(self.on_analysis_complete)
            self.analysis_thread.error_signal.connect(self.on_analysis_error)
            self.analysis_thread.finished_signal.connect(
                lambda _: self.btn_an.setEnabled(True) if hasattr(self, "btn_an") else None
            )
            self.analysis_thread.finished_signal.connect(
                lambda _: self.btn_an.setText("Run Pipeline (Analyze)") if hasattr(self, "btn_an") else None
            )
            self.analysis_thread.error_signal.connect(
                lambda _: self.btn_an.setEnabled(True) if hasattr(self, "btn_an") else None
            )
            self.analysis_thread.error_signal.connect(
                lambda _: self.btn_an.setText("Run Pipeline (Analyze)") if hasattr(self, "btn_an") else None
            )
            self.analysis_thread.start()

        except Exception as e:
            if hasattr(self, "btn_an"):
                self.btn_an.setEnabled(True)
                self.btn_an.setText("Run Pipeline (Analyze)")
            QMessageBox.critical(self, "Analysis Launch Error", str(e))


    def refresh_alq_table(self):
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