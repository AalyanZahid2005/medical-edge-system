import re
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont

class TegrastatsParser:
    def __init__(self):
        self.cpu_pattern = re.compile(r'CPU \[([^\]]+)\]')
        self.gpu_pattern = re.compile(r'GR3D_FREQ (\d+)%')
        self.temp_pattern = re.compile(r'[a-zA-Z]+@([\d.]+)C') 
        self.power_pattern = re.compile(r'POM_5V_IN (\d+)/\d+')

    def parse_line(self, line: str) -> dict:
        metrics = {"cpu_percent": 0.0, "gpu_percent": 0.0, "temp_c": 0.0, "power_w": 0.0, "error": None}
        try:
            cpu_match = self.cpu_pattern.search(line)
            if cpu_match:
                core_loads = re.findall(r'(\d+)%@', cpu_match.group(1))
                if core_loads:
                    metrics["cpu_percent"] = round(sum(int(load) for load in core_loads) / len(core_loads), 1)

            gpu_match = self.gpu_pattern.search(line)
            if gpu_match:
                metrics["gpu_percent"] = float(gpu_match.group(1))

            temp_matches = self.temp_pattern.findall(line)
            if temp_matches:
                metrics["temp_c"] = round(max(float(temp) for temp in temp_matches), 1)

            power_match = self.power_pattern.search(line)
            if power_match:
                metrics["power_w"] = round(int(power_match.group(1)) / 1000.0, 2)
        except Exception as e:
            metrics["error"] = str(e)
        return metrics

def get_latest_telemetry(parser: TegrastatsParser) -> dict:
    # Mock data until you hook up the live stream from the Jetson
    mock_tegrastats_line = "RAM 1500/3964MB (lfb 1x4MB) CPU [15%@1479,10%@1479,0%@1479,5%@1479] GR3D_FREQ 45% thermal@42.5C POM_5V_IN 4500/4500"
    return parser.parse_line(mock_tegrastats_line)

class HardwareTelemetryWidget(QWidget):
    def __init__(self, parent=None):
        super(HardwareTelemetryWidget, self).__init__(parent)
        self.parser = TegrastatsParser()
        self.init_ui()
        self.start_monitoring()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.group_box = QGroupBox("Jetson Nano Edge Server Telemetry")
        self.group_box.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid gray; border-radius: 5px; margin-top: 1ex; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        
        metrics_layout = QHBoxLayout()
        font = QFont("Arial", 12, QFont.Bold)
        
        self.lbl_cpu = self.create_metric_label("CPU: -- %", font)
        self.lbl_gpu = self.create_metric_label("GPU: -- %", font)
        self.lbl_temp = self.create_metric_label("Temp: -- °C", font)
        self.lbl_power = self.create_metric_label("Power: -- W", font)

        metrics_layout.addWidget(self.lbl_cpu)
        metrics_layout.addWidget(self.lbl_gpu)
        metrics_layout.addWidget(self.lbl_temp)
        metrics_layout.addWidget(self.lbl_power)

        self.group_box.setLayout(metrics_layout)
        layout.addWidget(self.group_box)

    def create_metric_label(self, text, font):
        label = QLabel(text)
        label.setFont(font)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 4px;")
        return label

    def start_monitoring(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_telemetry)
        self.timer.start(1000) 

    def update_telemetry(self):
        metrics = get_latest_telemetry(self.parser)
        if metrics.get("error"):
            self.lbl_cpu.setText("Data Error")
            return

        self.lbl_cpu.setText(f"CPU: {metrics['cpu_percent']}%")
        self.lbl_gpu.setText(f"GPU: {metrics['gpu_percent']}%")
        
        temp = metrics['temp_c']
        self.lbl_temp.setText(f"Temp: {temp}°C")
        if temp > 75.0:
            self.lbl_temp.setStyleSheet("padding: 10px; background-color: #ffcccc; color: red; border-radius: 4px;")
        else:
            self.lbl_temp.setStyleSheet("padding: 10px; background-color: #f0f0f0; color: black; border-radius: 4px;")

        self.lbl_power.setText(f"Power: {metrics['power_w']} W")
