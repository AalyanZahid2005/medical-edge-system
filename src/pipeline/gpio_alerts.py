# Hardware GPIO Stub for Jetson Nano
try:
    import Jetson.GPIO as GPIO
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    print("[HARDWARE] Jetson.GPIO library not found. Running in simulation mode.")

def setup_pins():
    print("[GPIO] Pins initialized. LEDs ready.")
    if HARDWARE_AVAILABLE:
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        # Setup pins here when hardware is attached

def trigger_success_led():
    print("[GPIO] 🟢 GREEN LED ON (High Confidence)")

def trigger_warning_led():
    print("[GPIO] 🟡 YELLOW LED BLINKING (Medium Confidence)")

def trigger_emergency_alarm():
    print("[GPIO] 🔴 RED LED ON + BUZZER ACTIVATED (Critical/OOD)")
