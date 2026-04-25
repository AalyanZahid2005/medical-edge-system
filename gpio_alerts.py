import time
import sys

# Safe import for Windows 11 testing vs Jetson Nano deployment
try:
    import Jetson.GPIO as GPIO
    ON_JETSON = True
except ImportError:
    print("[WARNING] Jetson.GPIO library not found. Running in Windows Simulation Mode.")
    ON_JETSON = False

# ---------------------------------------------------------
# 1. Hardware Configuration
# ---------------------------------------------------------
RED_LED_PIN = 13
GREEN_LED_PIN = 11
BUZZER_PIN = 15

# Updated CRITICAL_FINDINGS with specific MIT-BIH Arrhythmia classes
CRITICAL_FINDINGS = [
    "Glioma", "Meningioma", "Pituitary",          # Brain Tumor
    "1", "2", "3", "4",                           # Retinal Fundus
    "MEL", "BCC", "AKIEC", "BKL", "DF", "VASC",   # Skin Lesion
    "Pneumonia",                                  # Chest X-Ray
    "Fracture",                                   # Bone Fracture
    'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?' # ECG
]

def setup_hardware():
    """Initializes the Jetson Nano GPIO pins."""
    if not ON_JETSON:
        return
        
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup(RED_LED_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(GREEN_LED_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)
    print("[SYSTEM] GPIO Pins Initialized.")

def trigger_alert(predicted_class, uncertainty_percent):
    """Evaluates the model's diagnosis and triggers physical hardware."""
    print(f"\n[INFERENCE] Class: {predicted_class} | Uncertainty: {uncertainty_percent:.2f}%")

    if predicted_class in CRITICAL_FINDINGS or uncertainty_percent > 5.0:
        print("[ALERT] Critical Finding or High Uncertainty! Engaging Red LED and Buzzer.")
        if ON_JETSON:
            GPIO.output(GREEN_LED_PIN, GPIO.LOW)
            GPIO.output(RED_LED_PIN, GPIO.HIGH)
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            time.sleep(1.5)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
    else:
        print("[STATUS] Normal Finding. Engaging Green LED.")
        if ON_JETSON:
            GPIO.output(RED_LED_PIN, GPIO.LOW)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            GPIO.output(GREEN_LED_PIN, GPIO.HIGH)

def reset_hardware():
    """Cleans up GPIO state upon shutdown."""
    if ON_JETSON:
        GPIO.cleanup()
        print("[SYSTEM] GPIO Pins Cleaned Up.")

if __name__ == "__main__":
    setup_hardware()
    try:
        # Simulate pipeline outputs to test the routing logic
        test_cases = [
            ("Normal Skin", 1.2),   # Normal: Should trigger Green
            ("Glioma", 2.4),        # Critical: Should trigger Red/Buzzer
            ("V", 1.8),             # ECG Arrhythmia: Should trigger Red/Buzzer
            ("Normal", 6.8)         # High Uncertainty: Should trigger Red/Buzzer
        ]
        
        for case, uncertainty in test_cases:
            trigger_alert(case, uncertainty)
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n[SYSTEM] Terminated.")
    finally:
        reset_hardware()