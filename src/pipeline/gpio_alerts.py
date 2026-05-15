import Jetson.GPIO as GPIO

def setup_pins():
    # CRITICAL: Use physical board pin numbers to match your actual wiring
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    
    # Setup pins 11 (Green), 13 (Red), 15 (Buzzer)
    GPIO.setup(11, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(13, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(15, GPIO.OUT, initial=GPIO.LOW)

def trigger_success_led():
    GPIO.output(11, GPIO.HIGH) # Green ON
    GPIO.output(13, GPIO.LOW)  # Red OFF
    GPIO.output(15, GPIO.LOW)  # Buzzer OFF

def trigger_warning_led():
    # Placeholder if you decide to add a yellow LED later
    pass 

def trigger_emergency_alarm():
    GPIO.output(13, GPIO.HIGH) # Red ON
    GPIO.output(15, GPIO.HIGH) # Buzzer ON
    GPIO.output(11, GPIO.LOW)  # Green OFF
