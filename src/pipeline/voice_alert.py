import subprocess
import threading
import time

class VoiceAnnouncer:
    def __init__(self):
        print("[VOICE SYSTEM] Initializing raw espeak OS engine...")
        self.is_ready = True
        print("[VOICE SYSTEM] Ready.")

    def _speak_thread(self, text, rate):
        """Runs espeak as an isolated OS process. Immune to Python segfaults."""
        try:
            # -s controls the words-per-minute speed
            cmd = ['espeak', '-s', str(rate), text]
            # stderr=subprocess.DEVNULL hides the annoying ALSA hardware warnings
            subprocess.run(cmd, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[VOICE ERROR] Hardware audio bus busy: {e}")

    def announce(self, text, priority="NORMAL"):
        if not self.is_ready:
            return

        if priority == "CRITICAL":
            rate = 175 # Speak faster for emergencies
            prefix = "Critical Alert. "
        else:
            rate = 140 # Slower, calmer for standard info
            prefix = "System Update. "

        full_message = prefix + text
        print(f"\n[SPEAKER] 🔊 '{full_message}'")
        
        # Fire and forget (OS handles the process safely)
        t = threading.Thread(target=self._speak_thread, args=(full_message, rate))
        t.daemon = True
        t.start()

# ==========================================
# QUICK VERIFICATION TEST
# ==========================================
if __name__ == "__main__":
    announcer = VoiceAnnouncer()
    
    print("\n--- Testing Standard Alert ---")
    announcer.announce("Scan complete. Patient routed to Tier 2.", priority="NORMAL")
    time.sleep(3) 
    
    print("\n--- Testing Critical Alert ---")
    announcer.announce("High grade Glioma detected. Immediate oncology review required.", priority="CRITICAL")
    time.sleep(4)
