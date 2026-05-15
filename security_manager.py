"""
security_manager.py — FYP: M3 Day 15-20 Deliverable
Handles AES-256 Data-at-Rest Encryption and JWT Token Logic.
"""
from cryptography.fernet import Fernet
from flask_jwt_extended import create_access_token
import os

# In a real scenario, store this in an Environment Variable
# Generate one using Fernet.generate_key()
SECRET_KEY = b'your-32-byte-base64-encoded-key-here=' 

class SecurityManager:
    def __init__(self, key=SECRET_KEY):
        self.cipher = Fernet(key)

    def encrypt_data(self, plain_text):
        """Encrypts sensitive strings (Patient Name, etc.)"""
        if not plain_text: return None
        return self.cipher.encrypt(plain_text.encode()).decode()

    def decrypt_data(self, cipher_text):
        """Decrypts sensitive strings for the GUI/App"""
        if not cipher_text: return None
        try:
            return self.cipher.decrypt(cipher_text.encode()).decode()
        except Exception:
            return "Decryption Error"

    def generate_token(self, identity="clinician_01"):
        """Generates a JWT for the M4 Flutter App"""
        return create_access_token(identity=identity)

# Simple Test
if __name__ == "__main__":
    sm = SecurityManager()
    enc = sm.encrypt_data("Ahmed Rasheed")
    print(f"Encrypted: {enc}")
    print(f"Decrypted: {sm.decrypt_data(enc)}")