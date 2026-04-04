from cryptography.fernet import Fernet
import os
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class SecurityManager:
    """Handles AES-256 encryption/decryption for sensitive data."""
    
    def __init__(self, master_key: str = None):
        if not master_key:
            master_key = os.getenv("MASTER_KEY", "default-unsafe-key-change-me")
        
        # Derive a robust 32-byte key from the master_key
        salt = b'moneybott-salt' # In production, use a unique salt per installation
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        self.fernet = Fernet(key)

    def encrypt(self, data: str) -> str:
        """Encrypts a string and returns a base64 encoded string."""
        if not data:
            return ""
        return self.fernet.encrypt(data.encode()).decode()

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypts a base64 encoded string and returns the original string."""
        if not encrypted_data:
            return ""
        try:
            return self.fernet.decrypt(encrypted_data.encode()).decode()
        except Exception:
            return "DECRYPTION_FAILED"

# Global instance
security = SecurityManager()
