import os
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature
from base64 import urlsafe_b64encode, urlsafe_b64decode
import keyring

class EncryptionSystem:
    def __init__(self, app):
        self.app = app

    def setup(self, key=None):
        self.key = key if key else self.generate_key()
        self.fernet = Fernet(self.key)

    @staticmethod
    def generate_key():
        # Implement a more secure key management system
        return Fernet.generate_key()

    @staticmethod
    def derive_key(password: str, salt: bytes):
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return urlsafe_b64encode(kdf.derive(password.encode()))

    def encrypt(self, data: bytes) -> bytes:
        return self.fernet.encrypt(data)

    def decrypt(self, token: bytes) -> bytes:
        return self.fernet.decrypt(token)

    def encrypt_file(self, file_path: str, output_path: str):
        with open(file_path, 'rb') as file:
            file_data = file.read()
        encrypted_data = self.encrypt(file_data)
        with open(output_path, 'wb') as file:
            file.write(encrypted_data)

    def decrypt_file(self, file_path: str, output_path: str):
        with open(file_path, 'rb') as file:
            encrypted_data = file.read()
        decrypted_data = self.decrypt(encrypted_data)
        with open(output_path, 'wb') as file:
            file.write(decrypted_data)

    def generate_rsa_keys(self):
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return private_pem, public_pem

    def encrypt_with_public_key(self, public_key, data: bytes) -> bytes:
        """Encrypt data using the public key."""
        return public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

    def decrypt_with_private_key(self, private_key, encrypted_data: bytes) -> bytes:
        """Decrypt data using the private key."""
        return private_key.decrypt(
            encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
    def store_key_in_keyring(self, key_name, key_value):
        """Store a key in the system's keyring."""
        keyring.set_password("your_application_name", key_name, key_value)

    def retrieve_key_from_keyring(self, key_name):
        """Retrieve a key from the system's keyring."""
        return keyring.get_password("your_application_name", key_name)

    def encrypt_string(self, data: str) -> str:
        """Encrypt a string."""
        return self.fernet.encrypt(data.encode()).decode()

    def decrypt_string(self, token: str) -> str:
        """Decrypt a string."""
        return self.fernet.decrypt(token.encode()).decode()

    def initialize_encryption_keys(self):
        """Initializes and manages the primary and secondary encryption keys."""
        primary_key_name = "primary_encryption_key"
        first_encryption_key = self.retrieve_key_from_keyring(primary_key_name)

        if not first_encryption_key:
            first_encryption_key = self.generate_key()
            # Convert bytes to a string for storage
            first_encryption_key_str = first_encryption_key.decode()
            self.store_key_in_keyring(primary_key_name, first_encryption_key_str)

        # If you want to print the key for debugging purposes, ensure it's done securely
        #print("First encryption key:", first_encryption_key)

        # Initialize the encryption system with the first key
        self.setup(key=first_encryption_key)

    def sec_encrypt_key(self, env_file_path):
        # Work with the second encryption key
        second_encryption_key = None
        with open(env_file_path, 'r+') as file:
            first_line = file.readline().strip()
            if first_line:
                second_encryption_key = self.decrypt_string(first_line)

            else:
                # Generate, encrypt, and store the second key
                second_encryption_key = Fernet.generate_key().decode()
                encrypted_second_key = self.encrypt_string(second_encryption_key)
                remaining_content = file.read()
                file.seek(0)
                file.write(encrypted_second_key + '\n' + remaining_content)

        return second_encryption_key

# Inside the EncryptionSystem class:

    def encrypt_file(self, file_path: str, output_path: str):
        try:
            with open(file_path, 'rb') as file:
                file_data = file.read()
            encrypted_data = self.encrypt(file_data)
            with open(output_path, 'wb') as file:
                file.write(encrypted_data)
            self.app.logger.info(f"File encrypted at {output_path}")
        except Exception as e:
            self.app.logger.error(f"Error encrypting file {file_path}: {e}")
            raise Exception(f"Error encrypting file: {e}")

    def decrypt_file(self, file_path: str, output_path: str):
        try:
            with open(file_path, 'rb') as file:
                encrypted_data = file.read()
            decrypted_data = self.decrypt(encrypted_data)
            with open(output_path, 'wb') as file:
                file.write(decrypted_data)
            self.app.logger.info(f"File decrypted at {output_path}")
        except Exception as e:
            self.app.logger.error(f"Error decrypting file {file_path}: {e}")
            raise Exception(f"Error decrypting file: {e}")



    # Add more methods for asymmetric encryption, integrity checks, etc.
