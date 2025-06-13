#!/usr/bin/env python3
"""
Advanced Encryption and Security System
Comprehensive cryptographic operations, security management, and access control
"""

import asyncio
import os
import secrets
import hashlib
import hmac
import base64
import time
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import jwt
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
import ssl
import socket


class EncryptionType(Enum):
    """Encryption algorithm types"""
    AES_256_GCM = auto()
    AES_256_CBC = auto()
    CHACHA20_POLY1305 = auto()
    FERNET = auto()
    RSA_2048 = auto()
    RSA_4096 = auto()


class HashAlgorithm(Enum):
    """Hash algorithm types"""
    SHA256 = 'sha256'
    SHA512 = 'sha512'
    SHA3_256 = 'sha3_256'
    SHA3_512 = 'sha3_512'
    BLAKE2B = 'blake2b'
    BLAKE2S = 'blake2s'


class SecurityLevel(Enum):
    """Security levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    MAXIMUM = 4


@dataclass
class CryptoKey:
    """Cryptographic key information"""
    key_id: str
    key_type: EncryptionType
    key_data: bytes
    created_at: float
    expires_at: Optional[float] = None
    usage_count: int = 0
    max_usage: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    name: str
    min_password_length: int = 12
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digits: bool = True
    require_special_chars: bool = True
    max_login_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes
    session_timeout: int = 3600  # 1 hour
    require_2fa: bool = False
    allowed_ip_ranges: List[str] = field(default_factory=list)
    blocked_ip_ranges: List[str] = field(default_factory=list)
    encryption_level: SecurityLevel = SecurityLevel.HIGH


@dataclass
class SecurityEvent:
    """Security event record"""
    event_id: str
    event_type: str
    timestamp: float
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    severity: str = "INFO"
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class KeyDerivation:
    """Key derivation functions"""
    
    @staticmethod
    def pbkdf2(password: bytes, salt: bytes, iterations: int = 100000, 
               key_length: int = 32, hash_algorithm=hashes.SHA256()) -> bytes:
        """Derive key using PBKDF2"""
        kdf = PBKDF2HMAC(
            algorithm=hash_algorithm,
            length=key_length,
            salt=salt,
            iterations=iterations,
            backend=default_backend()
        )
        return kdf.derive(password)
    
    @staticmethod
    def scrypt(password: bytes, salt: bytes, n: int = 2**14, r: int = 8, 
               p: int = 1, key_length: int = 32) -> bytes:
        """Derive key using Scrypt"""
        kdf = Scrypt(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=salt,
            n=n,
            r=r,
            p=p,
            backend=default_backend()
        )
        return kdf.derive(password)


class SymmetricEncryption:
    """Symmetric encryption operations"""
    
    def __init__(self):
        self.backend = default_backend()
    
    def encrypt_aes_gcm(self, plaintext: bytes, key: bytes, 
                       associated_data: bytes = None) -> Tuple[bytes, bytes, bytes]:
        """Encrypt using AES-256-GCM"""
        iv = os.urandom(12)  # 96-bit IV for GCM
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        
        if associated_data:
            encryptor.authenticate_additional_data(associated_data)
        
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        return ciphertext, iv, encryptor.tag
    
    def decrypt_aes_gcm(self, ciphertext: bytes, key: bytes, iv: bytes, 
                       tag: bytes, associated_data: bytes = None) -> bytes:
        """Decrypt using AES-256-GCM"""
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv, tag),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        
        if associated_data:
            decryptor.authenticate_additional_data(associated_data)
        
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    def encrypt_chacha20_poly1305(self, plaintext: bytes, key: bytes, 
                                 associated_data: bytes = None) -> Tuple[bytes, bytes]:
        """Encrypt using ChaCha20-Poly1305"""
        nonce = os.urandom(12)
        cipher = Cipher(
            algorithms.ChaCha20(key, nonce),
            modes.Poly1305(),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        
        if associated_data:
            encryptor.authenticate_additional_data(associated_data)
        
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        return ciphertext + encryptor.tag, nonce
    
    def decrypt_chacha20_poly1305(self, ciphertext_with_tag: bytes, key: bytes, 
                                 nonce: bytes, associated_data: bytes = None) -> bytes:
        """Decrypt using ChaCha20-Poly1305"""
        ciphertext = ciphertext_with_tag[:-16]
        tag = ciphertext_with_tag[-16:]
        
        cipher = Cipher(
            algorithms.ChaCha20(key, nonce),
            modes.Poly1305(tag),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        
        if associated_data:
            decryptor.authenticate_additional_data(associated_data)
        
        return decryptor.update(ciphertext) + decryptor.finalize()


class AsymmetricEncryption:
    """Asymmetric encryption operations"""
    
    def __init__(self):
        self.backend = default_backend()
    
    def generate_rsa_keypair(self, key_size: int = 2048) -> Tuple[bytes, bytes]:
        """Generate RSA key pair"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=self.backend
        )
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def encrypt_rsa(self, plaintext: bytes, public_key_pem: bytes) -> bytes:
        """Encrypt using RSA public key"""
        public_key = serialization.load_pem_public_key(
            public_key_pem, backend=self.backend
        )
        
        ciphertext = public_key.encrypt(
            plaintext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return ciphertext
    
    def decrypt_rsa(self, ciphertext: bytes, private_key_pem: bytes) -> bytes:
        """Decrypt using RSA private key"""
        private_key = serialization.load_pem_private_key(
            private_key_pem, password=None, backend=self.backend
        )
        
        plaintext = private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return plaintext
    
    def sign_rsa(self, message: bytes, private_key_pem: bytes) -> bytes:
        """Sign message using RSA private key"""
        private_key = serialization.load_pem_private_key(
            private_key_pem, password=None, backend=self.backend
        )
        
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature
    
    def verify_rsa(self, message: bytes, signature: bytes, public_key_pem: bytes) -> bool:
        """Verify RSA signature"""
        try:
            public_key = serialization.load_pem_public_key(
                public_key_pem, backend=self.backend
            )
            
            public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False


class HashManager:
    """Hash and HMAC operations"""
    
    @staticmethod
    def hash_data(data: bytes, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
        """Hash data using specified algorithm"""
        if algorithm == HashAlgorithm.SHA256:
            return hashlib.sha256(data).hexdigest()
        elif algorithm == HashAlgorithm.SHA512:
            return hashlib.sha512(data).hexdigest()
        elif algorithm == HashAlgorithm.SHA3_256:
            return hashlib.sha3_256(data).hexdigest()
        elif algorithm == HashAlgorithm.SHA3_512:
            return hashlib.sha3_512(data).hexdigest()
        elif algorithm == HashAlgorithm.BLAKE2B:
            return hashlib.blake2b(data).hexdigest()
        elif algorithm == HashAlgorithm.BLAKE2S:
            return hashlib.blake2s(data).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    @staticmethod
    def hmac_sign(data: bytes, key: bytes, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
        """Create HMAC signature"""
        if algorithm == HashAlgorithm.SHA256:
            return hmac.new(key, data, hashlib.sha256).hexdigest()
        elif algorithm == HashAlgorithm.SHA512:
            return hmac.new(key, data, hashlib.sha512).hexdigest()
        else:
            raise ValueError(f"Unsupported HMAC algorithm: {algorithm}")
    
    @staticmethod
    def verify_hmac(data: bytes, signature: str, key: bytes, 
                   algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> bool:
        """Verify HMAC signature"""
        try:
            expected = HashManager.hmac_sign(data, key, algorithm)
            return hmac.compare_digest(signature, expected)
        except Exception:
            return False
    
    @staticmethod
    def hash_password(password: str, rounds: int = 12) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt(rounds=rounds)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception:
            return False


class SecureStorage:
    """Secure storage for sensitive data"""
    
    def __init__(self, storage_path: Path, master_key: bytes):
        self.storage_path = storage_path
        self.master_key = master_key
        self.fernet = Fernet(base64.urlsafe_b64encode(master_key[:32].ljust(32)[:32]))
        self.storage = {}
        self.lock = threading.Lock()
        
        self.load_storage()
    
    def store(self, key: str, data: Union[str, bytes, Dict, List]) -> bool:
        """Store encrypted data"""
        try:
            with self.lock:
                # Serialize data
                if isinstance(data, (dict, list)):
                    serialized = json.dumps(data).encode('utf-8')
                elif isinstance(data, str):
                    serialized = data.encode('utf-8')
                else:
                    serialized = data
                
                # Encrypt data
                encrypted = self.fernet.encrypt(serialized)
                
                # Store with metadata
                self.storage[key] = {
                    'data': base64.b64encode(encrypted).decode('utf-8'),
                    'timestamp': time.time(),
                    'type': type(data).__name__
                }
                
                self.save_storage()
                return True
        except Exception:
            return False
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve and decrypt data"""
        try:
            with self.lock:
                if key not in self.storage:
                    return None
                
                entry = self.storage[key]
                encrypted = base64.b64decode(entry['data'])
                
                # Decrypt data
                decrypted = self.fernet.decrypt(encrypted)
                
                # Deserialize based on original type
                if entry['type'] in ['dict', 'list']:
                    return json.loads(decrypted.decode('utf-8'))
                elif entry['type'] == 'str':
                    return decrypted.decode('utf-8')
                else:
                    return decrypted
        except Exception:
            return None
    
    def delete(self, key: str) -> bool:
        """Delete stored data"""
        try:
            with self.lock:
                if key in self.storage:
                    del self.storage[key]
                    self.save_storage()
                    return True
                return False
        except Exception:
            return False
    
    def list_keys(self) -> List[str]:
        """List all storage keys"""
        with self.lock:
            return list(self.storage.keys())
    
    def save_storage(self):
        """Save storage to disk"""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(self.storage, f, indent=2)
        except Exception:
            pass
    
    def load_storage(self):
        """Load storage from disk"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    self.storage = json.load(f)
        except Exception:
            self.storage = {}


class TokenManager:
    """JWT token management"""
    
    def __init__(self, secret_key: bytes, issuer: str = "InnovationWorkstation"):
        self.secret_key = secret_key
        self.issuer = issuer
        self.algorithm = 'HS256'
    
    def generate_token(self, payload: Dict[str, Any], expires_in: int = 3600) -> str:
        """Generate JWT token"""
        now = time.time()
        token_payload = {
            'iss': self.issuer,
            'iat': now,
            'exp': now + expires_in,
            **payload
        }
        
        return jwt.encode(token_payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token, self.secret_key, 
                algorithms=[self.algorithm],
                options={'verify_exp': True}
            )
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def refresh_token(self, token: str, expires_in: int = 3600) -> Optional[str]:
        """Refresh JWT token"""
        payload = self.verify_token(token)
        if payload:
            # Remove standard claims
            for claim in ['iss', 'iat', 'exp']:
                payload.pop(claim, None)
            
            return self.generate_token(payload, expires_in)
        return None


class SecurityAudit:
    """Security auditing and monitoring"""
    
    def __init__(self, max_events: int = 10000):
        self.events = deque(maxlen=max_events)
        self.failed_attempts = defaultdict(list)  # ip -> List[timestamp]
        self.blocked_ips = set()
        self.lock = threading.Lock()
    
    def log_event(self, event_type: str, severity: str = "INFO", 
                  source_ip: str = None, user_id: str = None, 
                  description: str = "", **details):
        """Log security event"""
        with self.lock:
            event = SecurityEvent(
                event_id=secrets.token_hex(16),
                event_type=event_type,
                timestamp=time.time(),
                source_ip=source_ip,
                user_id=user_id,
                severity=severity,
                description=description,
                details=details
            )
            
            self.events.append(event)
            
            # Track failed login attempts
            if event_type == "login_failed" and source_ip:
                self.failed_attempts[source_ip].append(time.time())
                
                # Clean old attempts (older than 1 hour)
                cutoff = time.time() - 3600
                self.failed_attempts[source_ip] = [
                    t for t in self.failed_attempts[source_ip] if t > cutoff
                ]
                
                # Check if IP should be blocked
                if len(self.failed_attempts[source_ip]) >= 5:
                    self.blocked_ips.add(source_ip)
                    self.log_event("ip_blocked", "WARNING", source_ip=source_ip,
                                 description=f"IP blocked due to {len(self.failed_attempts[source_ip])} failed attempts")
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked"""
        with self.lock:
            return ip in self.blocked_ips
    
    def unblock_ip(self, ip: str):
        """Unblock IP address"""
        with self.lock:
            self.blocked_ips.discard(ip)
            if ip in self.failed_attempts:
                del self.failed_attempts[ip]
    
    def get_recent_events(self, count: int = 100, 
                         event_type: str = None, severity: str = None) -> List[SecurityEvent]:
        """Get recent security events"""
        with self.lock:
            events = list(self.events)
            
            # Filter by event type
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            # Filter by severity
            if severity:
                events = [e for e in events if e.severity == severity]
            
            # Return most recent
            return events[-count:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get security statistics"""
        with self.lock:
            event_types = defaultdict(int)
            severities = defaultdict(int)
            
            for event in self.events:
                event_types[event.event_type] += 1
                severities[event.severity] += 1
            
            return {
                'total_events': len(self.events),
                'blocked_ips': len(self.blocked_ips),
                'failed_attempts_tracking': len(self.failed_attempts),
                'event_types': dict(event_types),
                'severities': dict(severities)
            }


class EncryptionManager:
    """Main encryption manager"""
    
    def __init__(self, app):
        self.app = app
        self.symmetric_crypto = SymmetricEncryption()
        self.asymmetric_crypto = AsymmetricEncryption()
        self.hash_manager = HashManager()
        
        # Key management
        self.keys = {}  # key_id -> CryptoKey
        self.master_key = None
        self.secure_storage = None
        self.token_manager = None
        
        # Configuration
        self.default_encryption_type = EncryptionType.AES_256_GCM
        self.key_rotation_interval = 86400 * 30  # 30 days
        self.max_key_usage = 1000000
        
        self.lock = threading.Lock()
    
    async def setup(self) -> bool:
        """Setup encryption manager"""
        try:
            # Generate or load master key
            await self._initialize_master_key()
            
            # Initialize secure storage
            storage_path = self.app.file_manager.data_dir / "secure_storage.json"
            self.secure_storage = SecureStorage(storage_path, self.master_key)
            
            # Initialize token manager
            self.token_manager = TokenManager(self.master_key)
            
            # Generate default encryption keys
            await self._generate_default_keys()
            
            self.app.logger.info("Encryption manager initialized successfully")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "EncryptionManager.setup")
            return False
    
    async def _initialize_master_key(self):
        """Initialize master encryption key"""
        key_file = self.app.file_manager.data_dir / ".master_key"
        
        if key_file.exists():
            # Load existing key
            key_data = await self.app.file_manager.read_binary_async(key_file)
            if key_data and len(key_data) == 32:
                self.master_key = key_data
                return
        
        # Generate new master key
        self.master_key = secrets.token_bytes(32)
        
        # Save securely
        await self.app.file_manager.write_binary_async(key_file, self.master_key)
        
        # Set restrictive permissions
        try:
            os.chmod(key_file, 0o600)
        except:
            pass  # Windows doesn't support chmod
        
        self.app.logger.info("New master key generated and saved")
    
    async def _generate_default_keys(self):
        """Generate default encryption keys"""
        # AES-256 key for general encryption
        aes_key = secrets.token_bytes(32)
        await self.add_key("default_aes", EncryptionType.AES_256_GCM, aes_key)
        
        # RSA key pair for asymmetric operations
        private_key, public_key = self.asymmetric_crypto.generate_rsa_keypair(2048)
        await self.add_key("default_rsa_private", EncryptionType.RSA_2048, private_key)
        await self.add_key("default_rsa_public", EncryptionType.RSA_2048, public_key)
    
    async def add_key(self, key_id: str, key_type: EncryptionType, 
                     key_data: bytes, expires_in: int = None) -> bool:
        """Add encryption key"""
        try:
            with self.lock:
                expires_at = time.time() + expires_in if expires_in else None
                
                crypto_key = CryptoKey(
                    key_id=key_id,
                    key_type=key_type,
                    key_data=key_data,
                    created_at=time.time(),
                    expires_at=expires_at,
                    max_usage=self.max_key_usage
                )
                
                self.keys[key_id] = crypto_key
                
                # Store in secure storage
                await self._store_key_securely(crypto_key)
                
                return True
        except Exception as e:
            self.app.error_handler.handle_error(e, f"EncryptionManager.add_key({key_id})")
            return False
    
    async def _store_key_securely(self, crypto_key: CryptoKey):
        """Store key in secure storage"""
        key_info = {
            'key_type': crypto_key.key_type.name,
            'key_data': base64.b64encode(crypto_key.key_data).decode('utf-8'),
            'created_at': crypto_key.created_at,
            'expires_at': crypto_key.expires_at,
            'usage_count': crypto_key.usage_count,
            'max_usage': crypto_key.max_usage,
            'metadata': crypto_key.metadata
        }
        
        self.secure_storage.store(f"key_{crypto_key.key_id}", key_info)
    
    async def encrypt_data(self, data: Union[str, bytes], key_id: str = "default_aes",
                          associated_data: bytes = None) -> Optional[Dict[str, str]]:
        """Encrypt data using specified key"""
        try:
            if key_id not in self.keys:
                return None
            
            crypto_key = self.keys[key_id]
            
            # Check key expiration and usage
            if not await self._validate_key_usage(crypto_key):
                return None
            
            # Convert string to bytes
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Encrypt based on key type
            if crypto_key.key_type == EncryptionType.AES_256_GCM:
                ciphertext, iv, tag = self.symmetric_crypto.encrypt_aes_gcm(
                    data, crypto_key.key_data, associated_data
                )
                
                return {
                    'algorithm': 'AES_256_GCM',
                    'ciphertext': base64.b64encode(ciphertext).decode('utf-8'),
                    'iv': base64.b64encode(iv).decode('utf-8'),
                    'tag': base64.b64encode(tag).decode('utf-8'),
                    'key_id': key_id
                }
            
            elif crypto_key.key_type == EncryptionType.CHACHA20_POLY1305:
                ciphertext_with_tag, nonce = self.symmetric_crypto.encrypt_chacha20_poly1305(
                    data, crypto_key.key_data, associated_data
                )
                
                return {
                    'algorithm': 'CHACHA20_POLY1305',
                    'ciphertext': base64.b64encode(ciphertext_with_tag).decode('utf-8'),
                    'nonce': base64.b64encode(nonce).decode('utf-8'),
                    'key_id': key_id
                }
            
            # Update key usage
            crypto_key.usage_count += 1
            await self._store_key_securely(crypto_key)
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"EncryptionManager.encrypt_data({key_id})")
            return None
    
    async def decrypt_data(self, encrypted_data: Dict[str, str], 
                          associated_data: bytes = None) -> Optional[bytes]:
        """Decrypt data"""
        try:
            key_id = encrypted_data.get('key_id')
            if not key_id or key_id not in self.keys:
                return None
            
            crypto_key = self.keys[key_id]
            algorithm = encrypted_data.get('algorithm')
            
            if algorithm == 'AES_256_GCM':
                ciphertext = base64.b64decode(encrypted_data['ciphertext'])
                iv = base64.b64decode(encrypted_data['iv'])
                tag = base64.b64decode(encrypted_data['tag'])
                
                return self.symmetric_crypto.decrypt_aes_gcm(
                    ciphertext, crypto_key.key_data, iv, tag, associated_data
                )
            
            elif algorithm == 'CHACHA20_POLY1305':
                ciphertext_with_tag = base64.b64decode(encrypted_data['ciphertext'])
                nonce = base64.b64decode(encrypted_data['nonce'])
                
                return self.symmetric_crypto.decrypt_chacha20_poly1305(
                    ciphertext_with_tag, crypto_key.key_data, nonce, associated_data
                )
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "EncryptionManager.decrypt_data")
            return None
    
    async def _validate_key_usage(self, crypto_key: CryptoKey) -> bool:
        """Validate key usage limits"""
        current_time = time.time()
        
        # Check expiration
        if crypto_key.expires_at and current_time > crypto_key.expires_at:
            return False
        
        # Check usage limit
        if crypto_key.max_usage and crypto_key.usage_count >= crypto_key.max_usage:
            return False
        
        return True
    
    def hash_data(self, data: Union[str, bytes], 
                 algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
        """Hash data"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return self.hash_manager.hash_data(data, algorithm)
    
    def hash_password(self, password: str) -> str:
        """Hash password securely"""
        return self.hash_manager.hash_password(password)
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password"""
        return self.hash_manager.verify_password(password, hashed)
    
    def generate_token(self, payload: Dict[str, Any], expires_in: int = 3600) -> str:
        """Generate authentication token"""
        return self.token_manager.generate_token(payload, expires_in)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify authentication token"""
        return self.token_manager.verify_token(token)
    
    def get_key_info(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get key information"""
        if key_id in self.keys:
            key = self.keys[key_id]
            return {
                'key_id': key.key_id,
                'key_type': key.key_type.name,
                'created_at': key.created_at,
                'expires_at': key.expires_at,
                'usage_count': key.usage_count,
                'max_usage': key.max_usage,
                'is_expired': key.expires_at and time.time() > key.expires_at if key.expires_at else False
            }
        return None
    
    def list_keys(self) -> List[str]:
        """List all key IDs"""
        return list(self.keys.keys())


class SecurityManager:
    """Main security manager"""
    
    def __init__(self, app):
        self.app = app
        self.encryption_manager = None
        self.security_audit = SecurityAudit()
        self.security_policies = {}
        self.active_sessions = {}  # session_id -> session_info
        self.rate_limiters = defaultdict(deque)  # endpoint -> request_times
        
        # Default security policy
        self.default_policy = SecurityPolicy("default")
        self.security_policies["default"] = self.default_policy
    
    async def setup(self) -> bool:
        """Setup security manager"""
        try:
            # Initialize encryption manager
            self.encryption_manager = EncryptionManager(self.app)
            if not await self.encryption_manager.setup():
                return False
            
            # Load security policies
            await self._load_security_policies()
            
            self.app.logger.info("Security manager initialized successfully")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "SecurityManager.setup")
            return False
    
    async def _load_security_policies(self):
        """Load security policies from configuration"""
        try:
            policies_file = self.app.file_manager.config_dir / "security_policies.json"
            
            if policies_file.exists():
                content = await self.app.file_manager.read_text_async(policies_file)
                if content:
                    policies_data = json.loads(content)
                    
                    for name, policy_data in policies_data.items():
                        policy = SecurityPolicy(
                            name=name,
                            min_password_length=policy_data.get('min_password_length', 12),
                            require_uppercase=policy_data.get('require_uppercase', True),
                            require_lowercase=policy_data.get('require_lowercase', True),
                            require_digits=policy_data.get('require_digits', True),
                            require_special_chars=policy_data.get('require_special_chars', True),
                            max_login_attempts=policy_data.get('max_login_attempts', 5),
                            lockout_duration=policy_data.get('lockout_duration', 900),
                            session_timeout=policy_data.get('session_timeout', 3600),
                            require_2fa=policy_data.get('require_2fa', False),
                            allowed_ip_ranges=policy_data.get('allowed_ip_ranges', []),
                            blocked_ip_ranges=policy_data.get('blocked_ip_ranges', []),
                            encryption_level=SecurityLevel[policy_data.get('encryption_level', 'HIGH')]
                        )
                        
                        self.security_policies[name] = policy
        except Exception as e:
            self.app.logger.warning(f"Could not load security policies: {e}")
    
    def validate_password(self, password: str, policy_name: str = "default") -> Tuple[bool, List[str]]:
        """Validate password against security policy"""
        policy = self.security_policies.get(policy_name, self.default_policy)
        errors = []
        
        if len(password) < policy.min_password_length:
            errors.append(f"Password must be at least {policy.min_password_length} characters long")
        
        if policy.require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if policy.require_lowercase and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if policy.require_digits and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        if policy.require_special_chars and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        return len(errors) == 0, errors
    
    def create_session(self, user_id: str, ip_address: str = None, 
                      policy_name: str = "default") -> str:
        """Create authenticated session"""
        policy = self.security_policies.get(policy_name, self.default_policy)
        session_id = secrets.token_urlsafe(32)
        
        session_info = {
            'session_id': session_id,
            'user_id': user_id,
            'created_at': time.time(),
            'last_activity': time.time(),
            'ip_address': ip_address,
            'expires_at': time.time() + policy.session_timeout,
            'policy': policy_name
        }
        
        self.active_sessions[session_id] = session_info
        
        # Log session creation
        self.security_audit.log_event(
            "session_created", "INFO",
            source_ip=ip_address, user_id=user_id,
            description=f"Session created for user {user_id}",
            session_id=session_id
        )
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate session"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        current_time = time.time()
        
        # Check expiration
        if current_time > session['expires_at']:
            del self.active_sessions[session_id]
            
            self.security_audit.log_event(
                "session_expired", "INFO",
                user_id=session['user_id'],
                description=f"Session expired for user {session['user_id']}",
                session_id=session_id
            )
            
            return None
        
        # Update last activity
        session['last_activity'] = current_time
        
        return session
    
    def invalidate_session(self, session_id: str):
        """Invalidate session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            del self.active_sessions[session_id]
            
            self.security_audit.log_event(
                "session_invalidated", "INFO",
                user_id=session['user_id'],
                description=f"Session invalidated for user {session['user_id']}",
                session_id=session_id
            )
    
    def check_rate_limit(self, identifier: str, max_requests: int = 100, 
                        window_seconds: int = 3600) -> bool:
        """Check rate limiting"""
        current_time = time.time()
        requests = self.rate_limiters[identifier]
        
        # Remove old requests outside the window
        while requests and current_time - requests[0] > window_seconds:
            requests.popleft()
        
        # Check if limit exceeded
        if len(requests) >= max_requests:
            return False
        
        # Add current request
        requests.append(current_time)
        return True
    
    def is_ip_allowed(self, ip_address: str, policy_name: str = "default") -> bool:
        """Check if IP address is allowed"""
        if self.security_audit.is_ip_blocked(ip_address):
            return False
        
        policy = self.security_policies.get(policy_name, self.default_policy)
        
        # Check blocked ranges
        for blocked_range in policy.blocked_ip_ranges:
            if self._ip_in_range(ip_address, blocked_range):
                return False
        
        # Check allowed ranges (if specified)
        if policy.allowed_ip_ranges:
            for allowed_range in policy.allowed_ip_ranges:
                if self._ip_in_range(ip_address, allowed_range):
                    return True
            return False  # Not in any allowed range
        
        return True  # No restrictions
    
    def _ip_in_range(self, ip: str, ip_range: str) -> bool:
        """Check if IP is in range (simplified implementation)"""
        # This is a simplified implementation
        # In production, use proper IP address libraries
        if '/' in ip_range:
            # CIDR notation
            network, prefix = ip_range.split('/')
            # Simplified check - in production use ipaddress module
            return ip.startswith(network.rsplit('.', 1)[0])
        else:
            # Single IP or wildcard
            return ip == ip_range or ip_range == '*'
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics"""
        return {
            'encryption_manager': {
                'keys_count': len(self.encryption_manager.keys) if self.encryption_manager else 0,
                'secure_storage_keys': len(self.encryption_manager.secure_storage.list_keys()) if self.encryption_manager and self.encryption_manager.secure_storage else 0
            },
            'sessions': {
                'active_sessions': len(self.active_sessions),
                'rate_limiters': len(self.rate_limiters)
            },
            'audit': self.security_audit.get_statistics(),
            'policies': len(self.security_policies)
        }