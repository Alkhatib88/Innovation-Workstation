#!/usr/bin/env python3
"""
Advanced Storage and Cache Management System
Comprehensive storage backends, caching strategies, and data persistence
"""

import asyncio
import time
import threading
import uuid
import json
import pickle
import hashlib
import gzip
import lz4.frame
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Set, Tuple, Generic, TypeVar
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque, OrderedDict
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import weakref
import mmap
import sqlite3
import os


T = TypeVar('T')


class StorageType(Enum):
    """Storage backend types"""
    MEMORY = auto()
    DISK = auto()
    REDIS = auto()
    SQLITE = auto()
    HYBRID = auto()
    DISTRIBUTED = auto()


class CachePolicy(Enum):
    """Cache eviction policies"""
    LRU = auto()          # Least Recently Used
    LFU = auto()          # Least Frequently Used
    FIFO = auto()         # First In, First Out
    LIFO = auto()         # Last In, First Out
    TTL = auto()          # Time To Live
    RANDOM = auto()       # Random eviction


class CompressionType(Enum):
    """Compression algorithms"""
    NONE = auto()
    GZIP = auto()
    LZ4 = auto()
    ZSTD = auto()
    BZIP2 = auto()


class SerializationType(Enum):
    """Serialization formats"""
    JSON = auto()
    PICKLE = auto()
    MSGPACK = auto()
    PROTOBUF = auto()
    AVRO = auto()


@dataclass
class StorageConfig:
    """Storage configuration"""
    name: str
    storage_type: StorageType = StorageType.MEMORY
    max_size: int = 1000000  # Maximum number of items
    max_memory: int = 1024 * 1024 * 1024  # Maximum memory in bytes (1GB)
    cache_policy: CachePolicy = CachePolicy.LRU
    compression: CompressionType = CompressionType.NONE
    serialization: SerializationType = SerializationType.PICKLE
    persistence_path: Optional[Path] = None
    auto_persist: bool = False
    persist_interval: float = 300.0  # 5 minutes
    encryption_enabled: bool = False
    ttl_default: Optional[float] = None
    cleanup_interval: float = 60.0  # 1 minute


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None
    size: int = 0
    compressed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StorageStats:
    """Storage statistics"""
    total_items: int = 0
    memory_usage: int = 0
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    persistence_count: int = 0
    compression_ratio: float = 0.0
    average_access_time: float = 0.0
    cache_hit_ratio: float = 0.0


class StorageBackend(ABC, Generic[T]):
    """Abstract storage backend"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[T]:
        """Get value by key"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: T, ttl: Optional[float] = None) -> bool:
        """Set value with optional TTL"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all data"""
        pass
    
    @abstractmethod
    async def keys(self) -> List[str]:
        """Get all keys"""
        pass
    
    @abstractmethod
    async def size(self) -> int:
        """Get number of items"""
        pass


class MemoryStorage(StorageBackend[T]):
    """High-performance in-memory storage"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.data = {}  # key -> CacheEntry
        self.access_order = OrderedDict()  # For LRU
        self.frequency = defaultdict(int)  # For LFU
        self.stats = StorageStats()
        self.lock = asyncio.Lock()
        
        # Compression and serialization
        self.compressor = self._get_compressor()
        self.serializer = self._get_serializer()
        
        # Background tasks
        self.cleanup_task = None
        self.persist_task = None
        self.running = False
    
    def _get_compressor(self):
        """Get compression function"""
        if self.config.compression == CompressionType.GZIP:
            return lambda data: gzip.compress(data)
        elif self.config.compression == CompressionType.LZ4:
            return lambda data: lz4.frame.compress(data)
        else:
            return lambda data: data
    
    def _get_decompressor(self):
        """Get decompression function"""
        if self.config.compression == CompressionType.GZIP:
            return lambda data: gzip.decompress(data)
        elif self.config.compression == CompressionType.LZ4:
            return lambda data: lz4.frame.decompress(data)
        else:
            return lambda data: data
    
    def _get_serializer(self):
        """Get serialization function"""
        if self.config.serialization == SerializationType.JSON:
            return lambda obj: json.dumps(obj).encode()
        elif self.config.serialization == SerializationType.PICKLE:
            return lambda obj: pickle.dumps(obj)
        else:
            return lambda obj: pickle.dumps(obj)
    
    def _get_deserializer(self):
        """Get deserialization function"""
        if self.config.serialization == SerializationType.JSON:
            return lambda data: json.loads(data.decode())
        elif self.config.serialization == SerializationType.PICKLE:
            return lambda data: pickle.loads(data)
        else:
            return lambda data: pickle.loads(data)
    
    async def start(self):
        """Start background tasks"""
        self.running = True
        
        if self.config.cleanup_interval > 0:
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        if self.config.auto_persist and self.config.persistence_path:
            self.persist_task = asyncio.create_task(self._persist_loop())
    
    async def stop(self):
        """Stop background tasks"""
        self.running = False
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.persist_task:
            self.persist_task.cancel()
    
    async def get(self, key: str) -> Optional[T]:
        """Get value by key"""
        start_time = time.time()
        
        async with self.lock:
            if key not in self.data:
                self.stats.miss_count += 1
                return None
            
            entry = self.data[key]
            
            # Check TTL
            if entry.ttl and (time.time() - entry.created_at) > entry.ttl:
                await self._remove_entry(key)
                self.stats.miss_count += 1
                return None
            
            # Update access metadata
            entry.accessed_at = time.time()
            entry.access_count += 1
            
            # Update access order for LRU
            if self.config.cache_policy == CachePolicy.LRU:
                self.access_order.move_to_end(key)
            
            # Update frequency for LFU
            if self.config.cache_policy == CachePolicy.LFU:
                self.frequency[key] += 1
            
            self.stats.hit_count += 1
            
            # Deserialize value
            try:
                if entry.compressed:
                    decompressor = self._get_decompressor()
                    deserializer = self._get_deserializer()
                    return deserializer(decompressor(entry.value))
                else:
                    deserializer = self._get_deserializer()
                    return deserializer(entry.value)
            except Exception:
                # If deserialization fails, remove entry
                await self._remove_entry(key)
                self.stats.miss_count += 1
                return None
        
        # Update statistics
        access_time = time.time() - start_time
        total_accesses = self.stats.hit_count + self.stats.miss_count
        if total_accesses > 0:
            self.stats.average_access_time = (
                (self.stats.average_access_time * (total_accesses - 1) + access_time) / total_accesses
            )
            self.stats.cache_hit_ratio = self.stats.hit_count / total_accesses
    
    async def set(self, key: str, value: T, ttl: Optional[float] = None) -> bool:
        """Set value with optional TTL"""
        try:
            async with self.lock:
                # Serialize and optionally compress
                serializer = self.serializer
                serialized = serializer(value)
                
                compressed = False
                if self.config.compression != CompressionType.NONE:
                    compressed_data = self.compressor(serialized)
                    if len(compressed_data) < len(serialized):
                        serialized = compressed_data
                        compressed = True
                
                # Calculate size
                size = len(serialized)
                
                # Check if we need to evict items
                while (len(self.data) >= self.config.max_size or 
                       self.stats.memory_usage + size > self.config.max_memory):
                    if not await self._evict_item():
                        return False  # Cannot evict any more items
                
                # Create entry
                entry = CacheEntry(
                    key=key,
                    value=serialized,
                    ttl=ttl or self.config.ttl_default,
                    size=size,
                    compressed=compressed
                )
                
                # Remove old entry if exists
                if key in self.data:
                    await self._remove_entry(key)
                
                # Add new entry
                self.data[key] = entry
                self.stats.memory_usage += size
                self.stats.total_items += 1
                
                # Update access tracking
                if self.config.cache_policy == CachePolicy.LRU:
                    self.access_order[key] = True
                elif self.config.cache_policy == CachePolicy.LFU:
                    self.frequency[key] = 1
                
                return True
        
        except Exception as e:
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key"""
        async with self.lock:
            if key in self.data:
                await self._remove_entry(key)
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        async with self.lock:
            if key not in self.data:
                return False
            
            entry = self.data[key]
            
            # Check TTL
            if entry.ttl and (time.time() - entry.created_at) > entry.ttl:
                await self._remove_entry(key)
                return False
            
            return True
    
    async def clear(self) -> bool:
        """Clear all data"""
        async with self.lock:
            self.data.clear()
            self.access_order.clear()
            self.frequency.clear()
            self.stats = StorageStats()
            return True
    
    async def keys(self) -> List[str]:
        """Get all keys"""
        async with self.lock:
            # Clean expired keys first
            await self._cleanup_expired()
            return list(self.data.keys())
    
    async def size(self) -> int:
        """Get number of items"""
        async with self.lock:
            return len(self.data)
    
    async def _remove_entry(self, key: str):
        """Remove entry and update statistics"""
        if key in self.data:
            entry = self.data[key]
            self.stats.memory_usage -= entry.size
            self.stats.total_items -= 1
            del self.data[key]
            
            # Clean up tracking structures
            self.access_order.pop(key, None)
            self.frequency.pop(key, None)
    
    async def _evict_item(self) -> bool:
        """Evict item based on cache policy"""
        if not self.data:
            return False
        
        key_to_evict = None
        
        if self.config.cache_policy == CachePolicy.LRU:
            # Evict least recently used
            key_to_evict = next(iter(self.access_order))
        
        elif self.config.cache_policy == CachePolicy.LFU:
            # Evict least frequently used
            key_to_evict = min(self.frequency.keys(), key=lambda k: self.frequency[k])
        
        elif self.config.cache_policy == CachePolicy.FIFO:
            # Evict oldest
            key_to_evict = min(self.data.keys(), key=lambda k: self.data[k].created_at)
        
        elif self.config.cache_policy == CachePolicy.LIFO:
            # Evict newest
            key_to_evict = max(self.data.keys(), key=lambda k: self.data[k].created_at)
        
        elif self.config.cache_policy == CachePolicy.TTL:
            # Evict expired items first, then oldest
            current_time = time.time()
            expired_keys = [
                k for k, v in self.data.items()
                if v.ttl and (current_time - v.created_at) > v.ttl
            ]
            if expired_keys:
                key_to_evict = expired_keys[0]
            else:
                key_to_evict = min(self.data.keys(), key=lambda k: self.data[k].created_at)
        
        elif self.config.cache_policy == CachePolicy.RANDOM:
            # Random eviction
            import random
            key_to_evict = random.choice(list(self.data.keys()))
        
        if key_to_evict:
            await self._remove_entry(key_to_evict)
            self.stats.eviction_count += 1
            return True
        
        return False
    
    async def _cleanup_expired(self):
        """Clean up expired entries"""
        if not self.config.ttl_default and not any(v.ttl for v in self.data.values()):
            return
        
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.data.items():
            if entry.ttl and (current_time - entry.created_at) > entry.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            await self._remove_entry(key)
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.running:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                async with self.lock:
                    await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception:
                pass
    
    async def _persist_loop(self):
        """Background persistence loop"""
        while self.running:
            try:
                await asyncio.sleep(self.config.persist_interval)
                await self.persist()
            except asyncio.CancelledError:
                break
            except Exception:
                pass
    
    async def persist(self) -> bool:
        """Persist data to disk"""
        if not self.config.persistence_path:
            return False
        
        try:
            async with self.lock:
                # Prepare data for persistence
                persist_data = {}
                for key, entry in self.data.items():
                    persist_data[key] = {
                        'value': entry.value,
                        'created_at': entry.created_at,
                        'ttl': entry.ttl,
                        'compressed': entry.compressed,
                        'metadata': entry.metadata
                    }
                
                # Write to file
                self.config.persistence_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(self.config.persistence_path, 'wb') as f:
                    pickle.dump(persist_data, f)
                
                self.stats.persistence_count += 1
                return True
        
        except Exception:
            return False
    
    async def load(self) -> bool:
        """Load data from disk"""
        if not self.config.persistence_path or not self.config.persistence_path.exists():
            return False
        
        try:
            with open(self.config.persistence_path, 'rb') as f:
                persist_data = pickle.load(f)
            
            async with self.lock:
                for key, data in persist_data.items():
                    entry = CacheEntry(
                        key=key,
                        value=data['value'],
                        created_at=data['created_at'],
                        ttl=data['ttl'],
                        compressed=data['compressed'],
                        metadata=data['metadata']
                    )
                    entry.size = len(entry.value)
                    
                    # Check if still valid
                    if entry.ttl and (time.time() - entry.created_at) > entry.ttl:
                        continue
                    
                    self.data[key] = entry
                    self.stats.memory_usage += entry.size
                    self.stats.total_items += 1
                    
                    # Update access tracking
                    if self.config.cache_policy == CachePolicy.LRU:
                        self.access_order[key] = True
                    elif self.config.cache_policy == CachePolicy.LFU:
                        self.frequency[key] = 1
            
            return True
        
        except Exception:
            return False
    
    def get_statistics(self) -> StorageStats:
        """Get storage statistics"""
        return self.stats


class SQLiteStorage(StorageBackend[T]):
    """SQLite-based persistent storage"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.db_path = config.persistence_path or Path("storage.db")
        self.connection = None
        self.lock = asyncio.Lock()
        self.stats = StorageStats()
    
    async def initialize(self):
        """Initialize SQLite database"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.connection.execute('''
            CREATE TABLE IF NOT EXISTS storage (
                key TEXT PRIMARY KEY,
                value BLOB,
                created_at REAL,
                ttl REAL,
                metadata TEXT
            )
        ''')
        self.connection.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON storage(created_at)')
        self.connection.execute('CREATE INDEX IF NOT EXISTS idx_ttl ON storage(ttl)')
        self.connection.commit()
    
    async def get(self, key: str) -> Optional[T]:
        """Get value by key"""
        async with self.lock:
            cursor = self.connection.cursor()
            cursor.execute('SELECT value, created_at, ttl FROM storage WHERE key = ?', (key,))
            row = cursor.fetchone()
            
            if not row:
                self.stats.miss_count += 1
                return None
            
            value_blob, created_at, ttl = row
            
            # Check TTL
            if ttl and (time.time() - created_at) > ttl:
                cursor.execute('DELETE FROM storage WHERE key = ?', (key,))
                self.connection.commit()
                self.stats.miss_count += 1
                return None
            
            try:
                value = pickle.loads(value_blob)
                self.stats.hit_count += 1
                return value
            except Exception:
                self.stats.miss_count += 1
                return None
    
    async def set(self, key: str, value: T, ttl: Optional[float] = None) -> bool:
        """Set value with optional TTL"""
        try:
            async with self.lock:
                value_blob = pickle.dumps(value)
                created_at = time.time()
                
                cursor = self.connection.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO storage (key, value, created_at, ttl)
                    VALUES (?, ?, ?, ?)
                ''', (key, value_blob, created_at, ttl))
                self.connection.commit()
                
                self.stats.total_items += 1
                return True
        except Exception:
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key"""
        async with self.lock:
            cursor = self.connection.cursor()
            cursor.execute('DELETE FROM storage WHERE key = ?', (key,))
            deleted = cursor.rowcount > 0
            self.connection.commit()
            return deleted
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        async with self.lock:
            cursor = self.connection.cursor()
            cursor.execute('SELECT 1 FROM storage WHERE key = ? LIMIT 1', (key,))
            return cursor.fetchone() is not None
    
    async def clear(self) -> bool:
        """Clear all data"""
        async with self.lock:
            cursor = self.connection.cursor()
            cursor.execute('DELETE FROM storage')
            self.connection.commit()
            self.stats = StorageStats()
            return True
    
    async def keys(self) -> List[str]:
        """Get all keys"""
        async with self.lock:
            cursor = self.connection.cursor()
            cursor.execute('SELECT key FROM storage')
            return [row[0] for row in cursor.fetchall()]
    
    async def size(self) -> int:
        """Get number of items"""
        async with self.lock:
            cursor = self.connection.cursor()
            cursor.execute('SELECT COUNT(*) FROM storage')
            return cursor.fetchone()[0]


class StorageManager:
    """Advanced storage management system"""
    
    def __init__(self, app):
        self.app = app
        self.storages = {}  # storage_name -> StorageBackend
        self.configs = {}  # storage_name -> StorageConfig
        self.default_storage = None
        
        # Monitoring
        self.monitor_task = None
        self.running = False
        self.monitor_interval = 60.0
        
        self.lock = asyncio.Lock()
    
    async def setup(self) -> bool:
        """Setup storage manager"""
        try:
            # Create default memory storage
            default_config = StorageConfig(
                name="default",
                storage_type=StorageType.MEMORY,
                max_size=10000,
                cache_policy=CachePolicy.LRU
            )
            
            await self.create_storage("default", default_config)
            self.default_storage = "default"
            
            # Create persistent storage
            persistent_config = StorageConfig(
                name="persistent",
                storage_type=StorageType.SQLITE,
                persistence_path=self.app.file_manager.data_dir / "storage.db"
            )
            
            await self.create_storage("persistent", persistent_config)
            
            self.running = True
            self.monitor_task = asyncio.create_task(self._monitor_storages())
            
            self.app.logger.info("Storage manager initialized successfully")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "StorageManager.setup")
            return False
    
    async def create_storage(self, name: str, config: StorageConfig) -> bool:
        """Create storage backend"""
        try:
            async with self.lock:
                if name in self.storages:
                    return False
                
                # Create storage backend
                if config.storage_type == StorageType.MEMORY:
                    storage = MemoryStorage(config)
                    await storage.start()
                elif config.storage_type == StorageType.SQLITE:
                    storage = SQLiteStorage(config)
                    await storage.initialize()
                else:
                    raise ValueError(f"Unsupported storage type: {config.storage_type}")
                
                self.storages[name] = storage
                self.configs[name] = config
                
                self.app.logger.info(f"Storage '{name}' created with type {config.storage_type.name}")
                return True
        
        except Exception as e:
            self.app.error_handler.handle_error(e, f"StorageManager.create_storage({name})")
            return False
    
    async def get(self, key: str, storage_name: str = None) -> Optional[Any]:
        """Get value from storage"""
        storage_name = storage_name or self.default_storage
        
        if storage_name not in self.storages:
            return None
        
        return await self.storages[storage_name].get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None, 
                 storage_name: str = None) -> bool:
        """Set value in storage"""
        storage_name = storage_name or self.default_storage
        
        if storage_name not in self.storages:
            return False
        
        return await self.storages[storage_name].set(key, value, ttl)
    
    async def delete(self, key: str, storage_name: str = None) -> bool:
        """Delete key from storage"""
        storage_name = storage_name or self.default_storage
        
        if storage_name not in self.storages:
            return False
        
        return await self.storages[storage_name].delete(key)
    
    async def exists(self, key: str, storage_name: str = None) -> bool:
        """Check if key exists in storage"""
        storage_name = storage_name or self.default_storage
        
        if storage_name not in self.storages:
            return False
        
        return await self.storages[storage_name].exists(key)
    
    async def _monitor_storages(self):
        """Monitor storage performance"""
        while self.running:
            try:
                await asyncio.sleep(self.monitor_interval)
                
                # Collect statistics from all storages
                total_stats = {
                    'total_storages': len(self.storages),
                    'total_items': 0,
                    'total_memory': 0,
                    'total_hits': 0,
                    'total_misses': 0
                }
                
                for name, storage in self.storages.items():
                    if hasattr(storage, 'get_statistics'):
                        stats = storage.get_statistics()
                        total_stats['total_items'] += stats.total_items
                        total_stats['total_memory'] += stats.memory_usage
                        total_stats['total_hits'] += stats.hit_count
                        total_stats['total_misses'] += stats.miss_count
                
                # Log statistics
                if self.app.logger:
                    await self.app.logger.debug_async(
                        "Storage manager statistics",
                        extra=total_stats
                    )
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.app.error_handler.handle_error(e, "StorageManager._monitor_storages")
    
    def get_storage_statistics(self, storage_name: str = None) -> Dict[str, Any]:
        """Get storage statistics"""
        if storage_name:
            if storage_name in self.storages and hasattr(self.storages[storage_name], 'get_statistics'):
                return self.storages[storage_name].get_statistics().__dict__
            return {}
        else:
            stats = {}
            for name, storage in self.storages.items():
                if hasattr(storage, 'get_statistics'):
                    stats[name] = storage.get_statistics().__dict__
            return stats
    
    def list_storages(self) -> List[str]:
        """List all storage names"""
        return list(self.storages.keys())
    
    async def shutdown_async(self):
        """Shutdown storage manager"""
        try:
            self.running = False
            
            # Stop monitoring
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Stop all storages
            for storage in self.storages.values():
                if hasattr(storage, 'stop'):
                    await storage.stop()
            
            self.app.logger.info("Storage manager shutdown completed")
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "StorageManager.shutdown_async")


class CacheManager:
    """High-level cache management interface"""
    
    def __init__(self, app):
        self.app = app
        self.storage_manager = None
        self.cache_strategies = {}
        self.default_ttl = 3600.0  # 1 hour
    
    async def setup(self) -> bool:
        """Setup cache manager"""
        try:
            self.storage_manager = StorageManager(self.app)
            success = await self.storage_manager.setup()
            
            if success:
                # Create specialized cache storages
                await self._create_specialized_caches()
            
            self.app.logger.info("Cache manager initialized successfully")
            return success
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "CacheManager.setup")
            return False
    
    async def _create_specialized_caches(self):
        """Create specialized cache storages"""
        # Session cache (short-lived)
        session_config = StorageConfig(
            name="session",
            storage_type=StorageType.MEMORY,
            max_size=1000,
            cache_policy=CachePolicy.TTL,
            ttl_default=1800.0  # 30 minutes
        )
        await self.storage_manager.create_storage("session", session_config)
        
        # API response cache (medium-lived)
        api_config = StorageConfig(
            name="api_cache",
            storage_type=StorageType.MEMORY,
            max_size=5000,
            cache_policy=CachePolicy.LRU,
            ttl_default=3600.0  # 1 hour
        )
        await self.storage_manager.create_storage("api_cache", api_config)
        
        # File cache (long-lived)
        file_config = StorageConfig(
            name="file_cache",
            storage_type=StorageType.MEMORY,
            max_size=500,
            cache_policy=CachePolicy.LFU,
            compression=CompressionType.LZ4
        )
        await self.storage_manager.create_storage("file_cache", file_config)
    
    async def get(self, key: str, cache_type: str = "default") -> Optional[Any]:
        """Get value from cache"""
        return await self.storage_manager.get(key, cache_type)
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None, 
                 cache_type: str = "default") -> bool:
        """Set value in cache"""
        ttl = ttl or self.default_ttl
        return await self.storage_manager.set(key, value, ttl, cache_type)
    
    async def delete(self, key: str, cache_type: str = "default") -> bool:
        """Delete key from cache"""
        return await self.storage_manager.delete(key, cache_type)
    
    async def cached(self, key: str, func: Callable, ttl: Optional[float] = None,
                    cache_type: str = "default") -> Any:
        """Get cached value or compute and cache"""
        # Try to get from cache
        value = await self.get(key, cache_type)
        
        if value is not None:
            return value
        
        # Compute value
        if asyncio.iscoroutinefunction(func):
            value = await func()
        else:
            value = func()
        
        # Cache the result
        await self.set(key, value, ttl, cache_type)
        
        return value
    
    def cache_decorator(self, key_func: Callable = None, ttl: Optional[float] = None,
                       cache_type: str = "default"):
        """Decorator for caching function results"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
                
                return await self.cached(cache_key, lambda: func(*args, **kwargs), ttl, cache_type)
            
            return wrapper
        return decorator
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.storage_manager:
            return self.storage_manager.get_storage_statistics()
        return {}
    
    async def shutdown_async(self):
        """Shutdown cache manager"""
        if self.storage_manager:
            await self.storage_manager.shutdown_async()