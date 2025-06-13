#!/usr/bin/env python3
"""
Advanced Pool and Resource Management System
Comprehensive resource pooling, connection management, and object lifecycle
"""

import asyncio
import time
import threading
import uuid
import weakref
import gc
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Set, Tuple, Generic, TypeVar
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import concurrent.futures


T = TypeVar('T')  # Generic type for pooled resources


class PoolStatus(Enum):
    """Pool status"""
    INITIALIZING = auto()
    ACTIVE = auto()
    DRAINING = auto()
    CLOSED = auto()
    ERROR = auto()


class ResourceState(Enum):
    """Resource state"""
    AVAILABLE = auto()
    IN_USE = auto()
    INVALID = auto()
    EXPIRED = auto()
    DESTROYED = auto()


class PoolStrategy(Enum):
    """Pool management strategies"""
    FIFO = auto()        # First In, First Out
    LIFO = auto()        # Last In, First Out
    ROUND_ROBIN = auto() # Round-robin selection
    LEAST_USED = auto()  # Select least used resource
    RANDOM = auto()      # Random selection


@dataclass
class PoolConfig:
    """Pool configuration"""
    name: str
    min_size: int = 5
    max_size: int = 20
    initial_size: int = 5
    max_idle_time: float = 300.0  # 5 minutes
    max_lifetime: float = 3600.0  # 1 hour
    validation_interval: float = 60.0  # 1 minute
    strategy: PoolStrategy = PoolStrategy.FIFO
    pre_create: bool = True
    validate_on_borrow: bool = True
    validate_on_return: bool = False
    test_on_idle: bool = True
    auto_resize: bool = True
    resource_timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class ResourceInfo:
    """Resource information"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resource: Any = None
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    last_validated: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    state: ResourceState = ResourceState.AVAILABLE
    borrowed_at: Optional[datetime] = None
    borrowed_by: Optional[str] = None
    validation_failures: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResourceFactory(ABC, Generic[T]):
    """Abstract resource factory"""
    
    @abstractmethod
    async def create_resource(self) -> T:
        """Create new resource"""
        pass
    
    @abstractmethod
    async def validate_resource(self, resource: T) -> bool:
        """Validate resource"""
        pass
    
    @abstractmethod
    async def destroy_resource(self, resource: T):
        """Destroy resource"""
        pass
    
    async def reset_resource(self, resource: T) -> bool:
        """Reset resource to initial state"""
        return True


class GenericPool(Generic[T]):
    """Generic resource pool"""
    
    def __init__(self, factory: ResourceFactory[T], config: PoolConfig):
        self.factory = factory
        self.config = config
        self.status = PoolStatus.INITIALIZING
        
        # Resource management
        self.resources = {}  # resource_id -> ResourceInfo
        self.available_resources = deque()  # Available resource IDs
        self.borrowed_resources = {}  # resource_id -> borrower_info
        
        # Statistics
        self.stats = {
            'total_created': 0,
            'total_destroyed': 0,
            'total_borrowed': 0,
            'total_returned': 0,
            'current_size': 0,
            'current_available': 0,
            'current_borrowed': 0,
            'validation_failures': 0,
            'creation_failures': 0,
            'peak_size': 0,
            'average_borrow_time': 0.0,
            'total_borrow_time': 0.0
        }
        
        # Synchronization
        self.lock = asyncio.Lock()
        self.not_empty = asyncio.Condition(self.lock)
        self.not_full = asyncio.Condition(self.lock)
        
        # Background tasks
        self.maintenance_task = None
        self.running = False
        
        # Round-robin counter for strategy
        self.round_robin_index = 0
    
    async def initialize(self) -> bool:
        """Initialize pool"""
        try:
            async with self.lock:
                if self.status != PoolStatus.INITIALIZING:
                    return False
                
                # Create initial resources
                if self.config.pre_create:
                    for _ in range(self.config.initial_size):
                        await self._create_resource()
                
                self.status = PoolStatus.ACTIVE
                self.running = True
                
                # Start maintenance task
                self.maintenance_task = asyncio.create_task(self._maintenance_loop())
                
                return True
        
        except Exception as e:
            self.status = PoolStatus.ERROR
            raise RuntimeError(f"Pool initialization failed: {e}")
    
    async def borrow_resource(self, timeout: Optional[float] = None, 
                            borrower_id: str = None) -> T:
        """Borrow resource from pool"""
        start_time = time.time()
        timeout = timeout or self.config.resource_timeout
        
        async with self.not_empty:
            # Wait for available resource
            while not self.available_resources and self.status == PoolStatus.ACTIVE:
                # Try to create new resource if under max size
                if len(self.resources) < self.config.max_size:
                    try:
                        await self._create_resource()
                        break
                    except Exception:
                        pass  # Continue waiting
                
                # Wait for resource to become available
                try:
                    await asyncio.wait_for(
                        self.not_empty.wait(),
                        timeout=max(0.1, timeout - (time.time() - start_time))
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(f"Timeout waiting for resource from pool '{self.config.name}'")
            
            if self.status != PoolStatus.ACTIVE:
                raise RuntimeError(f"Pool '{self.config.name}' is not active")
            
            # Get resource based on strategy
            resource_id = await self._select_resource()
            resource_info = self.resources[resource_id]
            
            # Validate resource if required
            if self.config.validate_on_borrow:
                if not await self._validate_resource(resource_info):
                    # Remove invalid resource and try again
                    await self._destroy_resource(resource_id)
                    return await self.borrow_resource(timeout, borrower_id)
            
            # Mark as borrowed
            resource_info.state = ResourceState.IN_USE
            resource_info.borrowed_at = datetime.now()
            resource_info.borrowed_by = borrower_id
            resource_info.usage_count += 1
            resource_info.last_used = datetime.now()
            
            self.borrowed_resources[resource_id] = {
                'borrower_id': borrower_id,
                'borrowed_at': resource_info.borrowed_at
            }
            
            # Update statistics
            self.stats['total_borrowed'] += 1
            self.stats['current_borrowed'] += 1
            self.stats['current_available'] -= 1
            
            return resource_info.resource
    
    async def return_resource(self, resource: T, borrower_id: str = None) -> bool:
        """Return resource to pool"""
        try:
            async with self.lock:
                # Find resource info
                resource_info = None
                resource_id = None
                
                for rid, rinfo in self.resources.items():
                    if rinfo.resource is resource:
                        resource_info = rinfo
                        resource_id = rid
                        break
                
                if not resource_info:
                    return False
                
                # Validate resource if required
                if self.config.validate_on_return:
                    if not await self._validate_resource(resource_info):
                        await self._destroy_resource(resource_id)
                        return True
                
                # Reset resource if possible
                try:
                    await self.factory.reset_resource(resource)
                except Exception:
                    # If reset fails, destroy resource
                    await self._destroy_resource(resource_id)
                    return True
                
                # Mark as available
                resource_info.state = ResourceState.AVAILABLE
                resource_info.borrowed_at = None
                resource_info.borrowed_by = None
                
                # Calculate borrow time
                if resource_id in self.borrowed_resources:
                    borrow_info = self.borrowed_resources[resource_id]
                    borrow_time = (datetime.now() - borrow_info['borrowed_at']).total_seconds()
                    self.stats['total_borrow_time'] += borrow_time
                    self.stats['average_borrow_time'] = (
                        self.stats['total_borrow_time'] / self.stats['total_borrowed']
                    )
                    del self.borrowed_resources[resource_id]
                
                # Add back to available pool
                self.available_resources.append(resource_id)
                
                # Update statistics
                self.stats['total_returned'] += 1
                self.stats['current_borrowed'] -= 1
                self.stats['current_available'] += 1
            
            # Notify waiting borrowers
            async with self.not_empty:
                self.not_empty.notify()
            
            return True
        
        except Exception as e:
            # Log error and remove resource
            await self._destroy_resource_by_object(resource)
            return False
    
    async def _select_resource(self) -> str:
        """Select resource based on strategy"""
        if not self.available_resources:
            raise RuntimeError("No available resources")
        
        if self.config.strategy == PoolStrategy.FIFO:
            return self.available_resources.popleft()
        
        elif self.config.strategy == PoolStrategy.LIFO:
            return self.available_resources.pop()
        
        elif self.config.strategy == PoolStrategy.ROUND_ROBIN:
            if self.round_robin_index >= len(self.available_resources):
                self.round_robin_index = 0
            resource_id = self.available_resources[self.round_robin_index]
            del self.available_resources[self.round_robin_index]
            self.round_robin_index = (self.round_robin_index + 1) % max(1, len(self.available_resources))
            return resource_id
        
        elif self.config.strategy == PoolStrategy.LEAST_USED:
            # Find least used resource
            least_used_id = min(
                self.available_resources,
                key=lambda rid: self.resources[rid].usage_count
            )
            self.available_resources.remove(least_used_id)
            return least_used_id
        
        elif self.config.strategy == PoolStrategy.RANDOM:
            import random
            resource_id = random.choice(self.available_resources)
            self.available_resources.remove(resource_id)
            return resource_id
        
        else:
            return self.available_resources.popleft()
    
    async def _create_resource(self) -> str:
        """Create new resource"""
        try:
            resource = await self.factory.create_resource()
            
            resource_info = ResourceInfo(
                resource=resource,
                state=ResourceState.AVAILABLE
            )
            
            self.resources[resource_info.id] = resource_info
            self.available_resources.append(resource_info.id)
            
            # Update statistics
            self.stats['total_created'] += 1
            self.stats['current_size'] += 1
            self.stats['current_available'] += 1
            self.stats['peak_size'] = max(self.stats['peak_size'], self.stats['current_size'])
            
            return resource_info.id
        
        except Exception as e:
            self.stats['creation_failures'] += 1
            raise RuntimeError(f"Failed to create resource: {e}")
    
    async def _destroy_resource(self, resource_id: str):
        """Destroy resource"""
        if resource_id not in self.resources:
            return
        
        try:
            resource_info = self.resources[resource_id]
            
            # Destroy the actual resource
            await self.factory.destroy_resource(resource_info.resource)
            
            # Remove from collections
            if resource_id in self.available_resources:
                self.available_resources.remove(resource_id)
            if resource_id in self.borrowed_resources:
                del self.borrowed_resources[resource_id]
            
            # Update state and remove
            resource_info.state = ResourceState.DESTROYED
            del self.resources[resource_id]
            
            # Update statistics
            self.stats['total_destroyed'] += 1
            self.stats['current_size'] -= 1
            if resource_info.state == ResourceState.AVAILABLE:
                self.stats['current_available'] -= 1
            else:
                self.stats['current_borrowed'] -= 1
        
        except Exception as e:
            # Log error but continue cleanup
            pass
    
    async def _destroy_resource_by_object(self, resource: T):
        """Destroy resource by object reference"""
        for resource_id, resource_info in list(self.resources.items()):
            if resource_info.resource is resource:
                await self._destroy_resource(resource_id)
                break
    
    async def _validate_resource(self, resource_info: ResourceInfo) -> bool:
        """Validate resource"""
        try:
            is_valid = await self.factory.validate_resource(resource_info.resource)
            resource_info.last_validated = datetime.now()
            
            if not is_valid:
                resource_info.validation_failures += 1
                self.stats['validation_failures'] += 1
            else:
                resource_info.validation_failures = 0
            
            return is_valid
        
        except Exception:
            resource_info.validation_failures += 1
            self.stats['validation_failures'] += 1
            return False
    
    async def _maintenance_loop(self):
        """Background maintenance loop"""
        while self.running:
            try:
                await asyncio.sleep(self.config.validation_interval)
                await self._perform_maintenance()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue
                pass
    
    async def _perform_maintenance(self):
        """Perform pool maintenance"""
        current_time = datetime.now()
        resources_to_remove = []
        
        async with self.lock:
            # Check for expired or idle resources
            for resource_id, resource_info in self.resources.items():
                # Check lifetime
                if (current_time - resource_info.created_at).total_seconds() > self.config.max_lifetime:
                    if resource_info.state == ResourceState.AVAILABLE:
                        resources_to_remove.append(resource_id)
                        continue
                
                # Check idle time
                if (resource_info.state == ResourceState.AVAILABLE and
                    (current_time - resource_info.last_used).total_seconds() > self.config.max_idle_time):
                    # Keep minimum pool size
                    if len(self.resources) > self.config.min_size:
                        resources_to_remove.append(resource_id)
                        continue
                
                # Validate idle resources
                if (self.config.test_on_idle and 
                    resource_info.state == ResourceState.AVAILABLE):
                    try:
                        if not await self._validate_resource(resource_info):
                            resources_to_remove.append(resource_id)
                    except Exception:
                        resources_to_remove.append(resource_id)
            
            # Remove invalid/expired resources
            for resource_id in resources_to_remove:
                await self._destroy_resource(resource_id)
            
            # Auto-resize: ensure minimum size
            if (self.config.auto_resize and 
                len(self.resources) < self.config.min_size):
                needed = self.config.min_size - len(self.resources)
                for _ in range(needed):
                    try:
                        await self._create_resource()
                    except Exception:
                        break  # Stop if creation fails
    
    async def close(self):
        """Close pool"""
        async with self.lock:
            if self.status == PoolStatus.CLOSED:
                return
            
            self.status = PoolStatus.DRAINING
            self.running = False
            
            # Cancel maintenance task
            if self.maintenance_task:
                self.maintenance_task.cancel()
                try:
                    await self.maintenance_task
                except asyncio.CancelledError:
                    pass
            
            # Wait for borrowed resources to be returned (with timeout)
            wait_start = time.time()
            while self.borrowed_resources and (time.time() - wait_start) < 30:
                await asyncio.sleep(0.1)
            
            # Destroy all resources
            for resource_id in list(self.resources.keys()):
                await self._destroy_resource(resource_id)
            
            self.status = PoolStatus.CLOSED
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            'config': {
                'name': self.config.name,
                'min_size': self.config.min_size,
                'max_size': self.config.max_size,
                'strategy': self.config.strategy.name
            },
            'status': self.status.name,
            'stats': self.stats.copy(),
            'current_state': {
                'total_resources': len(self.resources),
                'available_resources': len(self.available_resources),
                'borrowed_resources': len(self.borrowed_resources)
            }
        }


class ConnectionFactory(ResourceFactory[object]):
    """Generic connection factory"""
    
    def __init__(self, connection_func: Callable, *args, **kwargs):
        self.connection_func = connection_func
        self.args = args
        self.kwargs = kwargs
    
    async def create_resource(self) -> object:
        """Create connection"""
        if asyncio.iscoroutinefunction(self.connection_func):
            return await self.connection_func(*self.args, **self.kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.connection_func, *self.args, **self.kwargs)
    
    async def validate_resource(self, resource: object) -> bool:
        """Validate connection"""
        try:
            # Basic validation - check if object has common connection methods
            if hasattr(resource, 'ping'):
                if asyncio.iscoroutinefunction(resource.ping):
                    await resource.ping()
                else:
                    resource.ping()
            elif hasattr(resource, 'is_connected'):
                return resource.is_connected()
            elif hasattr(resource, 'closed'):
                return not resource.closed
            return True
        except Exception:
            return False
    
    async def destroy_resource(self, resource: object):
        """Destroy connection"""
        try:
            if hasattr(resource, 'close'):
                if asyncio.iscoroutinefunction(resource.close):
                    await resource.close()
                else:
                    resource.close()
        except Exception:
            pass


class PoolManager:
    """Pool manager for multiple resource pools"""
    
    def __init__(self, app):
        self.app = app
        self.pools = {}  # pool_name -> GenericPool
        self.pool_configs = {}  # pool_name -> PoolConfig
        self.pool_factories = {}  # pool_name -> ResourceFactory
        
        # Monitoring
        self.monitor_task = None
        self.running = False
        self.monitor_interval = 60.0  # 1 minute
        
        self.lock = asyncio.Lock()
    
    async def setup(self) -> bool:
        """Setup pool manager"""
        try:
            self.running = True
            
            # Start monitoring
            self.monitor_task = asyncio.create_task(self._monitor_pools())
            
            self.app.logger.info("Pool manager initialized successfully")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "PoolManager.setup")
            return False
    
    async def create_pool(self, name: str, factory: ResourceFactory, 
                         config: PoolConfig = None) -> bool:
        """Create resource pool"""
        try:
            async with self.lock:
                if name in self.pools:
                    return False
                
                if config is None:
                    config = PoolConfig(name=name)
                
                pool = GenericPool(factory, config)
                await pool.initialize()
                
                self.pools[name] = pool
                self.pool_configs[name] = config
                self.pool_factories[name] = factory
                
                self.app.logger.info(f"Resource pool '{name}' created")
                return True
        
        except Exception as e:
            self.app.error_handler.handle_error(e, f"PoolManager.create_pool({name})")
            return False
    
    async def create_connection_pool(self, name: str, connection_func: Callable,
                                   *args, config: PoolConfig = None, **kwargs) -> bool:
        """Create connection pool"""
        factory = ConnectionFactory(connection_func, *args, **kwargs)
        return await self.create_pool(name, factory, config)
    
    async def get_resource(self, pool_name: str, timeout: Optional[float] = None,
                          borrower_id: str = None):
        """Get resource from pool"""
        if pool_name not in self.pools:
            raise ValueError(f"Pool '{pool_name}' not found")
        
        return await self.pools[pool_name].borrow_resource(timeout, borrower_id)
    
    async def return_resource(self, pool_name: str, resource, borrower_id: str = None) -> bool:
        """Return resource to pool"""
        if pool_name not in self.pools:
            return False
        
        return await self.pools[pool_name].return_resource(resource, borrower_id)
    
    async def close_pool(self, pool_name: str) -> bool:
        """Close specific pool"""
        async with self.lock:
            if pool_name not in self.pools:
                return False
            
            await self.pools[pool_name].close()
            del self.pools[pool_name]
            del self.pool_configs[pool_name]
            del self.pool_factories[pool_name]
            
            self.app.logger.info(f"Resource pool '{pool_name}' closed")
            return True
    
    async def _monitor_pools(self):
        """Monitor all pools"""
        while self.running:
            try:
                await asyncio.sleep(self.monitor_interval)
                
                # Collect statistics from all pools
                total_stats = {
                    'total_pools': len(self.pools),
                    'total_resources': 0,
                    'total_borrowed': 0,
                    'total_available': 0
                }
                
                for pool_name, pool in self.pools.items():
                    stats = pool.get_statistics()
                    total_stats['total_resources'] += stats['current_state']['total_resources']
                    total_stats['total_borrowed'] += stats['current_state']['borrowed_resources']
                    total_stats['total_available'] += stats['current_state']['available_resources']
                
                # Log aggregated statistics
                if self.app.logger:
                    await self.app.logger.debug_async(
                        "Pool manager statistics",
                        extra=total_stats
                    )
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.app.error_handler.handle_error(e, "PoolManager._monitor_pools")
    
    def get_pool_statistics(self, pool_name: str = None) -> Dict[str, Any]:
        """Get pool statistics"""
        if pool_name:
            if pool_name in self.pools:
                return self.pools[pool_name].get_statistics()
            return {}
        else:
            # Return all pool statistics
            stats = {}
            for name, pool in self.pools.items():
                stats[name] = pool.get_statistics()
            return stats
    
    def list_pools(self) -> List[str]:
        """List all pool names"""
        return list(self.pools.keys())
    
    async def shutdown_async(self):
        """Shutdown pool manager"""
        try:
            self.running = False
            
            # Stop monitoring
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Close all pools
            for pool_name in list(self.pools.keys()):
                await self.close_pool(pool_name)
            
            self.app.logger.info("Pool manager shutdown completed")
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "PoolManager.shutdown_async")


class ResourcePool:
    """Simplified resource pool interface"""
    
    def __init__(self, app):
        self.app = app
        self.pool_manager = None
    
    async def setup(self) -> bool:
        """Setup resource pool"""
        try:
            self.pool_manager = PoolManager(self.app)
            return await self.pool_manager.setup()
        
        except Exception as e:
            self.app.error_handler.handle_error(e, "ResourcePool.setup")
            return False
    
    async def create_http_connection_pool(self, name: str = "http_pool", 
                                        max_connections: int = 20) -> bool:
        """Create HTTP connection pool"""
        try:
            import aiohttp
            
            async def create_http_session():
                connector = aiohttp.TCPConnector(limit=max_connections)
                return aiohttp.ClientSession(connector=connector)
            
            config = PoolConfig(
                name=name,
                min_size=2,
                max_size=max_connections,
                initial_size=5
            )
            
            return await self.pool_manager.create_connection_pool(
                name, create_http_session, config=config
            )
        
        except ImportError:
            self.app.logger.warning("aiohttp not available for HTTP connection pool")
            return False
        except Exception as e:
            self.app.error_handler.handle_error(e, f"ResourcePool.create_http_connection_pool({name})")
            return False
    
    async def create_redis_connection_pool(self, name: str = "redis_pool",
                                         redis_url: str = "redis://localhost:6379") -> bool:
        """Create Redis connection pool"""
        try:
            import redis.asyncio as redis
            
            async def create_redis_connection():
                return await redis.from_url(redis_url)
            
            config = PoolConfig(
                name=name,
                min_size=5,
                max_size=20,
                initial_size=5
            )
            
            return await self.pool_manager.create_connection_pool(
                name, create_redis_connection, config=config
            )
        
        except ImportError:
            self.app.logger.warning("redis not available for Redis connection pool")
            return False
        except Exception as e:
            self.app.error_handler.handle_error(e, f"ResourcePool.create_redis_connection_pool({name})")
            return False
    
    async def get_connection(self, pool_name: str, timeout: Optional[float] = None):
        """Get connection from pool"""
        return await self.pool_manager.get_resource(pool_name, timeout)
    
    async def return_connection(self, pool_name: str, connection) -> bool:
        """Return connection to pool"""
        return await self.pool_manager.return_resource(pool_name, connection)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get resource pool statistics"""
        if self.pool_manager:
            return self.pool_manager.get_pool_statistics()
        return {}
    
    async def shutdown_async(self):
        """Shutdown resource pool"""
        if self.pool_manager:
            await self.pool_manager.shutdown_async()


# Context manager for automatic resource management
class PooledResource:
    """Context manager for pooled resources"""
    
    def __init__(self, pool_manager: PoolManager, pool_name: str, 
                 timeout: Optional[float] = None, borrower_id: str = None):
        self.pool_manager = pool_manager
        self.pool_name = pool_name
        self.timeout = timeout
        self.borrower_id = borrower_id
        self.resource = None
    
    async def __aenter__(self):
        self.resource = await self.pool_manager.get_resource(
            self.pool_name, self.timeout, self.borrower_id
        )
        return self.resource
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.resource:
            await self.pool_manager.return_resource(
                self.pool_name, self.resource, self.borrower_id
            )
            self.resource = None