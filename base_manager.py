#!/usr/bin/env python3
"""
Advanced Base and Service Management System
Comprehensive service lifecycle, dependency management, and orchestration
"""

import asyncio
import time
import threading
import uuid
import json
import weakref
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Set, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import inspect
import traceback


class ServiceState(Enum):
    """Service states"""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    INITIALIZED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()
    DEGRADED = auto()


class ServiceType(Enum):
    """Service types"""
    CORE = auto()        # Core system services
    EXTENSION = auto()   # Extension services
    PLUGIN = auto()      # Plugin services
    EXTERNAL = auto()    # External services
    WORKER = auto()      # Worker services
    MONITOR = auto()     # Monitoring services


class DependencyType(Enum):
    """Dependency types"""
    REQUIRED = auto()    # Hard dependency - service cannot start without it
    OPTIONAL = auto()    # Soft dependency - service can start without it
    CIRCULAR = auto()    # Circular dependency (to be avoided)


@dataclass
class ServiceDependency:
    """Service dependency definition"""
    service_name: str
    dependency_type: DependencyType = DependencyType.REQUIRED
    wait_timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class ServiceConfig:
    """Service configuration"""
    name: str
    service_type: ServiceType = ServiceType.EXTENSION
    priority: int = 100  # Lower numbers = higher priority
    auto_start: bool = True
    restart_on_failure: bool = True
    max_restart_attempts: int = 3
    restart_delay: float = 5.0
    health_check_interval: float = 30.0
    dependencies: List[ServiceDependency] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceMetrics:
    """Service metrics"""
    name: str
    state: ServiceState
    uptime: float = 0.0
    restart_count: int = 0
    last_restart: Optional[datetime] = None
    health_checks_passed: int = 0
    health_checks_failed: int = 0
    last_health_check: Optional[datetime] = None
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class BaseService(ABC):
    """Base class for all services"""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.name = config.name
        self.state = ServiceState.UNINITIALIZED
        self.metrics = ServiceMetrics(name=config.name, state=self.state)
        
        # Lifecycle timestamps
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.stopped_at: Optional[datetime] = None
        
        # Internal state
        self._startup_task: Optional[asyncio.Task] = None
        self._shutdown_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Event callbacks
        self._state_change_callbacks: List[Callable] = []
        self._error_callbacks: List[Callable] = []
        
        # Lock for state changes
        self._state_lock = asyncio.Lock()
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize service (called once)"""
        pass
    
    @abstractmethod
    async def start(self) -> bool:
        """Start service"""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop service"""
        pass
    
    async def health_check(self) -> bool:
        """Health check (override if needed)"""
        return self.state == ServiceState.RUNNING
    
    async def cleanup(self):
        """Cleanup service resources (override if needed)"""
        pass
    
    async def restart(self) -> bool:
        """Restart service"""
        if await self.stop():
            return await self.start()
        return False
    
    def add_state_change_callback(self, callback: Callable):
        """Add state change callback"""
        self._state_change_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """Add error callback"""
        self._error_callbacks.append(callback)
    
    async def _set_state(self, new_state: ServiceState):
        """Set service state and notify callbacks"""
        async with self._state_lock:
            old_state = self.state
            self.state = new_state
            self.metrics.state = new_state
            
            # Update timestamps
            if new_state == ServiceState.RUNNING:
                self.started_at = datetime.now()
                self._running = True
            elif new_state in [ServiceState.STOPPED, ServiceState.FAILED]:
                self.stopped_at = datetime.now()
                self._running = False
            
            # Notify callbacks
            for callback in self._state_change_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(self, old_state, new_state)
                    else:
                        callback(self, old_state, new_state)
                except Exception as e:
                    await self._handle_error(e, "state_change_callback")
    
    async def _handle_error(self, error: Exception, context: str = ""):
        """Handle service error"""
        error_info = {
            'service': self.name,
            'context': context,
            'error': str(error),
            'traceback': traceback.format_exc()
        }
        
        # Notify error callbacks
        for callback in self._error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error_info)
                else:
                    callback(error_info)
            except Exception:
                pass  # Avoid infinite error loops
    
    def get_uptime(self) -> float:
        """Get service uptime in seconds"""
        if self.started_at and self.state == ServiceState.RUNNING:
            return (datetime.now() - self.started_at).total_seconds()
        return 0.0
    
    def get_metrics(self) -> ServiceMetrics:
        """Get service metrics"""
        self.metrics.uptime = self.get_uptime()
        return self.metrics


class BaseManager:
    """Base manager for component lifecycle"""
    
    def __init__(self, app):
        self.app = app
        self.components = {}  # component_name -> component_instance
        self.component_configs = {}  # component_name -> config
        self.lifecycle_hooks = defaultdict(list)  # hook_name -> List[callback]
        
        # Dependency tracking
        self.dependency_graph = defaultdict(set)  # component -> Set[dependencies]
        self.reverse_dependencies = defaultdict(set)  # component -> Set[dependents]
        
        # State tracking
        self.initialization_order = []
        self.shutdown_order = []
        self.failed_components = set()
        
        # Statistics
        self.stats = {
            'total_components': 0,
            'initialized_components': 0,
            'failed_components': 0,
            'startup_time': 0.0,
            'shutdown_time': 0.0
        }
        
        self.lock = asyncio.Lock()
    
    async def setup(self) -> bool:
        """Setup base manager"""
        try:
            # Register default lifecycle hooks
            self.add_lifecycle_hook('before_initialize', self._log_component_initialization)
            self.add_lifecycle_hook('after_initialize', self._update_component_stats)
            self.add_lifecycle_hook('on_failure', self._handle_component_failure)
            
            self.app.logger.info("Base manager initialized successfully")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "BaseManager.setup")
            return False
    
    def register_component(self, name: str, component: Any, 
                          dependencies: List[str] = None, 
                          config: Dict[str, Any] = None):
        """Register component with dependencies"""
        self.components[name] = component
        self.component_configs[name] = config or {}
        
        # Register dependencies
        if dependencies:
            for dep in dependencies:
                self.dependency_graph[name].add(dep)
                self.reverse_dependencies[dep].add(name)
        
        self.stats['total_components'] += 1
    
    def add_lifecycle_hook(self, hook_name: str, callback: Callable):
        """Add lifecycle hook"""
        self.lifecycle_hooks[hook_name].append(callback)
    
    async def _execute_hooks(self, hook_name: str, component_name: str, **kwargs):
        """Execute lifecycle hooks"""
        for callback in self.lifecycle_hooks[hook_name]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(component_name, **kwargs)
                else:
                    callback(component_name, **kwargs)
            except Exception as e:
                self.app.error_handler.handle_error(e, f"BaseManager.hook.{hook_name}")
    
    async def initialize_all(self) -> bool:
        """Initialize all components in dependency order"""
        start_time = time.time()
        
        try:
            # Calculate initialization order
            self.initialization_order = self._calculate_dependency_order()
            
            # Initialize components
            for component_name in self.initialization_order:
                if component_name in self.components:
                    success = await self._initialize_component(component_name)
                    if not success:
                        return False
            
            self.stats['startup_time'] = time.time() - start_time
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "BaseManager.initialize_all")
            return False
    
    async def _initialize_component(self, component_name: str) -> bool:
        """Initialize single component"""
        try:
            component = self.components[component_name]
            
            # Execute before hooks
            await self._execute_hooks('before_initialize', component_name, component=component)
            
            # Initialize component
            if hasattr(component, 'setup'):
                if asyncio.iscoroutinefunction(component.setup):
                    success = await component.setup()
                else:
                    success = component.setup()
            elif hasattr(component, 'initialize'):
                if asyncio.iscoroutinefunction(component.initialize):
                    success = await component.initialize()
                else:
                    success = component.initialize()
            else:
                success = True  # No setup method
            
            if success:
                self.stats['initialized_components'] += 1
                await self._execute_hooks('after_initialize', component_name, component=component)
            else:
                self.failed_components.add(component_name)
                self.stats['failed_components'] += 1
                await self._execute_hooks('on_failure', component_name, component=component)
            
            return success
            
        except Exception as e:
            self.failed_components.add(component_name)
            self.stats['failed_components'] += 1
            await self._execute_hooks('on_failure', component_name, component=self.components[component_name], error=e)
            self.app.error_handler.handle_error(e, f"BaseManager._initialize_component({component_name})")
            return False
    
    def _calculate_dependency_order(self) -> List[str]:
        """Calculate component initialization order using topological sort"""
        # Kahn's algorithm for topological sorting
        in_degree = defaultdict(int)
        
        # Calculate in-degrees
        for component in self.dependency_graph:
            for dependency in self.dependency_graph[component]:
                in_degree[component] += 1
        
        # Initialize queue with components that have no dependencies
        queue = deque([comp for comp in self.components.keys() if in_degree[comp] == 0])
        result = []
        
        while queue:
            component = queue.popleft()
            result.append(component)
            
            # Reduce in-degree for dependent components
            for dependent in self.reverse_dependencies[component]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Check for circular dependencies
        if len(result) != len(self.components):
            remaining = set(self.components.keys()) - set(result)
            self.app.logger.warning(f"Circular dependencies detected: {remaining}")
            result.extend(remaining)  # Add remaining components anyway
        
        return result
    
    async def shutdown_all(self) -> bool:
        """Shutdown all components in reverse order"""
        start_time = time.time()
        
        try:
            # Shutdown in reverse order
            self.shutdown_order = list(reversed(self.initialization_order))
            
            for component_name in self.shutdown_order:
                if component_name in self.components:
                    await self._shutdown_component(component_name)
            
            self.stats['shutdown_time'] = time.time() - start_time
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "BaseManager.shutdown_all")
            return False
    
    async def _shutdown_component(self, component_name: str):
        """Shutdown single component"""
        try:
            component = self.components[component_name]
            
            # Execute before hooks
            await self._execute_hooks('before_shutdown', component_name, component=component)
            
            # Shutdown component
            if hasattr(component, 'cleanup'):
                if asyncio.iscoroutinefunction(component.cleanup):
                    await component.cleanup()
                else:
                    component.cleanup()
            elif hasattr(component, 'shutdown'):
                if asyncio.iscoroutinefunction(component.shutdown):
                    await component.shutdown()
                else:
                    component.shutdown()
            
            # Execute after hooks
            await self._execute_hooks('after_shutdown', component_name, component=component)
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"BaseManager._shutdown_component({component_name})")
    
    # Default lifecycle hook implementations
    async def _log_component_initialization(self, component_name: str, **kwargs):
        """Log component initialization"""
        self.app.logger.debug(f"Initializing component: {component_name}")
    
    async def _update_component_stats(self, component_name: str, **kwargs):
        """Update component statistics"""
        pass  # Stats are updated in _initialize_component
    
    async def _handle_component_failure(self, component_name: str, **kwargs):
        """Handle component failure"""
        error = kwargs.get('error')
        self.app.logger.error(f"Component {component_name} failed to initialize: {error}")
    
    def get_component(self, name: str) -> Any:
        """Get component by name"""
        return self.components.get(name)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get base manager statistics"""
        return {
            'stats': self.stats.copy(),
            'initialization_order': self.initialization_order.copy(),
            'failed_components': list(self.failed_components),
            'dependency_count': len(self.dependency_graph),
            'total_hooks': sum(len(hooks) for hooks in self.lifecycle_hooks.values())
        }


class ServiceManager:
    """Advanced service management system"""
    
    def __init__(self, app):
        self.app = app
        self.services = {}  # service_name -> BaseService
        self.service_registry = {}  # service_name -> service_class
        self.service_instances = {}  # service_name -> service_instance
        
        # Service discovery
        self.service_discovery = {}  # service_name -> service_info
        self.health_monitors = {}  # service_name -> health_monitor_task
        
        # Configuration
        self.auto_restart = True
        self.health_check_interval = 30.0
        self.service_timeout = 60.0
        
        # Statistics
        self.stats = {
            'total_services': 0,
            'running_services': 0,
            'failed_services': 0,
            'restart_count': 0,
            'average_startup_time': 0.0,
            'total_startup_time': 0.0
        }
        
        # Monitoring
        self.monitor_task = None
        self.running = False
        
        self.lock = asyncio.Lock()
    
    async def setup(self) -> bool:
        """Setup service manager"""
        try:
            self.running = True
            
            # Start monitoring
            self.monitor_task = asyncio.create_task(self._monitor_services())
            
            self.app.logger.info("Service manager initialized successfully")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "ServiceManager.setup")
            return False
    
    def register_service(self, service_class: Type[BaseService], config: ServiceConfig):
        """Register service class"""
        self.service_registry[config.name] = service_class
        self.service_discovery[config.name] = {
            'class': service_class,
            'config': config,
            'state': ServiceState.UNINITIALIZED,
            'registered_at': datetime.now()
        }
        
        self.stats['total_services'] += 1
    
    async def create_service(self, service_name: str) -> bool:
        """Create service instance"""
        try:
            if service_name not in self.service_registry:
                return False
            
            service_class = self.service_registry[service_name]
            config = self.service_discovery[service_name]['config']
            
            # Create service instance
            service = service_class(config)
            
            # Add callbacks
            service.add_state_change_callback(self._on_service_state_change)
            service.add_error_callback(self._on_service_error)
            
            self.services[service_name] = service
            self.service_instances[service_name] = service
            
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"ServiceManager.create_service({service_name})")
            return False
    
    async def start_service(self, service_name: str) -> bool:
        """Start service"""
        try:
            if service_name not in self.services:
                if not await self.create_service(service_name):
                    return False
            
            service = self.services[service_name]
            start_time = time.time()
            
            # Check dependencies
            config = self.service_discovery[service_name]['config']
            for dependency in config.dependencies:
                if not await self._wait_for_dependency(dependency):
                    self.app.logger.error(f"Service {service_name} dependency {dependency.service_name} not available")
                    return False
            
            # Initialize if needed
            if service.state == ServiceState.UNINITIALIZED:
                await service._set_state(ServiceState.INITIALIZING)
                if not await service.initialize():
                    await service._set_state(ServiceState.FAILED)
                    return False
                await service._set_state(ServiceState.INITIALIZED)
            
            # Start service
            await service._set_state(ServiceState.STARTING)
            if await service.start():
                await service._set_state(ServiceState.RUNNING)
                
                # Start health monitoring
                if config.health_check_interval > 0:
                    self.health_monitors[service_name] = asyncio.create_task(
                        self._health_monitor(service_name)
                    )
                
                # Update statistics
                startup_time = time.time() - start_time
                self.stats['total_startup_time'] += startup_time
                self.stats['running_services'] += 1
                
                if self.stats['total_services'] > 0:
                    self.stats['average_startup_time'] = (
                        self.stats['total_startup_time'] / self.stats['total_services']
                    )
                
                return True
            else:
                await service._set_state(ServiceState.FAILED)
                self.stats['failed_services'] += 1
                return False
                
        except Exception as e:
            if service_name in self.services:
                await self.services[service_name]._set_state(ServiceState.FAILED)
            self.stats['failed_services'] += 1
            self.app.error_handler.handle_error(e, f"ServiceManager.start_service({service_name})")
            return False
    
    async def stop_service(self, service_name: str) -> bool:
        """Stop service"""
        try:
            if service_name not in self.services:
                return False
            
            service = self.services[service_name]
            
            # Stop health monitoring
            if service_name in self.health_monitors:
                self.health_monitors[service_name].cancel()
                del self.health_monitors[service_name]
            
            # Stop service
            await service._set_state(ServiceState.STOPPING)
            if await service.stop():
                await service._set_state(ServiceState.STOPPED)
                self.stats['running_services'] -= 1
                return True
            else:
                await service._set_state(ServiceState.FAILED)
                return False
                
        except Exception as e:
            self.app.error_handler.handle_error(e, f"ServiceManager.stop_service({service_name})")
            return False
    
    async def restart_service(self, service_name: str) -> bool:
        """Restart service"""
        if await self.stop_service(service_name):
            self.stats['restart_count'] += 1
            return await self.start_service(service_name)
        return False
    
    async def _wait_for_dependency(self, dependency: ServiceDependency) -> bool:
        """Wait for service dependency"""
        if dependency.dependency_type == DependencyType.OPTIONAL:
            return True
        
        for attempt in range(dependency.retry_attempts):
            if dependency.service_name in self.services:
                service = self.services[dependency.service_name]
                if service.state == ServiceState.RUNNING:
                    return True
            
            if attempt < dependency.retry_attempts - 1:
                await asyncio.sleep(dependency.retry_delay)
        
        return False
    
    async def _health_monitor(self, service_name: str):
        """Monitor service health"""
        service = self.services[service_name]
        config = self.service_discovery[service_name]['config']
        
        while service.state == ServiceState.RUNNING:
            try:
                await asyncio.sleep(config.health_check_interval)
                
                # Perform health check
                is_healthy = await service.health_check()
                
                if is_healthy:
                    service.metrics.health_checks_passed += 1
                    service.metrics.last_health_check = datetime.now()
                else:
                    service.metrics.health_checks_failed += 1
                    await service._set_state(ServiceState.DEGRADED)
                    
                    # Auto-restart if configured
                    if self.auto_restart and config.restart_on_failure:
                        await self._auto_restart_service(service_name)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                service.metrics.health_checks_failed += 1
                await service._handle_error(e, "health_check")
    
    async def _auto_restart_service(self, service_name: str):
        """Auto-restart failed service"""
        try:
            service = self.services[service_name]
            config = self.service_discovery[service_name]['config']
            
            if service.metrics.restart_count >= config.max_restart_attempts:
                self.app.logger.error(f"Service {service_name} exceeded max restart attempts")
                await service._set_state(ServiceState.FAILED)
                return
            
            self.app.logger.info(f"Auto-restarting service {service_name}")
            
            await asyncio.sleep(config.restart_delay)
            
            if await self.restart_service(service_name):
                service.metrics.restart_count += 1
                service.metrics.last_restart = datetime.now()
                self.app.logger.info(f"Service {service_name} restarted successfully")
            else:
                self.app.logger.error(f"Failed to restart service {service_name}")
                
        except Exception as e:
            self.app.error_handler.handle_error(e, f"ServiceManager._auto_restart_service({service_name})")
    
    async def _on_service_state_change(self, service: BaseService, 
                                     old_state: ServiceState, new_state: ServiceState):
        """Handle service state change"""
        self.service_discovery[service.name]['state'] = new_state
        
        self.app.logger.info(f"Service {service.name} state changed: {old_state.name} -> {new_state.name}")
        
        # Emit event
        if hasattr(self.app, 'event_system'):
            await self.app.event_system.emit('service.state_changed', {
                'service_name': service.name,
                'old_state': old_state.name,
                'new_state': new_state.name,
                'timestamp': time.time()
            })
    
    async def _on_service_error(self, error_info: Dict[str, Any]):
        """Handle service error"""
        self.app.logger.error(f"Service error: {error_info}")
        
        # Emit event
        if hasattr(self.app, 'event_system'):
            await self.app.event_system.emit('service.error', error_info)
    
    async def _monitor_services(self):
        """Monitor all services"""
        while self.running:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Update service discovery info
                for service_name, service in self.services.items():
                    self.service_discovery[service_name]['state'] = service.state
                
                # Log service statistics
                if self.app.logger:
                    await self.app.logger.debug_async(
                        "Service manager statistics",
                        extra=self.get_statistics()
                    )
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.app.error_handler.handle_error(e, "ServiceManager._monitor_services")
    
    async def start_all_services(self) -> bool:
        """Start all registered services"""
        success = True
        
        # Sort by priority (lower number = higher priority)
        sorted_services = sorted(
            self.service_discovery.items(),
            key=lambda x: x[1]['config'].priority
        )
        
        for service_name, service_info in sorted_services:
            if service_info['config'].auto_start:
                if not await self.start_service(service_name):
                    success = False
                    if service_info['config'].service_type == ServiceType.CORE:
                        break  # Stop if core service fails
        
        return success
    
    async def stop_all_services(self) -> bool:
        """Stop all services"""
        # Stop in reverse priority order
        sorted_services = sorted(
            self.service_discovery.items(),
            key=lambda x: x[1]['config'].priority,
            reverse=True
        )
        
        for service_name, _ in sorted_services:
            if service_name in self.services:
                await self.stop_service(service_name)
        
        return True
    
    def get_service_status(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get service status"""
        if service_name not in self.service_discovery:
            return None
        
        service_info = self.service_discovery[service_name]
        status = {
            'name': service_name,
            'state': service_info['state'].name,
            'registered_at': service_info['registered_at'].isoformat()
        }
        
        if service_name in self.services:
            service = self.services[service_name]
            metrics = service.get_metrics()
            status.update({
                'uptime': metrics.uptime,
                'restart_count': metrics.restart_count,
                'health_checks_passed': metrics.health_checks_passed,
                'health_checks_failed': metrics.health_checks_failed
            })
        
        return status
    
    def list_services(self) -> List[str]:
        """List all registered services"""
        return list(self.service_discovery.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service manager statistics"""
        return {
            'stats': self.stats.copy(),
            'services': {
                name: {
                    'state': info['state'].name,
                    'type': info['config'].service_type.name
                }
                for name, info in self.service_discovery.items()
            }
        }
    
    async def shutdown_all_async(self):
        """Shutdown service manager"""
        try:
            self.running = False
            
            # Stop monitoring
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Stop all health monitors
            for task in self.health_monitors.values():
                task.cancel()
            
            if self.health_monitors:
                await asyncio.gather(*self.health_monitors.values(), return_exceptions=True)
            
            # Stop all services
            await self.stop_all_services()
            
            self.app.logger.info("Service manager shutdown completed")
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "ServiceManager.shutdown_all_async")