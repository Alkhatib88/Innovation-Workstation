#!/usr/bin/env python3
"""
Advanced AsyncIO Management System
Comprehensive async task management, event loops, and coroutine orchestration
"""

import asyncio
import time
import threading
import uuid
import inspect
import weakref
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Set, Tuple, Coroutine
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
from datetime import datetime, timedelta
import traceback
import sys
import gc


class AsyncTaskStatus(Enum):
    """Async task status"""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


class AsyncTaskPriority(Enum):
    """Async task priority"""
    LOWEST = 1
    LOW = 2
    NORMAL = 3
    HIGH = 4
    HIGHEST = 5
    CRITICAL = 6


class LoopPolicy(Enum):
    """Event loop policy types"""
    DEFAULT = auto()
    UVLOOP = auto()
    WINLOOP = auto()
    CUSTOM = auto()


@dataclass
class AsyncTask:
    """Async task data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    coroutine: Optional[Coroutine] = None
    function: Optional[Callable] = None
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: AsyncTaskPriority = AsyncTaskPriority.NORMAL
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 0
    retry_delay: float = 1.0
    status: AsyncTaskStatus = AsyncTaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    task_obj: Optional[asyncio.Task] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)


@dataclass
class LoopInfo:
    """Event loop information"""
    id: str
    thread_id: int
    thread_name: str
    is_running: bool = False
    is_main_loop: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    task_count: int = 0
    total_tasks_run: int = 0
    cpu_time: float = 0.0
    loop_obj: Optional[asyncio.AbstractEventLoop] = None


class AsyncTaskGroup:
    """Group of related async tasks"""
    
    def __init__(self, name: str, max_concurrent: int = 0):
        self.name = name
        self.id = str(uuid.uuid4())
        self.tasks = {}  # task_id -> AsyncTask
        self.max_concurrent = max_concurrent  # 0 = unlimited
        self.running_tasks = set()
        self.completed_tasks = set()
        self.failed_tasks = set()
        self.cancelled_tasks = set()
        self.created_at = datetime.now()
        self.semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        self.results = {}  # task_id -> result
        self.errors = {}  # task_id -> error
    
    async def add_task(self, task: AsyncTask) -> str:
        """Add task to group"""
        self.tasks[task.id] = task
        return task.id
    
    async def wait_for_completion(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for all tasks in group to complete"""
        if not self.tasks:
            return {'completed': [], 'failed': [], 'cancelled': []}
        
        # Create asyncio tasks for all pending tasks
        asyncio_tasks = []
        for task in self.tasks.values():
            if task.task_obj:
                asyncio_tasks.append(task.task_obj)
        
        if not asyncio_tasks:
            return {'completed': [], 'failed': [], 'cancelled': []}
        
        try:
            # Wait for all tasks
            done, pending = await asyncio.wait(
                asyncio_tasks,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Cancel pending tasks if timeout occurred
            for task in pending:
                task.cancel()
            
            return {
                'completed': list(self.completed_tasks),
                'failed': list(self.failed_tasks),
                'cancelled': list(self.cancelled_tasks)
            }
        
        except Exception as e:
            return {'error': str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get group statistics"""
        return {
            'name': self.name,
            'id': self.id,
            'total_tasks': len(self.tasks),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'cancelled_tasks': len(self.cancelled_tasks),
            'success_rate': len(self.completed_tasks) / max(len(self.tasks), 1)
        }


class AsyncIOManager:
    """AsyncIO event loop manager"""
    
    def __init__(self, app):
        self.app = app
        self.main_loop = None
        self.loops = {}  # loop_id -> LoopInfo
        self.loop_policies = {}
        self.running = False
        
        # Loop configuration
        self.debug_mode = False
        self.slow_callback_duration = 0.1
        self.preferred_policy = LoopPolicy.DEFAULT
        
        # Statistics
        self.stats = {
            'total_loops': 0,
            'active_loops': 0,
            'total_callback_time': 0.0,
            'slow_callbacks': 0,
            'loop_exceptions': 0
        }
        
        self.lock = asyncio.Lock()
    
    async def setup(self) -> bool:
        """Setup AsyncIO manager"""
        try:
            # Get current event loop
            try:
                self.main_loop = asyncio.get_running_loop()
            except RuntimeError:
                self.main_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.main_loop)
            
            # Configure loop
            await self._configure_loop(self.main_loop)
            
            # Register main loop
            main_loop_info = LoopInfo(
                id="main",
                thread_id=threading.get_ident(),
                thread_name=threading.current_thread().name,
                is_running=True,
                is_main_loop=True,
                loop_obj=self.main_loop
            )
            
            self.loops["main"] = main_loop_info
            self.stats['total_loops'] += 1
            self.stats['active_loops'] += 1
            
            self.running = True
            
            self.app.logger.info("AsyncIO manager initialized successfully")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "AsyncIOManager.setup")
            return False
    
    async def _configure_loop(self, loop: asyncio.AbstractEventLoop):
        """Configure event loop"""
        try:
            # Set debug mode
            loop.set_debug(self.debug_mode)
            
            # Set slow callback duration
            if hasattr(loop, 'slow_callback_duration'):
                loop.slow_callback_duration = self.slow_callback_duration
            
            # Set exception handler
            loop.set_exception_handler(self._loop_exception_handler)
            
            # Try to use uvloop if available and preferred
            if self.preferred_policy == LoopPolicy.UVLOOP:
                try:
                    import uvloop
                    if not isinstance(loop, uvloop.Loop):
                        self.app.logger.info("Switching to uvloop for better performance")
                        uvloop.install()
                except ImportError:
                    self.app.logger.warning("uvloop not available, using default event loop")
        
        except Exception as e:
            self.app.logger.warning(f"Event loop configuration failed: {e}")
    
    def _loop_exception_handler(self, loop, context):
        """Handle loop exceptions"""
        self.stats['loop_exceptions'] += 1
        
        exception = context.get('exception')
        if exception:
            self.app.error_handler.handle_error(
                exception, 
                f"AsyncIO Loop Exception: {context.get('message', 'Unknown')}"
            )
        else:
            self.app.logger.error(f"AsyncIO Loop Error: {context}")
    
    async def create_loop(self, name: str, in_thread: bool = False) -> Optional[str]:
        """Create new event loop"""
        try:
            if in_thread:
                # Create loop in separate thread
                loop_id = await self._create_threaded_loop(name)
            else:
                # Create loop in current thread (not recommended for most cases)
                loop = asyncio.new_event_loop()
                await self._configure_loop(loop)
                
                loop_info = LoopInfo(
                    id=name,
                    thread_id=threading.get_ident(),
                    thread_name=threading.current_thread().name,
                    loop_obj=loop
                )
                
                self.loops[name] = loop_info
                loop_id = name
            
            if loop_id:
                self.stats['total_loops'] += 1
                self.stats['active_loops'] += 1
                self.app.logger.info(f"Event loop '{name}' created")
            
            return loop_id
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"AsyncIOManager.create_loop({name})")
            return None
    
    async def _create_threaded_loop(self, name: str) -> Optional[str]:
        """Create event loop in separate thread"""
        loop_id = None
        loop_ready = threading.Event()
        
        def loop_thread():
            nonlocal loop_id
            try:
                # Create new loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Configure loop
                asyncio.create_task(self._configure_loop_sync(loop))
                
                # Create loop info
                loop_info = LoopInfo(
                    id=name,
                    thread_id=threading.get_ident(),
                    thread_name=threading.current_thread().name,
                    is_running=True,
                    loop_obj=loop
                )
                
                self.loops[name] = loop_info
                loop_id = name
                loop_ready.set()
                
                # Run loop forever
                loop.run_forever()
                
            except Exception as e:
                self.app.logger.error(f"Threaded loop error: {e}")
                loop_ready.set()
        
        # Start thread
        thread = threading.Thread(target=loop_thread, name=f"AsyncLoop-{name}")
        thread.daemon = True
        thread.start()
        
        # Wait for loop to be ready
        loop_ready.wait(timeout=10.0)
        
        return loop_id
    
    def _configure_loop_sync(self, loop):
        """Synchronous loop configuration"""
        # This would be called within the loop's thread
        pass
    
    async def get_loop(self, loop_id: str = "main") -> Optional[asyncio.AbstractEventLoop]:
        """Get event loop by ID"""
        if loop_id in self.loops:
            return self.loops[loop_id].loop_obj
        return None
    
    def get_current_loop_info(self) -> Optional[LoopInfo]:
        """Get current loop information"""
        try:
            current_loop = asyncio.get_running_loop()
            for loop_info in self.loops.values():
                if loop_info.loop_obj is current_loop:
                    return loop_info
        except RuntimeError:
            pass
        return None
    
    async def shutdown_loop(self, loop_id: str) -> bool:
        """Shutdown specific event loop"""
        try:
            if loop_id not in self.loops:
                return False
            
            loop_info = self.loops[loop_id]
            loop = loop_info.loop_obj
            
            if loop and loop.is_running():
                # Cancel all tasks
                tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
                for task in tasks:
                    task.cancel()
                
                # Wait for tasks to finish
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # Stop loop
                loop.stop()
            
            del self.loops[loop_id]
            self.stats['active_loops'] -= 1
            
            self.app.logger.info(f"Event loop '{loop_id}' shutdown")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"AsyncIOManager.shutdown_loop({loop_id})")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get AsyncIO manager statistics"""
        loop_stats = {}
        for loop_id, loop_info in self.loops.items():
            loop_stats[loop_id] = {
                'thread_id': loop_info.thread_id,
                'thread_name': loop_info.thread_name,
                'is_running': loop_info.is_running,
                'is_main_loop': loop_info.is_main_loop,
                'task_count': loop_info.task_count,
                'total_tasks_run': loop_info.total_tasks_run
            }
        
        return {
            'general': self.stats.copy(),
            'loops': loop_stats
        }
    
    async def cleanup(self):
        """Cleanup AsyncIO manager"""
        try:
            self.running = False
            
            # Shutdown all non-main loops
            for loop_id in list(self.loops.keys()):
                if loop_id != "main":
                    await self.shutdown_loop(loop_id)
            
            self.app.logger.info("AsyncIO manager cleanup completed")
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "AsyncIOManager.cleanup")


class AsyncTaskManager:
    """Advanced async task management"""
    
    def __init__(self, app):
        self.app = app
        self.tasks = {}  # task_id -> AsyncTask
        self.task_groups = {}  # group_id -> AsyncTaskGroup
        self.task_dependencies = defaultdict(set)  # task_id -> Set[dependency_task_ids]
        self.task_dependents = defaultdict(set)  # task_id -> Set[dependent_task_ids]
        
        # Configuration
        self.max_concurrent_tasks = 1000
        self.default_timeout = 300.0  # 5 minutes
        self.cleanup_interval = 60.0  # 1 minute
        
        # Statistics
        self.stats = {
            'total_tasks': 0,
            'pending_tasks': 0,
            'running_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'cancelled_tasks': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0
        }
        
        # Monitoring
        self.running = False
        self.cleanup_task = None
        self.monitor_task = None
        
        self.lock = asyncio.Lock()
    
    async def setup(self) -> bool:
        """Setup async task manager"""
        try:
            self.running = True
            
            # Start background tasks
            self.cleanup_task = asyncio.create_task(self._cleanup_completed_tasks())
            self.monitor_task = asyncio.create_task(self._monitor_tasks())
            
            self.app.logger.info("Async task manager initialized successfully")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "AsyncTaskManager.setup")
            return False
    
    async def create_task(self, coro_or_func: Union[Coroutine, Callable], 
                         *args, name: str = "", priority: AsyncTaskPriority = AsyncTaskPriority.NORMAL,
                         timeout: Optional[float] = None, max_retries: int = 0,
                         dependencies: List[str] = None, **kwargs) -> str:
        """Create and schedule async task"""
        try:
            task = AsyncTask(
                name=name or (coro_or_func.__name__ if hasattr(coro_or_func, '__name__') else "unnamed"),
                priority=priority,
                timeout=timeout or self.default_timeout,
                max_retries=max_retries,
                dependencies=dependencies or []
            )
            
            # Handle coroutine or function
            if inspect.iscoroutine(coro_or_func):
                task.coroutine = coro_or_func
            elif callable(coro_or_func):
                task.function = coro_or_func
                task.args = args
                task.kwargs = kwargs
            else:
                raise ValueError("First argument must be coroutine or callable")
            
            # Store task
            async with self.lock:
                self.tasks[task.id] = task
                self.stats['total_tasks'] += 1
                self.stats['pending_tasks'] += 1
                
                # Handle dependencies
                for dep_id in task.dependencies:
                    self.task_dependencies[task.id].add(dep_id)
                    self.task_dependents[dep_id].add(task.id)
            
            # Check if dependencies are met and schedule if ready
            if await self._check_dependencies(task.id):
                await self._schedule_task(task.id)
            
            return task.id
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"AsyncTaskManager.create_task({name})")
            return ""
    
    async def _check_dependencies(self, task_id: str) -> bool:
        """Check if task dependencies are satisfied"""
        if task_id not in self.task_dependencies:
            return True
        
        dependencies = self.task_dependencies[task_id]
        for dep_id in dependencies:
            if dep_id in self.tasks:
                dep_task = self.tasks[dep_id]
                if dep_task.status != AsyncTaskStatus.COMPLETED:
                    return False
            else:
                # Dependency not found, assume it's satisfied
                continue
        
        return True
    
    async def _schedule_task(self, task_id: str) -> bool:
        """Schedule task for execution"""
        try:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            
            # Check if too many concurrent tasks
            if self.stats['running_tasks'] >= self.max_concurrent_tasks:
                # Could implement priority queue here
                return False
            
            # Create coroutine
            if task.coroutine:
                coro = task.coroutine
            elif task.function:
                if asyncio.iscoroutinefunction(task.function):
                    coro = task.function(*task.args, **task.kwargs)
                else:
                    # Wrap sync function
                    loop = asyncio.get_event_loop()
                    coro = loop.run_in_executor(None, task.function, *task.args, **task.kwargs)
            else:
                return False
            
            # Create asyncio task with timeout
            if task.timeout:
                coro = asyncio.wait_for(coro, timeout=task.timeout)
            
            asyncio_task = asyncio.create_task(coro, name=task.name)
            task.task_obj = asyncio_task
            task.status = AsyncTaskStatus.RUNNING
            task.started_at = datetime.now()
            
            # Update statistics
            async with self.lock:
                self.stats['pending_tasks'] -= 1
                self.stats['running_tasks'] += 1
            
            # Add completion callback
            asyncio_task.add_done_callback(
                lambda t: asyncio.create_task(self._task_completed(task_id))
            )
            
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"AsyncTaskManager._schedule_task({task_id})")
            return False
    
    async def _task_completed(self, task_id: str):
        """Handle task completion"""
        try:
            if task_id not in self.tasks:
                return
            
            task = self.tasks[task_id]
            task.completed_at = datetime.now()
            
            if not task.task_obj:
                return
            
            async with self.lock:
                self.stats['running_tasks'] -= 1
                
                try:
                    if task.task_obj.cancelled():
                        task.status = AsyncTaskStatus.CANCELLED
                        self.stats['cancelled_tasks'] += 1
                    else:
                        result = await task.task_obj
                        task.result = result
                        task.status = AsyncTaskStatus.COMPLETED
                        self.stats['completed_tasks'] += 1
                        
                        # Schedule dependent tasks
                        await self._schedule_dependents(task_id)
                
                except asyncio.TimeoutError:
                    task.status = AsyncTaskStatus.TIMEOUT
                    task.error = "Task timed out"
                    self.stats['failed_tasks'] += 1
                
                except Exception as e:
                    task.error = str(e)
                    task.status = AsyncTaskStatus.FAILED
                    self.stats['failed_tasks'] += 1
                    
                    # Retry logic
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        await asyncio.sleep(task.retry_delay * (2 ** task.retry_count))
                        await self._schedule_task(task_id)
                
                # Update execution time statistics
                if task.started_at and task.completed_at:
                    execution_time = (task.completed_at - task.started_at).total_seconds()
                    self.stats['total_execution_time'] += execution_time
                    
                    completed_count = (self.stats['completed_tasks'] + 
                                     self.stats['failed_tasks'] + 
                                     self.stats['cancelled_tasks'])
                    
                    if completed_count > 0:
                        self.stats['average_execution_time'] = (
                            self.stats['total_execution_time'] / completed_count
                        )
        
        except Exception as e:
            self.app.error_handler.handle_error(e, f"AsyncTaskManager._task_completed({task_id})")
    
    async def _schedule_dependents(self, completed_task_id: str):
        """Schedule tasks that depend on completed task"""
        if completed_task_id not in self.task_dependents:
            return
        
        dependents = self.task_dependents[completed_task_id].copy()
        for dependent_id in dependents:
            if await self._check_dependencies(dependent_id):
                await self._schedule_task(dependent_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel task"""
        try:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            
            if task.task_obj and not task.task_obj.done():
                task.task_obj.cancel()
                return True
            
            return False
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"AsyncTaskManager.cancel_task({task_id})")
            return False
    
    async def get_task_status(self, task_id: str) -> Optional[AsyncTaskStatus]:
        """Get task status"""
        if task_id in self.tasks:
            return self.tasks[task_id].status
        return None
    
    async def get_task_result(self, task_id: str) -> Any:
        """Get task result"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        if task.status == AsyncTaskStatus.COMPLETED:
            return task.result
        elif task.status == AsyncTaskStatus.FAILED:
            raise RuntimeError(f"Task failed: {task.error}")
        elif task.status == AsyncTaskStatus.CANCELLED:
            raise asyncio.CancelledError("Task was cancelled")
        
        return None
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for task completion and return result"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        
        if task.task_obj:
            try:
                if timeout:
                    return await asyncio.wait_for(task.task_obj, timeout=timeout)
                else:
                    return await task.task_obj
            except asyncio.TimeoutError:
                raise
            except Exception as e:
                raise RuntimeError(f"Task failed: {str(e)}")
        
        return task.result
    
    async def create_task_group(self, name: str, max_concurrent: int = 0) -> str:
        """Create task group"""
        group = AsyncTaskGroup(name, max_concurrent)
        self.task_groups[group.id] = group
        return group.id
    
    async def add_task_to_group(self, group_id: str, task_id: str) -> bool:
        """Add task to group"""
        if group_id not in self.task_groups or task_id not in self.tasks:
            return False
        
        group = self.task_groups[group_id]
        task = self.tasks[task_id]
        
        await group.add_task(task)
        return True
    
    async def _cleanup_completed_tasks(self):
        """Cleanup completed tasks periodically"""
        while self.running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                current_time = datetime.now()
                tasks_to_remove = []
                
                async with self.lock:
                    for task_id, task in self.tasks.items():
                        # Remove tasks completed more than 1 hour ago
                        if (task.status in [AsyncTaskStatus.COMPLETED, AsyncTaskStatus.FAILED, AsyncTaskStatus.CANCELLED] and
                            task.completed_at and 
                            (current_time - task.completed_at).total_seconds() > 3600):
                            tasks_to_remove.append(task_id)
                    
                    # Remove old tasks
                    for task_id in tasks_to_remove:
                        del self.tasks[task_id]
                        
                        # Clean up dependencies
                        if task_id in self.task_dependencies:
                            del self.task_dependencies[task_id]
                        if task_id in self.task_dependents:
                            del self.task_dependents[task_id]
                
                if tasks_to_remove:
                    self.app.logger.debug(f"Cleaned up {len(tasks_to_remove)} completed tasks")
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.app.error_handler.handle_error(e, "AsyncTaskManager._cleanup_completed_tasks")
    
    async def _monitor_tasks(self):
        """Monitor task performance"""
        while self.running:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Log statistics
                if self.app.logger:
                    await self.app.logger.debug_async(
                        "Async task manager statistics",
                        extra=self.get_statistics()
                    )
                
                # Check for stuck tasks
                current_time = datetime.now()
                stuck_tasks = []
                
                for task_id, task in self.tasks.items():
                    if (task.status == AsyncTaskStatus.RUNNING and 
                        task.started_at and
                        (current_time - task.started_at).total_seconds() > (task.timeout or self.default_timeout) * 2):
                        stuck_tasks.append(task_id)
                
                if stuck_tasks:
                    self.app.logger.warning(f"Potentially stuck tasks detected: {len(stuck_tasks)}")
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.app.error_handler.handle_error(e, "AsyncTaskManager._monitor_tasks")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get task manager statistics"""
        return {
            'general': self.stats.copy(),
            'tasks': {
                'total_in_memory': len(self.tasks),
                'task_groups': len(self.task_groups)
            },
            'dependencies': {
                'total_dependencies': len(self.task_dependencies),
                'total_dependents': len(self.task_dependents)
            }
        }
    
    async def shutdown_async(self):
        """Shutdown async task manager"""
        try:
            self.running = False
            
            # Cancel all running tasks
            running_tasks = [
                task.task_obj for task in self.tasks.values() 
                if task.task_obj and not task.task_obj.done()
            ]
            
            for task in running_tasks:
                task.cancel()
            
            # Wait for tasks to finish
            if running_tasks:
                await asyncio.gather(*running_tasks, return_exceptions=True)
            
            # Cancel background tasks
            if self.cleanup_task:
                self.cleanup_task.cancel()
            if self.monitor_task:
                self.monitor_task.cancel()
            
            self.app.logger.info("Async task manager shutdown completed")
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "AsyncTaskManager.shutdown_async")