#!/usr/bin/env python3
"""
Advanced Thread and Process Management System
Comprehensive thread pools, process pools, and concurrent execution management
"""

import asyncio
import threading
import multiprocessing as mp
import time
import uuid
import json
import queue
import signal
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from datetime import datetime, timedelta
import psutil
import weakref


class TaskPriority(Enum):
    """Task priority levels"""
    LOWEST = 1
    LOW = 2
    NORMAL = 3
    HIGH = 4
    HIGHEST = 5
    CRITICAL = 6


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


class WorkerType(Enum):
    """Worker types"""
    THREAD = auto()
    PROCESS = auto()
    ASYNC_TASK = auto()


@dataclass
class Task:
    """Task data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    function: Callable = None
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 0
    retry_delay: float = 1.0
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    worker_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerInfo:
    """Worker information"""
    id: str
    worker_type: WorkerType
    name: str = ""
    is_active: bool = False
    is_busy: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: Optional[datetime] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    current_task: Optional[str] = None
    process_info: Optional[Dict[str, Any]] = None
    thread_info: Optional[Dict[str, Any]] = None


class ThreadSafeCounter:
    """Thread-safe counter"""
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def increment(self, amount: int = 1) -> int:
        with self._lock:
            self._value += amount
            return self._value
    
    def decrement(self, amount: int = 1) -> int:
        with self._lock:
            self._value -= amount
            return self._value
    
    def get(self) -> int:
        with self._lock:
            return self._value
    
    def set(self, value: int):
        with self._lock:
            self._value = value


class TaskQueue:
    """Priority-based task queue"""
    
    def __init__(self, maxsize: int = 0):
        self._queue = queue.PriorityQueue(maxsize=maxsize)
        self._task_counter = ThreadSafeCounter()
    
    def put(self, task: Task, timeout: Optional[float] = None):
        """Put task in queue"""
        # Use negative priority for max-heap behavior and counter for FIFO within priority
        priority_tuple = (-task.priority.value, self._task_counter.increment(), task)
        self._queue.put(priority_tuple, timeout=timeout)
    
    def get(self, timeout: Optional[float] = None) -> Task:
        """Get task from queue"""
        _, _, task = self._queue.get(timeout=timeout)
        return task
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        return self._queue.empty()
    
    def qsize(self) -> int:
        """Get queue size"""
        return self._queue.qsize()
    
    def task_done(self):
        """Mark task as done"""
        self._queue.task_done()


class ThreadManager:
    """Advanced thread management system"""
    
    def __init__(self, app):
        self.app = app
        self.thread_pools = {}  # pool_name -> ThreadPoolExecutor
        self.workers = {}  # worker_id -> WorkerInfo
        self.tasks = {}  # task_id -> Task
        self.task_queue = TaskQueue()
        self.futures = {}  # task_id -> Future
        
        # Configuration
        self.default_pool_size = min(32, (os.cpu_count() or 1) + 4)
        self.max_pool_size = 100
        self.worker_timeout = 300.0  # 5 minutes
        self.task_timeout = 60.0  # 1 minute default
        
        # Statistics
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'active_tasks': 0,
            'active_threads': 0,
            'total_execution_time': 0.0,
            'average_execution_time': 0.0
        }
        
        # Monitoring
        self.monitoring_enabled = True
        self.monitor_task = None
        self.running = False
        
        self.lock = threading.Lock()
    
    async def setup(self) -> bool:
        """Setup thread manager"""
        try:
            # Create default thread pool
            await self.create_pool("default", self.default_pool_size)
            await self.create_pool("high_priority", min(8, self.default_pool_size // 2))
            await self.create_pool("background", min(4, self.default_pool_size // 4))
            
            self.running = True
            
            # Start monitoring
            if self.monitoring_enabled:
                self.monitor_task = asyncio.create_task(self._monitor_threads())
            
            self.app.logger.info("Thread manager initialized successfully")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "ThreadManager.setup")
            return False
    
    async def create_pool(self, name: str, max_workers: int) -> bool:
        """Create thread pool"""
        try:
            if name in self.thread_pools:
                return False
            
            pool = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix=f"workstation-{name}"
            )
            
            self.thread_pools[name] = pool
            
            # Register workers
            for i in range(max_workers):
                worker_id = f"{name}-worker-{i}"
                worker_info = WorkerInfo(
                    id=worker_id,
                    worker_type=WorkerType.THREAD,
                    name=f"{name} Thread Worker {i}",
                    is_active=True
                )
                self.workers[worker_id] = worker_info
            
            self.app.logger.info(f"Thread pool '{name}' created with {max_workers} workers")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"ThreadManager.create_pool({name})")
            return False
    
    async def submit_task(self, task: Task, pool_name: str = "default") -> Optional[str]:
        """Submit task for execution"""
        try:
            if pool_name not in self.thread_pools:
                return None
            
            pool = self.thread_pools[pool_name]
            
            # Wrap task execution
            def task_wrapper():
                return self._execute_task(task)
            
            # Submit to thread pool
            future = pool.submit(task_wrapper)
            
            # Store task and future
            with self.lock:
                task.status = TaskStatus.PENDING
                self.tasks[task.id] = task
                self.futures[task.id] = future
                self.stats['total_tasks'] += 1
                self.stats['active_tasks'] += 1
            
            # Add completion callback
            future.add_done_callback(lambda f: self._task_completed(task.id, f))
            
            return task.id
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"ThreadManager.submit_task({task.name})")
            return None
    
    def _execute_task(self, task: Task) -> Any:
        """Execute task in thread"""
        start_time = time.time()
        
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            # Get current thread info
            current_thread = threading.current_thread()
            thread_id = current_thread.ident
            
            # Update worker info
            for worker in self.workers.values():
                if worker.worker_type == WorkerType.THREAD and f"workstation-" in current_thread.name:
                    worker.is_busy = True
                    worker.current_task = task.id
                    worker.last_activity = datetime.now()
                    task.worker_id = worker.id
                    break
            
            # Execute with timeout
            if task.timeout:
                # Note: Threading timeout is complex, this is a simplified approach
                result = task.function(*task.args, **task.kwargs)
            else:
                result = task.function(*task.args, **task.kwargs)
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            return result
            
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                time.sleep(task.retry_delay * (2 ** task.retry_count))
                return self._execute_task(task)
            
            raise
        
        finally:
            task.completed_at = datetime.now()
            execution_time = time.time() - start_time
            
            # Update worker info
            if task.worker_id and task.worker_id in self.workers:
                worker = self.workers[task.worker_id]
                worker.is_busy = False
                worker.current_task = None
                worker.last_activity = datetime.now()
                worker.total_execution_time += execution_time
                
                if task.status == TaskStatus.COMPLETED:
                    worker.tasks_completed += 1
                else:
                    worker.tasks_failed += 1
    
    def _task_completed(self, task_id: str, future: Future):
        """Handle task completion"""
        try:
            with self.lock:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    
                    try:
                        if not future.cancelled():
                            result = future.result()
                            if task.status != TaskStatus.FAILED:
                                task.result = result
                                task.status = TaskStatus.COMPLETED
                                self.stats['completed_tasks'] += 1
                    except Exception as e:
                        task.error = str(e)
                        task.status = TaskStatus.FAILED
                        self.stats['failed_tasks'] += 1
                    
                    self.stats['active_tasks'] -= 1
                    
                    # Update average execution time
                    if task.started_at and task.completed_at:
                        execution_time = (task.completed_at - task.started_at).total_seconds()
                        self.stats['total_execution_time'] += execution_time
                        completed_count = self.stats['completed_tasks'] + self.stats['failed_tasks']
                        if completed_count > 0:
                            self.stats['average_execution_time'] = (
                                self.stats['total_execution_time'] / completed_count
                            )
                
                # Clean up future
                if task_id in self.futures:
                    del self.futures[task_id]
        
        except Exception as e:
            self.app.error_handler.handle_error(e, f"ThreadManager._task_completed({task_id})")
    
    async def _monitor_threads(self):
        """Monitor thread health and performance"""
        while self.running:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Update thread statistics
                active_threads = threading.active_count()
                self.stats['active_threads'] = active_threads
                
                # Check for stuck workers
                current_time = datetime.now()
                for worker in self.workers.values():
                    if (worker.is_busy and worker.last_activity and 
                        (current_time - worker.last_activity).total_seconds() > self.worker_timeout):
                        
                        self.app.logger.warning(
                            f"Worker {worker.id} may be stuck (last activity: {worker.last_activity})"
                        )
                
                # Log statistics
                if self.app.logger:
                    await self.app.logger.debug_async(
                        "Thread manager statistics",
                        extra=self.get_statistics()
                    )
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.app.error_handler.handle_error(e, "ThreadManager._monitor_threads")
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get task status"""
        with self.lock:
            if task_id in self.tasks:
                return self.tasks[task_id].status
        return None
    
    async def get_task_result(self, task_id: str) -> Any:
        """Get task result"""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.status == TaskStatus.COMPLETED:
                    return task.result
                elif task.status == TaskStatus.FAILED:
                    raise RuntimeError(f"Task failed: {task.error}")
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel task"""
        try:
            with self.lock:
                if task_id in self.futures:
                    future = self.futures[task_id]
                    cancelled = future.cancel()
                    
                    if cancelled and task_id in self.tasks:
                        self.tasks[task_id].status = TaskStatus.CANCELLED
                        self.stats['active_tasks'] -= 1
                    
                    return cancelled
            return False
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"ThreadManager.cancel_task({task_id})")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get thread manager statistics"""
        with self.lock:
            pool_stats = {}
            for name, pool in self.thread_pools.items():
                pool_stats[name] = {
                    'max_workers': pool._max_workers,
                    'threads': len(pool._threads),
                    'queue_size': pool._work_queue.qsize()
                }
            
            worker_stats = {
                'total_workers': len(self.workers),
                'active_workers': sum(1 for w in self.workers.values() if w.is_active),
                'busy_workers': sum(1 for w in self.workers.values() if w.is_busy)
            }
            
            return {
                'general': self.stats.copy(),
                'pools': pool_stats,
                'workers': worker_stats
            }
    
    async def shutdown_async(self):
        """Shutdown thread manager"""
        try:
            self.running = False
            
            # Stop monitoring
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown thread pools
            for name, pool in self.thread_pools.items():
                pool.shutdown(wait=True)
                self.app.logger.info(f"Thread pool '{name}' shutdown")
            
            self.thread_pools.clear()
            self.workers.clear()
            
            self.app.logger.info("Thread manager shutdown completed")
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "ThreadManager.shutdown_async")


class ProcessManager:
    """Advanced process management system"""
    
    def __init__(self, app):
        self.app = app
        self.process_pools = {}  # pool_name -> ProcessPoolExecutor
        self.processes = {}  # process_id -> process_info
        self.tasks = {}  # task_id -> Task
        self.futures = {}  # task_id -> Future
        
        # Configuration
        self.default_pool_size = min(mp.cpu_count(), 8)
        self.max_pool_size = mp.cpu_count() * 2
        self.process_timeout = 600.0  # 10 minutes
        
        # Statistics
        self.stats = {
            'total_processes': 0,
            'active_processes': 0,
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_execution_time': 0.0
        }
        
        # Monitoring
        self.monitoring_enabled = True
        self.monitor_task = None
        self.running = False
        
        self.lock = threading.Lock()
    
    async def setup(self) -> bool:
        """Setup process manager"""
        try:
            # Set multiprocessing start method
            if hasattr(mp, 'set_start_method'):
                try:
                    mp.set_start_method('spawn', force=True)
                except RuntimeError:
                    pass  # Start method already set
            
            # Create default process pool
            await self.create_pool("default", self.default_pool_size)
            await self.create_pool("cpu_intensive", min(mp.cpu_count(), 4))
            
            self.running = True
            
            # Start monitoring
            if self.monitoring_enabled:
                self.monitor_task = asyncio.create_task(self._monitor_processes())
            
            self.app.logger.info("Process manager initialized successfully")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "ProcessManager.setup")
            return False
    
    async def create_pool(self, name: str, max_workers: int) -> bool:
        """Create process pool"""
        try:
            if name in self.process_pools:
                return False
            
            # Create process pool
            pool = ProcessPoolExecutor(max_workers=max_workers)
            self.process_pools[name] = pool
            
            self.stats['total_processes'] += max_workers
            
            self.app.logger.info(f"Process pool '{name}' created with {max_workers} workers")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"ProcessManager.create_pool({name})")
            return False
    
    async def submit_task(self, task: Task, pool_name: str = "default") -> Optional[str]:
        """Submit task for execution"""
        try:
            if pool_name not in self.process_pools:
                return None
            
            pool = self.process_pools[pool_name]
            
            # Submit to process pool
            future = pool.submit(task.function, *task.args, **task.kwargs)
            
            # Store task and future
            with self.lock:
                task.status = TaskStatus.PENDING
                self.tasks[task.id] = task
                self.futures[task.id] = future
                self.stats['total_tasks'] += 1
            
            # Add completion callback
            future.add_done_callback(lambda f: self._task_completed(task.id, f))
            
            return task.id
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"ProcessManager.submit_task({task.name})")
            return None
    
    def _task_completed(self, task_id: str, future: Future):
        """Handle task completion"""
        try:
            with self.lock:
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    
                    try:
                        if not future.cancelled():
                            result = future.result()
                            task.result = result
                            task.status = TaskStatus.COMPLETED
                            self.stats['completed_tasks'] += 1
                    except Exception as e:
                        task.error = str(e)
                        task.status = TaskStatus.FAILED
                        self.stats['failed_tasks'] += 1
                    
                    task.completed_at = datetime.now()
                
                # Clean up future
                if task_id in self.futures:
                    del self.futures[task_id]
        
        except Exception as e:
            self.app.error_handler.handle_error(e, f"ProcessManager._task_completed({task_id})")
    
    async def _monitor_processes(self):
        """Monitor process health and performance"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Update process statistics
                active_processes = 0
                for pool in self.process_pools.values():
                    if hasattr(pool, '_processes'):
                        active_processes += len(pool._processes)
                
                self.stats['active_processes'] = active_processes
                
                # Log statistics
                if self.app.logger:
                    await self.app.logger.debug_async(
                        "Process manager statistics",
                        extra=self.get_statistics()
                    )
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.app.error_handler.handle_error(e, "ProcessManager._monitor_processes")
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel task"""
        try:
            with self.lock:
                if task_id in self.futures:
                    future = self.futures[task_id]
                    cancelled = future.cancel()
                    
                    if cancelled and task_id in self.tasks:
                        self.tasks[task_id].status = TaskStatus.CANCELLED
                    
                    return cancelled
            return False
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"ProcessManager.cancel_task({task_id})")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get process manager statistics"""
        with self.lock:
            pool_stats = {}
            for name, pool in self.process_pools.items():
                pool_stats[name] = {
                    'max_workers': pool._max_workers,
                    'processes': len(getattr(pool, '_processes', {}))
                }
            
            return {
                'general': self.stats.copy(),
                'pools': pool_stats
            }
    
    async def shutdown_async(self):
        """Shutdown process manager"""
        try:
            self.running = False
            
            # Stop monitoring
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown process pools
            for name, pool in self.process_pools.items():
                pool.shutdown(wait=True)
                self.app.logger.info(f"Process pool '{name}' shutdown")
            
            self.process_pools.clear()
            self.processes.clear()
            
            self.app.logger.info("Process manager shutdown completed")
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "ProcessManager.shutdown_async")


# Worker function for process execution (must be at module level for pickling)
def execute_task_in_process(task_func, *args, **kwargs):
    """Execute task in separate process"""
    try:
        return task_func(*args, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Process task execution failed: {str(e)}")