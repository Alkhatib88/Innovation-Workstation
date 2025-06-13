#!/usr/bin/env python3
"""
Advanced Worker and Task Management System
Comprehensive worker pools, task scheduling, and distributed processing
"""

import asyncio
import time
import threading
import multiprocessing as mp
import uuid
import json
import queue
import heapq
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
from datetime import datetime, timedelta
import schedule
import croniter
import concurrent.futures
import weakref


class WorkerStatus(Enum):
    """Worker status"""
    IDLE = auto()
    BUSY = auto()
    OFFLINE = auto()
    ERROR = auto()
    SHUTDOWN = auto()


class TaskType(Enum):
    """Task types"""
    IMMEDIATE = auto()
    SCHEDULED = auto()
    RECURRING = auto()
    BATCH = auto()
    STREAM = auto()


class TaskPriority(Enum):
    """Task priorities"""
    LOWEST = 1
    LOW = 2
    NORMAL = 3
    HIGH = 4
    HIGHEST = 5
    CRITICAL = 6


class TaskState(Enum):
    """Task execution states"""
    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    RETRY = auto()
    TIMEOUT = auto()


@dataclass
class WorkerTask:
    """Worker task definition"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    task_type: TaskType = TaskType.IMMEDIATE
    priority: TaskPriority = TaskPriority.NORMAL
    function: Callable = None
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    retry_delay: float = 1.0
    schedule: Optional[str] = None  # Cron expression for recurring tasks
    depends_on: List[str] = field(default_factory=list)
    state: TaskState = TaskState.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    worker_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    progress: float = 0.0  # 0.0 to 1.0
    estimated_duration: Optional[float] = None


@dataclass
class Worker:
    """Worker definition"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    status: WorkerStatus = WorkerStatus.IDLE
    worker_type: str = "thread"  # thread, process, remote
    capabilities: Set[str] = field(default_factory=set)
    max_concurrent_tasks: int = 1
    current_tasks: Set[str] = field(default_factory=set)
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    total_execution_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: Optional[datetime] = None
    heartbeat: Optional[datetime] = None
    process_info: Optional[Dict[str, Any]] = None
    thread_info: Optional[Dict[str, Any]] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class BatchTask:
    """Batch processing task"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    items: List[Any] = field(default_factory=list)
    batch_size: int = 10
    processor_function: Callable = None
    progress_callback: Optional[Callable] = None
    parallel: bool = True
    max_workers: int = 4
    timeout_per_item: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    processed_items: int = 0
    failed_items: int = 0
    results: List[Any] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class TaskQueue:
    """Priority-based task queue with scheduling"""
    
    def __init__(self, maxsize: int = 0):
        self._heap = []
        self._counter = 0
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._task_map = {}  # task_id -> task
        self._scheduled_tasks = {}  # task_id -> scheduled_time
    
    def put(self, task: WorkerTask, block: bool = True, timeout: Optional[float] = None):
        """Put task in queue"""
        with self._not_full:
            if self._maxsize > 0:
                while len(self._heap) >= self._maxsize:
                    if not block:
                        raise queue.Full
                    if not self._not_full.wait(timeout):
                        raise queue.Full
            
            # Handle scheduled tasks
            if task.scheduled_at:
                self._scheduled_tasks[task.id] = task.scheduled_at
                task.state = TaskState.QUEUED
                self._task_map[task.id] = task
            else:
                # Add to priority queue immediately
                priority = -task.priority.value  # Negative for max-heap behavior
                self._counter += 1
                entry = (priority, self._counter, task)
                heapq.heappush(self._heap, entry)
                task.state = TaskState.QUEUED
                self._task_map[task.id] = task
            
            self._not_empty.notify()
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> WorkerTask:
        """Get task from queue"""
        with self._not_empty:
            while True:
                # Check for scheduled tasks that are ready
                current_time = datetime.now()
                ready_tasks = []
                
                for task_id, scheduled_time in list(self._scheduled_tasks.items()):
                    if current_time >= scheduled_time:
                        task = self._task_map[task_id]
                        ready_tasks.append(task)
                        del self._scheduled_tasks[task_id]
                
                # Add ready scheduled tasks to heap
                for task in ready_tasks:
                    priority = -task.priority.value
                    self._counter += 1
                    entry = (priority, self._counter, task)
                    heapq.heappush(self._heap, entry)
                
                # Get task from heap
                if self._heap:
                    _, _, task = heapq.heappop(self._heap)
                    self._not_full.notify()
                    return task
                
                if not block:
                    raise queue.Empty
                if not self._not_empty.wait(timeout):
                    raise queue.Empty
    
    def qsize(self) -> int:
        """Get queue size"""
        with self._lock:
            return len(self._heap) + len(self._scheduled_tasks)
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        with self._lock:
            return len(self._heap) == 0 and len(self._scheduled_tasks) == 0
    
    def get_task(self, task_id: str) -> Optional[WorkerTask]:
        """Get specific task by ID"""
        with self._lock:
            return self._task_map.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel task"""
        with self._lock:
            if task_id in self._task_map:
                task = self._task_map[task_id]
                task.state = TaskState.CANCELLED
                
                # Remove from scheduled tasks
                if task_id in self._scheduled_tasks:
                    del self._scheduled_tasks[task_id]
                
                return True
            return False


class WorkerManager:
    """Advanced worker management system"""
    
    def __init__(self, app):
        self.app = app
        self.workers = {}  # worker_id -> Worker
        self.worker_pools = {}  # pool_name -> List[worker_ids]
        self.task_queue = TaskQueue(maxsize=10000)
        self.running_tasks = {}  # task_id -> WorkerTask
        self.completed_tasks = deque(maxlen=1000)
        
        # Thread/Process pools
        self.thread_pool = None
        self.process_pool = None
        
        # Configuration
        self.default_thread_workers = min(32, (mp.cpu_count() or 1) + 4)
        self.default_process_workers = mp.cpu_count() or 1
        self.worker_timeout = 300.0  # 5 minutes
        self.heartbeat_interval = 30.0  # 30 seconds
        
        # Statistics
        self.stats = {
            'total_workers': 0,
            'active_workers': 0,
            'idle_workers': 0,
            'total_tasks_processed': 0,
            'tasks_per_second': 0.0,
            'average_task_duration': 0.0,
            'worker_utilization': 0.0
        }
        
        # Monitoring
        self.running = False
        self.dispatcher_task = None
        self.monitor_task = None
        self.heartbeat_task = None
        
        self.lock = asyncio.Lock()
    
    async def setup(self) -> bool:
        """Setup worker manager"""
        try:
            # Create thread pool
            self.thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.default_thread_workers,
                thread_name_prefix="workstation-worker"
            )
            
            # Create process pool
            self.process_pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.default_process_workers
            )
            
            # Create default worker pools
            await self.create_worker_pool("default", "thread", self.default_thread_workers)
            await self.create_worker_pool("cpu_intensive", "process", self.default_process_workers)
            await self.create_worker_pool("io_bound", "thread", min(16, self.default_thread_workers))
            
            self.running = True
            
            # Start background tasks
            self.dispatcher_task = asyncio.create_task(self._task_dispatcher())
            self.monitor_task = asyncio.create_task(self._monitor_workers())
            self.heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
            
            self.app.logger.info("Worker manager initialized successfully")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "WorkerManager.setup")
            return False
    
    async def create_worker_pool(self, pool_name: str, worker_type: str, count: int) -> bool:
        """Create pool of workers"""
        try:
            if pool_name in self.worker_pools:
                return False
            
            worker_ids = []
            
            for i in range(count):
                worker = Worker(
                    name=f"{pool_name}-worker-{i}",
                    worker_type=worker_type,
                    capabilities={pool_name, worker_type}
                )
                
                self.workers[worker.id] = worker
                worker_ids.append(worker.id)
                self.stats['total_workers'] += 1
                self.stats['idle_workers'] += 1
            
            self.worker_pools[pool_name] = worker_ids
            
            self.app.logger.info(f"Created worker pool '{pool_name}' with {count} {worker_type} workers")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"WorkerManager.create_worker_pool({pool_name})")
            return False
    
    async def submit_task(self, task: WorkerTask) -> str:
        """Submit task for execution"""
        try:
            # Validate task
            if not task.function:
                raise ValueError("Task function is required")
            
            # Set scheduling if needed
            if task.schedule and task.task_type == TaskType.RECURRING:
                # Calculate next run time
                try:
                    cron = croniter.croniter(task.schedule, datetime.now())
                    task.scheduled_at = cron.get_next(datetime)
                except Exception:
                    task.scheduled_at = datetime.now() + timedelta(hours=1)
            
            # Add to queue
            self.task_queue.put(task)
            
            self.app.logger.debug(f"Task '{task.name}' submitted with ID {task.id}")
            return task.id
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"WorkerManager.submit_task({task.name})")
            return ""
    
    async def _task_dispatcher(self):
        """Dispatch tasks to available workers"""
        while self.running:
            try:
                # Get next task (with timeout to allow checking for shutdown)
                try:
                    task = self.task_queue.get(block=False)
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue
                
                # Find available worker
                worker = await self._find_available_worker(task)
                
                if worker:
                    # Execute task
                    await self._execute_task(task, worker)
                else:
                    # No available worker, put task back in queue
                    self.task_queue.put(task)
                    await asyncio.sleep(0.5)  # Wait before retrying
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.app.error_handler.handle_error(e, "WorkerManager._task_dispatcher")
                await asyncio.sleep(1)
    
    async def _find_available_worker(self, task: WorkerTask) -> Optional[Worker]:
        """Find available worker for task"""
        async with self.lock:
            # Find workers that can handle the task
            suitable_workers = []
            
            for worker in self.workers.values():
                if (worker.status == WorkerStatus.IDLE and
                    len(worker.current_tasks) < worker.max_concurrent_tasks):
                    
                    # Check capabilities
                    if task.metadata.get('required_capabilities'):
                        required = set(task.metadata['required_capabilities'])
                        if not required.issubset(worker.capabilities):
                            continue
                    
                    suitable_workers.append(worker)
            
            # Return best worker (for now, just the first available)
            return suitable_workers[0] if suitable_workers else None
    
    async def _execute_task(self, task: WorkerTask, worker: Worker):
        """Execute task on worker"""
        try:
            # Update worker and task state
            worker.status = WorkerStatus.BUSY
            worker.current_tasks.add(task.id)
            worker.last_activity = datetime.now()
            
            task.state = TaskState.RUNNING
            task.started_at = datetime.now()
            task.worker_id = worker.id
            
            self.running_tasks[task.id] = task
            
            # Update statistics
            async with self.lock:
                self.stats['idle_workers'] -= 1
                self.stats['active_workers'] += 1
            
            # Execute based on worker type
            if worker.worker_type == "thread":
                future = self.thread_pool.submit(self._run_task_sync, task)
            elif worker.worker_type == "process":
                future = self.process_pool.submit(_execute_task_in_process, task.function, task.args, task.kwargs)
            else:
                # Direct execution (for testing or simple tasks)
                future = asyncio.get_event_loop().run_in_executor(
                    None, self._run_task_sync, task
                )
            
            # Wait for completion with timeout
            try:
                if task.timeout:
                    result = await asyncio.wait_for(
                        asyncio.wrap_future(future),
                        timeout=task.timeout
                    )
                else:
                    result = await asyncio.wrap_future(future)
                
                # Task completed successfully
                task.result = result
                task.state = TaskState.COMPLETED
                task.completed_at = datetime.now()
                
                worker.total_tasks_completed += 1
                
            except asyncio.TimeoutError:
                # Task timed out
                future.cancel()
                task.state = TaskState.TIMEOUT
                task.error = f"Task timed out after {task.timeout} seconds"
                task.completed_at = datetime.now()
                
                worker.total_tasks_failed += 1
                
            except Exception as e:
                # Task failed
                task.state = TaskState.FAILED
                task.error = str(e)
                task.completed_at = datetime.now()
                
                worker.total_tasks_failed += 1
                
                # Handle retries
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.state = TaskState.RETRY
                    task.scheduled_at = datetime.now() + timedelta(seconds=task.retry_delay)
                    self.task_queue.put(task)
        
        except Exception as e:
            task.state = TaskState.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            self.app.error_handler.handle_error(e, f"WorkerManager._execute_task({task.id})")
        
        finally:
            # Clean up worker state
            worker.status = WorkerStatus.IDLE
            worker.current_tasks.discard(task.id)
            worker.last_activity = datetime.now()
            
            # Update execution time
            if task.started_at and task.completed_at:
                execution_time = (task.completed_at - task.started_at).total_seconds()
                worker.total_execution_time += execution_time
            
            # Move to completed tasks
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]
            
            self.completed_tasks.append(task)
            
            # Update statistics
            async with self.lock:
                self.stats['active_workers'] -= 1
                self.stats['idle_workers'] += 1
                self.stats['total_tasks_processed'] += 1
            
            # Handle recurring tasks
            if (task.task_type == TaskType.RECURRING and 
                task.state == TaskState.COMPLETED and 
                task.schedule):
                
                # Schedule next occurrence
                try:
                    cron = croniter.croniter(task.schedule, datetime.now())
                    next_run = cron.get_next(datetime)
                    
                    # Create new task instance
                    next_task = WorkerTask(
                        name=task.name,
                        task_type=TaskType.RECURRING,
                        priority=task.priority,
                        function=task.function,
                        args=task.args,
                        kwargs=task.kwargs,
                        timeout=task.timeout,
                        max_retries=task.max_retries,
                        schedule=task.schedule,
                        scheduled_at=next_run,
                        metadata=task.metadata
                    )
                    
                    self.task_queue.put(next_task)
                    
                except Exception as e:
                    self.app.logger.warning(f"Failed to schedule recurring task: {e}")
    
    def _run_task_sync(self, task: WorkerTask) -> Any:
        """Run task synchronously"""
        try:
            return task.function(*task.args, **task.kwargs)
        except Exception as e:
            raise RuntimeError(f"Task execution failed: {str(e)}")
    
    async def _monitor_workers(self):
        """Monitor worker health and performance"""
        while self.running:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                current_time = datetime.now()
                
                # Check for stuck workers
                for worker in self.workers.values():
                    if (worker.status == WorkerStatus.BUSY and 
                        worker.last_activity and
                        (current_time - worker.last_activity).total_seconds() > self.worker_timeout):
                        
                        self.app.logger.warning(f"Worker {worker.id} appears stuck")
                        worker.status = WorkerStatus.ERROR
                
                # Update utilization statistics
                total_workers = len(self.workers)
                if total_workers > 0:
                    self.stats['worker_utilization'] = self.stats['active_workers'] / total_workers
                
                # Calculate tasks per second
                if self.stats['total_tasks_processed'] > 0:
                    uptime = (current_time - self.app.startup_time).total_seconds() if self.app.startup_time else 1
                    self.stats['tasks_per_second'] = self.stats['total_tasks_processed'] / uptime
                
                # Log statistics
                if self.app.logger:
                    await self.app.logger.debug_async(
                        "Worker manager statistics",
                        extra=self.get_statistics()
                    )
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.app.error_handler.handle_error(e, "WorkerManager._monitor_workers")
    
    async def _heartbeat_monitor(self):
        """Monitor worker heartbeats"""
        while self.running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                current_time = datetime.now()
                
                # Update heartbeats
                for worker in self.workers.values():
                    worker.heartbeat = current_time
                
                # Check for dead workers (in a real system, this would be more sophisticated)
                for worker in self.workers.values():
                    if worker.status != WorkerStatus.OFFLINE:
                        # Simple heartbeat check - in practice, workers would actively send heartbeats
                        worker.status = WorkerStatus.IDLE if worker.status != WorkerStatus.BUSY else worker.status
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.app.error_handler.handle_error(e, "WorkerManager._heartbeat_monitor")
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        # Check running tasks
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            return {
                'id': task.id,
                'name': task.name,
                'state': task.state.name,
                'progress': task.progress,
                'worker_id': task.worker_id,
                'started_at': task.started_at.isoformat() if task.started_at else None
            }
        
        # Check queued tasks
        task = self.task_queue.get_task(task_id)
        if task:
            return {
                'id': task.id,
                'name': task.name,
                'state': task.state.name,
                'scheduled_at': task.scheduled_at.isoformat() if task.scheduled_at else None
            }
        
        # Check completed tasks
        for task in self.completed_tasks:
            if task.id == task_id:
                return {
                    'id': task.id,
                    'name': task.name,
                    'state': task.state.name,
                    'result': task.result,
                    'error': task.error,
                    'completed_at': task.completed_at.isoformat() if task.completed_at else None
                }
        
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel task"""
        # Try to cancel from queue
        if self.task_queue.cancel_task(task_id):
            return True
        
        # Try to cancel running task
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.state = TaskState.CANCELLED
            # Note: Actual cancellation of running tasks is complex and depends on implementation
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get worker manager statistics"""
        queue_stats = {
            'queue_size': self.task_queue.qsize(),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks)
        }
        
        worker_stats = {}
        for pool_name, worker_ids in self.worker_pools.items():
            pool_workers = [self.workers[wid] for wid in worker_ids if wid in self.workers]
            worker_stats[pool_name] = {
                'total_workers': len(pool_workers),
                'busy_workers': sum(1 for w in pool_workers if w.status == WorkerStatus.BUSY),
                'idle_workers': sum(1 for w in pool_workers if w.status == WorkerStatus.IDLE),
                'total_tasks_completed': sum(w.total_tasks_completed for w in pool_workers),
                'total_tasks_failed': sum(w.total_tasks_failed for w in pool_workers)
            }
        
        return {
            'general': self.stats,
            'queue': queue_stats,
            'pools': worker_stats
        }
    
    async def shutdown_async(self):
        """Shutdown worker manager"""
        try:
            self.running = False
            
            # Cancel background tasks
            if self.dispatcher_task:
                self.dispatcher_task.cancel()
            if self.monitor_task:
                self.monitor_task.cancel()
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
            
            # Wait for tasks to complete
            background_tasks = [t for t in [self.dispatcher_task, self.monitor_task, self.heartbeat_task] if t]
            if background_tasks:
                await asyncio.gather(*background_tasks, return_exceptions=True)
            
            # Shutdown executor pools
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
            if self.process_pool:
                self.process_pool.shutdown(wait=True)
            
            self.app.logger.info("Worker manager shutdown completed")
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "WorkerManager.shutdown_async")


class TaskManager:
    """High-level task management interface"""
    
    def __init__(self, app):
        self.app = app
        self.worker_manager = None
        self.task_templates = {}
        self.task_chains = {}
        self.batch_processor = None
    
    async def setup(self) -> bool:
        """Setup task manager"""
        try:
            self.worker_manager = WorkerManager(self.app)
            await self.worker_manager.setup()
            
            self.batch_processor = BatchProcessor(self.app)
            
            self.app.logger.info("Task manager initialized successfully")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "TaskManager.setup")
            return False
    
    async def run_task(self, function: Callable, *args, **kwargs) -> str:
        """Run immediate task"""
        task = WorkerTask(
            name=function.__name__,
            task_type=TaskType.IMMEDIATE,
            function=function,
            args=args,
            kwargs=kwargs
        )
        
        return await self.worker_manager.submit_task(task)
    
    async def schedule_task(self, function: Callable, schedule: str, 
                           name: str = "", *args, **kwargs) -> str:
        """Schedule recurring task"""
        task = WorkerTask(
            name=name or function.__name__,
            task_type=TaskType.RECURRING,
            function=function,
            args=args,
            kwargs=kwargs,
            schedule=schedule
        )
        
        return await self.worker_manager.submit_task(task)
    
    async def run_batch(self, items: List[Any], processor: Callable, 
                       batch_size: int = 10, parallel: bool = True) -> str:
        """Run batch processing task"""
        batch_task = BatchTask(
            name=f"batch_{processor.__name__}",
            items=items,
            batch_size=batch_size,
            processor_function=processor,
            parallel=parallel
        )
        
        return await self.batch_processor.process_batch(batch_task)
    
    async def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result"""
        start_time = time.time()
        
        while True:
            status = await self.worker_manager.get_task_status(task_id)
            
            if not status:
                raise ValueError(f"Task {task_id} not found")
            
            if status['state'] == 'COMPLETED':
                return status.get('result')
            elif status['state'] in ['FAILED', 'CANCELLED', 'TIMEOUT']:
                error = status.get('error', 'Task failed')
                raise RuntimeError(f"Task failed: {error}")
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
            
            await asyncio.sleep(0.5)


class TaskScheduler:
    """Advanced task scheduling system"""
    
    def __init__(self, app):
        self.app = app
        self.scheduled_tasks = {}
        self.cron_jobs = {}
        self.scheduler = schedule
        self.running = False
        self.scheduler_task = None
    
    async def setup(self) -> bool:
        """Setup task scheduler"""
        try:
            self.running = True
            self.scheduler_task = asyncio.create_task(self._scheduler_loop())
            
            self.app.logger.info("Task scheduler initialized successfully")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "TaskScheduler.setup")
            return False
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                # Run pending scheduled tasks
                self.scheduler.run_pending()
                
                # Check cron jobs
                await self._check_cron_jobs()
                
                await asyncio.sleep(1)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.app.error_handler.handle_error(e, "TaskScheduler._scheduler_loop")
                await asyncio.sleep(5)
    
    async def _check_cron_jobs(self):
        """Check and execute cron jobs"""
        current_time = datetime.now()
        
        for job_id, job_info in list(self.cron_jobs.items()):
            try:
                cron = croniter.croniter(job_info['schedule'], job_info['last_run'])
                next_run = cron.get_next(datetime)
                
                if current_time >= next_run:
                    # Execute job
                    task = WorkerTask(
                        name=job_info['name'],
                        task_type=TaskType.SCHEDULED,
                        function=job_info['function'],
                        args=job_info.get('args', ()),
                        kwargs=job_info.get('kwargs', {})
                    )
                    
                    if hasattr(self.app, 'worker_manager'):
                        await self.app.worker_manager.submit_task(task)
                    
                    job_info['last_run'] = current_time
                    job_info['run_count'] += 1
            
            except Exception as e:
                self.app.logger.error(f"Cron job {job_id} failed: {e}")
    
    def add_cron_job(self, name: str, schedule: str, function: Callable, 
                     *args, **kwargs) -> str:
        """Add cron job"""
        job_id = str(uuid.uuid4())
        
        self.cron_jobs[job_id] = {
            'name': name,
            'schedule': schedule,
            'function': function,
            'args': args,
            'kwargs': kwargs,
            'last_run': datetime.now(),
            'run_count': 0
        }
        
        return job_id
    
    async def shutdown_async(self):
        """Shutdown task scheduler"""
        try:
            self.running = False
            
            if self.scheduler_task:
                self.scheduler_task.cancel()
                try:
                    await self.scheduler_task
                except asyncio.CancelledError:
                    pass
            
            self.app.logger.info("Task scheduler shutdown completed")
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "TaskScheduler.shutdown_async")


class BatchProcessor:
    """Batch processing system"""
    
    def __init__(self, app):
        self.app = app
        self.active_batches = {}
    
    async def process_batch(self, batch_task: BatchTask) -> str:
        """Process batch task"""
        try:
            self.active_batches[batch_task.id] = batch_task
            
            if batch_task.parallel:
                await self._process_parallel(batch_task)
            else:
                await self._process_sequential(batch_task)
            
            return batch_task.id
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"BatchProcessor.process_batch({batch_task.id})")
            return ""
    
    async def _process_parallel(self, batch_task: BatchTask):
        """Process batch in parallel"""
        semaphore = asyncio.Semaphore(batch_task.max_workers)
        
        async def process_item(item):
            async with semaphore:
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, batch_task.processor_function, item
                    )
                    batch_task.results.append(result)
                    batch_task.processed_items += 1
                    
                    if batch_task.progress_callback:
                        progress = batch_task.processed_items / len(batch_task.items)
                        batch_task.progress_callback(progress)
                
                except Exception as e:
                    batch_task.errors.append(str(e))
                    batch_task.failed_items += 1
        
        # Process all items
        tasks = [process_item(item) for item in batch_task.items]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_sequential(self, batch_task: BatchTask):
        """Process batch sequentially"""
        for i, item in enumerate(batch_task.items):
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, batch_task.processor_function, item
                )
                batch_task.results.append(result)
                batch_task.processed_items += 1
                
                if batch_task.progress_callback:
                    progress = (i + 1) / len(batch_task.items)
                    batch_task.progress_callback(progress)
            
            except Exception as e:
                batch_task.errors.append(str(e))
                batch_task.failed_items += 1


# Global function for process execution (must be at module level for pickling)
def _execute_task_in_process(func, args, kwargs):
    """Execute task in separate process"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Process task execution failed: {str(e)}")