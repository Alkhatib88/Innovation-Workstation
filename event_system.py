#!/usr/bin/env python3
"""
Advanced Event System
Comprehensive event handling, automation, scheduling, and pub/sub messaging
"""

import asyncio
import time
import threading
import uuid
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import heapq
import weakref
from datetime import datetime, timedelta
import re
import croniter
import inspect


class EventPriority(Enum):
    """Event priority levels"""
    LOWEST = 1
    LOW = 2
    NORMAL = 3
    HIGH = 4
    HIGHEST = 5
    CRITICAL = 6


class EventStatus(Enum):
    """Event processing status"""
    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


class EventType(Enum):
    """Event types"""
    SYSTEM = auto()
    USER = auto()
    APPLICATION = auto()
    NETWORK = auto()
    DATABASE = auto()
    SECURITY = auto()
    AUTOMATION = auto()
    WEBHOOK = auto()
    SCHEDULED = auto()
    CUSTOM = auto()


@dataclass
class Event:
    """Event data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    event_type: EventType = EventType.CUSTOM
    priority: EventPriority = EventPriority.NORMAL
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    target: Optional[str] = None
    correlation_id: Optional[str] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    ttl: Optional[float] = None  # Time to live in seconds
    retry_count: int = 0
    max_retries: int = 3
    retry_delay: float = 1.0
    status: EventStatus = EventStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Any = None


@dataclass
class EventHandler:
    """Event handler definition"""
    id: str
    name: str
    handler_func: Callable
    event_pattern: str
    priority: int = 0
    async_handler: bool = False
    conditions: List[Callable] = field(default_factory=list)
    filters: List[Callable] = field(default_factory=list)
    middleware: List[Callable] = field(default_factory=list)
    timeout: Optional[float] = None
    max_concurrent: int = 0  # 0 = unlimited
    enabled: bool = True
    statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduledEvent:
    """Scheduled event definition"""
    id: str
    name: str
    event_data: Dict[str, Any]
    schedule: str  # Cron expression or interval
    next_run: datetime
    enabled: bool = True
    max_runs: Optional[int] = None
    run_count: int = 0
    last_run: Optional[datetime] = None
    timeout: Optional[float] = None
    retry_on_failure: bool = True
    jitter: float = 0.0  # Random delay percentage


class EventPattern:
    """Event pattern matching"""
    
    def __init__(self, pattern: str):
        self.pattern = pattern
        self.compiled_pattern = self._compile_pattern(pattern)
    
    def _compile_pattern(self, pattern: str) -> re.Pattern:
        """Compile pattern to regex"""
        # Convert glob-like pattern to regex
        # * matches any characters except dots
        # ** matches any characters including dots
        # ? matches single character
        
        pattern = pattern.replace('.', r'\.')
        pattern = pattern.replace('**', '__DOUBLE_STAR__')
        pattern = pattern.replace('*', '[^.]*')
        pattern = pattern.replace('__DOUBLE_STAR__', '.*')
        pattern = pattern.replace('?', '.')
        
        return re.compile(f'^{pattern}$')
    
    def matches(self, event_name: str) -> bool:
        """Check if pattern matches event name"""
        return bool(self.compiled_pattern.match(event_name))


class EventBus:
    """Advanced event bus with pub/sub messaging"""
    
    def __init__(self, max_queue_size: int = 10000):
        self.handlers = defaultdict(list)  # event_name -> [EventHandler]
        self.global_handlers = []  # Handlers for all events
        self.event_queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing_tasks = set()
        self.handler_registry = {}  # handler_id -> EventHandler
        self.event_history = deque(maxlen=1000)
        self.metrics = {
            'events_published': 0,
            'events_processed': 0,
            'events_failed': 0,
            'handlers_executed': 0,
            'total_processing_time': 0.0
        }
        
        # Configuration
        self.max_concurrent_handlers = 100
        self.default_timeout = 30.0
        self.auto_retry = True
        self.dead_letter_queue = deque(maxlen=100)
        
        # Worker management
        self.workers = []
        self.running = False
        self.lock = asyncio.Lock()
    
    async def start(self, num_workers: int = 4):
        """Start event bus workers"""
        if self.running:
            return
        
        self.running = True
        
        # Start worker tasks
        for i in range(num_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
            self.processing_tasks.add(worker)
            worker.add_done_callback(self.processing_tasks.discard)
    
    async def stop(self):
        """Stop event bus workers"""
        self.running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        self.workers.clear()
        self.processing_tasks.clear()
    
    async def _worker(self, worker_id: str):
        """Event processing worker"""
        while self.running:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(
                    self.event_queue.get(), 
                    timeout=1.0
                )
                
                await self._process_event(event, worker_id)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
    
    async def _process_event(self, event: Event, worker_id: str):
        """Process single event"""
        start_time = time.time()
        event.status = EventStatus.PROCESSING
        event.processed_at = datetime.now()
        
        try:
            # Check TTL
            if event.ttl and (time.time() - event.timestamp) > event.ttl:
                event.status = EventStatus.TIMEOUT
                return
            
            # Find matching handlers
            handlers = self._find_handlers(event)
            
            if not handlers:
                event.status = EventStatus.COMPLETED
                return
            
            # Execute handlers
            handler_tasks = []
            for handler in handlers:
                if not handler.enabled:
                    continue
                
                # Check conditions
                if not self._check_conditions(event, handler):
                    continue
                
                # Apply filters
                if not self._apply_filters(event, handler):
                    continue
                
                # Create handler task
                task = asyncio.create_task(
                    self._execute_handler(event, handler, worker_id)
                )
                handler_tasks.append(task)
                
                # Limit concurrent handlers
                if len(handler_tasks) >= self.max_concurrent_handlers:
                    break
            
            # Wait for all handlers to complete
            if handler_tasks:
                results = await asyncio.gather(*handler_tasks, return_exceptions=True)
                
                # Check results
                failed_count = sum(1 for r in results if isinstance(r, Exception))
                if failed_count > 0:
                    event.status = EventStatus.FAILED
                    event.error = f"{failed_count}/{len(results)} handlers failed"
                else:
                    event.status = EventStatus.COMPLETED
            else:
                event.status = EventStatus.COMPLETED
            
            self.metrics['events_processed'] += 1
            
        except Exception as e:
            event.status = EventStatus.FAILED
            event.error = str(e)
            self.metrics['events_failed'] += 1
            
            # Retry logic
            if self.auto_retry and event.retry_count < event.max_retries:
                event.retry_count += 1
                event.status = EventStatus.PENDING
                
                # Add back to queue with delay
                await asyncio.sleep(event.retry_delay * (2 ** event.retry_count))
                await self.event_queue.put(event)
            else:
                # Send to dead letter queue
                self.dead_letter_queue.append(event)
        
        finally:
            event.completed_at = datetime.now()
            processing_time = time.time() - start_time
            self.metrics['total_processing_time'] += processing_time
            
            # Add to history
            self.event_history.append(event)
    
    def _find_handlers(self, event: Event) -> List[EventHandler]:
        """Find handlers matching event"""
        matching_handlers = []
        
        # Check specific event handlers
        for handler in self.handlers.get(event.name, []):
            if self._pattern_matches(handler.event_pattern, event.name):
                matching_handlers.append(handler)
        
        # Check global handlers
        for handler in self.global_handlers:
            if self._pattern_matches(handler.event_pattern, event.name):
                matching_handlers.append(handler)
        
        # Sort by priority
        matching_handlers.sort(key=lambda h: h.priority, reverse=True)
        
        return matching_handlers
    
    def _pattern_matches(self, pattern: str, event_name: str) -> bool:
        """Check if pattern matches event name"""
        if pattern == "*":
            return True
        
        event_pattern = EventPattern(pattern)
        return event_pattern.matches(event_name)
    
    def _check_conditions(self, event: Event, handler: EventHandler) -> bool:
        """Check handler conditions"""
        for condition in handler.conditions:
            try:
                if not condition(event):
                    return False
            except Exception:
                return False
        return True
    
    def _apply_filters(self, event: Event, handler: EventHandler) -> bool:
        """Apply handler filters"""
        for filter_func in handler.filters:
            try:
                if not filter_func(event):
                    return False
            except Exception:
                return False
        return True
    
    async def _execute_handler(self, event: Event, handler: EventHandler, worker_id: str):
        """Execute event handler"""
        start_time = time.time()
        
        try:
            # Apply middleware (pre-processing)
            for middleware in handler.middleware:
                if asyncio.iscoroutinefunction(middleware):
                    await middleware(event, 'pre')
                else:
                    middleware(event, 'pre')
            
            # Execute handler with timeout
            if handler.async_handler:
                if handler.timeout:
                    result = await asyncio.wait_for(
                        handler.handler_func(event),
                        timeout=handler.timeout
                    )
                else:
                    result = await handler.handler_func(event)
            else:
                # Run sync handler in thread pool
                loop = asyncio.get_event_loop()
                if handler.timeout:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, handler.handler_func, event),
                        timeout=handler.timeout
                    )
                else:
                    result = await loop.run_in_executor(None, handler.handler_func, event)
            
            event.result = result
            
            # Apply middleware (post-processing)
            for middleware in reversed(handler.middleware):
                if asyncio.iscoroutinefunction(middleware):
                    await middleware(event, 'post')
                else:
                    middleware(event, 'post')
            
            # Update handler statistics
            execution_time = time.time() - start_time
            handler.statistics.setdefault('executions', 0)
            handler.statistics.setdefault('total_time', 0.0)
            handler.statistics.setdefault('failures', 0)
            
            handler.statistics['executions'] += 1
            handler.statistics['total_time'] += execution_time
            
            self.metrics['handlers_executed'] += 1
            
        except asyncio.TimeoutError:
            handler.statistics.setdefault('timeouts', 0)
            handler.statistics['timeouts'] += 1
            raise
        except Exception as e:
            handler.statistics.setdefault('failures', 0)
            handler.statistics['failures'] += 1
            raise
    
    async def publish(self, event: Union[Event, str], data: Dict[str, Any] = None, 
                     priority: EventPriority = EventPriority.NORMAL,
                     **kwargs) -> str:
        """Publish event to bus"""
        if isinstance(event, str):
            event_obj = Event(
                name=event,
                data=data or {},
                priority=priority,
                **kwargs
            )
        else:
            event_obj = event
        
        try:
            await self.event_queue.put(event_obj)
            self.metrics['events_published'] += 1
            return event_obj.id
        except asyncio.QueueFull:
            raise RuntimeError("Event queue is full")
    
    def subscribe(self, event_pattern: str, handler: Callable,
                 priority: int = 0, conditions: List[Callable] = None,
                 filters: List[Callable] = None, timeout: Optional[float] = None,
                 name: str = None) -> str:
        """Subscribe to events"""
        handler_id = str(uuid.uuid4())
        
        # Determine if handler is async
        async_handler = asyncio.iscoroutinefunction(handler)
        
        event_handler = EventHandler(
            id=handler_id,
            name=name or handler.__name__,
            handler_func=handler,
            event_pattern=event_pattern,
            priority=priority,
            async_handler=async_handler,
            conditions=conditions or [],
            filters=filters or [],
            timeout=timeout,
            statistics={}
        )
        
        self.handler_registry[handler_id] = event_handler
        
        if event_pattern == "*":
            self.global_handlers.append(event_handler)
        else:
            # Extract base event name for optimization
            base_name = event_pattern.split('.')[0].split('*')[0]
            self.handlers[base_name].append(event_handler)
        
        return handler_id
    
    def unsubscribe(self, handler_id: str) -> bool:
        """Unsubscribe handler"""
        if handler_id not in self.handler_registry:
            return False
        
        handler = self.handler_registry[handler_id]
        
        # Remove from global handlers
        if handler in self.global_handlers:
            self.global_handlers.remove(handler)
        
        # Remove from specific handlers
        for handlers_list in self.handlers.values():
            if handler in handlers_list:
                handlers_list.remove(handler)
        
        del self.handler_registry[handler_id]
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics"""
        avg_processing_time = (
            self.metrics['total_processing_time'] / 
            max(self.metrics['events_processed'], 1)
        )
        
        return {
            **self.metrics,
            'queue_size': self.event_queue.qsize(),
            'active_workers': len(self.workers),
            'registered_handlers': len(self.handler_registry),
            'dead_letter_queue_size': len(self.dead_letter_queue),
            'average_processing_time': avg_processing_time,
            'success_rate': (
                self.metrics['events_processed'] / 
                max(self.metrics['events_published'], 1)
            )
        }


class EventScheduler:
    """Advanced event scheduler with cron support"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.scheduled_events = {}  # id -> ScheduledEvent
        self.schedule_heap = []  # (next_run_timestamp, event_id)
        self.running = False
        self.scheduler_task = None
        self.lock = asyncio.Lock()
    
    async def start(self):
        """Start event scheduler"""
        if self.running:
            return
        
        self.running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
    
    async def stop(self):
        """Stop event scheduler"""
        self.running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            try:
                await asyncio.sleep(1)  # Check every second
                
                current_time = time.time()
                events_to_run = []
                
                async with self.lock:
                    # Get events ready to run
                    while (self.schedule_heap and 
                           self.schedule_heap[0][0] <= current_time):
                        _, event_id = heapq.heappop(self.schedule_heap)
                        
                        if event_id in self.scheduled_events:
                            scheduled_event = self.scheduled_events[event_id]
                            if scheduled_event.enabled:
                                events_to_run.append(scheduled_event)
                
                # Execute scheduled events
                for scheduled_event in events_to_run:
                    await self._execute_scheduled_event(scheduled_event)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Scheduler error: {e}")
    
    async def _execute_scheduled_event(self, scheduled_event: ScheduledEvent):
        """Execute scheduled event"""
        try:
            # Apply jitter
            if scheduled_event.jitter > 0:
                import random
                jitter_delay = random.uniform(0, scheduled_event.jitter)
                await asyncio.sleep(jitter_delay)
            
            # Create event
            event = Event(
                name=scheduled_event.name,
                data=scheduled_event.event_data,
                event_type=EventType.SCHEDULED,
                source="scheduler",
                metadata={"scheduled_event_id": scheduled_event.id}
            )
            
            # Publish event
            await self.event_bus.publish(event)
            
            # Update scheduled event
            scheduled_event.run_count += 1
            scheduled_event.last_run = datetime.now()
            
            # Schedule next run
            await self._schedule_next_run(scheduled_event)
            
        except Exception as e:
            print(f"Scheduled event execution error: {e}")
            
            if scheduled_event.retry_on_failure:
                # Retry in 60 seconds
                next_run = datetime.now() + timedelta(seconds=60)
                await self._add_to_schedule(scheduled_event.id, next_run)
    
    async def _schedule_next_run(self, scheduled_event: ScheduledEvent):
        """Schedule next run for event"""
        # Check max runs
        if (scheduled_event.max_runs and 
            scheduled_event.run_count >= scheduled_event.max_runs):
            return
        
        # Calculate next run time
        if scheduled_event.schedule.startswith('@'):
            # Handle special schedules
            next_run = self._parse_special_schedule(scheduled_event.schedule)
        elif ' ' in scheduled_event.schedule:
            # Cron expression
            try:
                cron = croniter.croniter(scheduled_event.schedule, datetime.now())
                next_run = cron.get_next(datetime)
            except Exception:
                # Fallback to 1 hour interval
                next_run = datetime.now() + timedelta(hours=1)
        else:
            # Simple interval (seconds)
            try:
                interval = int(scheduled_event.schedule)
                next_run = datetime.now() + timedelta(seconds=interval)
            except ValueError:
                # Invalid schedule
                return
        
        scheduled_event.next_run = next_run
        await self._add_to_schedule(scheduled_event.id, next_run)
    
    def _parse_special_schedule(self, schedule: str) -> datetime:
        """Parse special schedule formats"""
        if schedule == '@yearly' or schedule == '@annually':
            return datetime.now().replace(month=1, day=1, hour=0, minute=0, second=0) + timedelta(days=365)
        elif schedule == '@monthly':
            current = datetime.now()
            if current.month == 12:
                return current.replace(year=current.year+1, month=1, day=1, hour=0, minute=0, second=0)
            else:
                return current.replace(month=current.month+1, day=1, hour=0, minute=0, second=0)
        elif schedule == '@weekly':
            return datetime.now() + timedelta(weeks=1)
        elif schedule == '@daily':
            return datetime.now() + timedelta(days=1)
        elif schedule == '@hourly':
            return datetime.now() + timedelta(hours=1)
        else:
            return datetime.now() + timedelta(hours=1)
    
    async def _add_to_schedule(self, event_id: str, next_run: datetime):
        """Add event to schedule heap"""
        async with self.lock:
            heapq.heappush(self.schedule_heap, (next_run.timestamp(), event_id))
    
    async def schedule_event(self, name: str, event_data: Dict[str, Any],
                           schedule: str, max_runs: Optional[int] = None,
                           enabled: bool = True, jitter: float = 0.0) -> str:
        """Schedule recurring event"""
        event_id = str(uuid.uuid4())
        
        # Calculate first run
        if schedule.startswith('@'):
            next_run = self._parse_special_schedule(schedule)
        elif ' ' in schedule:
            # Cron expression
            try:
                cron = croniter.croniter(schedule, datetime.now())
                next_run = cron.get_next(datetime)
            except Exception:
                next_run = datetime.now() + timedelta(hours=1)
        else:
            # Simple interval
            try:
                interval = int(schedule)
                next_run = datetime.now() + timedelta(seconds=interval)
            except ValueError:
                raise ValueError(f"Invalid schedule format: {schedule}")
        
        scheduled_event = ScheduledEvent(
            id=event_id,
            name=name,
            event_data=event_data,
            schedule=schedule,
            next_run=next_run,
            enabled=enabled,
            max_runs=max_runs,
            jitter=jitter
        )
        
        self.scheduled_events[event_id] = scheduled_event
        await self._add_to_schedule(event_id, next_run)
        
        return event_id
    
    async def unschedule_event(self, event_id: str) -> bool:
        """Unschedule event"""
        if event_id in self.scheduled_events:
            del self.scheduled_events[event_id]
            # Note: We don't remove from heap as it's too expensive
            # Instead, we check if event exists when processing
            return True
        return False
    
    def get_scheduled_events(self) -> List[ScheduledEvent]:
        """Get all scheduled events"""
        return list(self.scheduled_events.values())


class EventAutomation:
    """Event-driven automation system"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.rules = {}  # rule_id -> AutomationRule
        self.workflows = {}  # workflow_id -> Workflow
        self.active_workflows = {}  # instance_id -> WorkflowInstance
    
    async def add_rule(self, name: str, trigger_pattern: str, 
                      actions: List[Callable], conditions: List[Callable] = None) -> str:
        """Add automation rule"""
        rule_id = str(uuid.uuid4())
        
        # Create handler for rule
        async def rule_handler(event: Event):
            # Check conditions
            if conditions:
                for condition in conditions:
                    if not condition(event):
                        return
            
            # Execute actions
            for action in actions:
                try:
                    if asyncio.iscoroutinefunction(action):
                        await action(event)
                    else:
                        action(event)
                except Exception as e:
                    print(f"Automation action error: {e}")
        
        handler_id = self.event_bus.subscribe(
            trigger_pattern,
            rule_handler,
            name=f"automation_rule_{name}"
        )
        
        self.rules[rule_id] = {
            'id': rule_id,
            'name': name,
            'trigger_pattern': trigger_pattern,
            'actions': actions,
            'conditions': conditions or [],
            'handler_id': handler_id,
            'enabled': True
        }
        
        return rule_id
    
    async def remove_rule(self, rule_id: str) -> bool:
        """Remove automation rule"""
        if rule_id not in self.rules:
            return False
        
        rule = self.rules[rule_id]
        self.event_bus.unsubscribe(rule['handler_id'])
        del self.rules[rule_id]
        return True
    
    def get_rules(self) -> List[Dict[str, Any]]:
        """Get all automation rules"""
        return [
            {
                'id': rule['id'],
                'name': rule['name'],
                'trigger_pattern': rule['trigger_pattern'],
                'enabled': rule['enabled']
            }
            for rule in self.rules.values()
        ]


class EventSystem:
    """Main event system orchestrator"""
    
    def __init__(self, app):
        self.app = app
        self.event_bus = EventBus()
        self.scheduler = EventScheduler(self.event_bus)
        self.automation = EventAutomation(self.event_bus)
        self.plugins = {}
        
        # Built-in event handlers
        self._setup_builtin_handlers()
    
    async def start(self):
        """Start event system"""
        await self.event_bus.start()
        await self.scheduler.start()
        
        # Emit system started event
        await self.emit('system.started', {
            'timestamp': time.time(),
            'app_name': self.app.app_name
        })
    
    async def stop(self):
        """Stop event system"""
        # Emit system stopping event
        await self.emit('system.stopping', {
            'timestamp': time.time()
        })
        
        await self.scheduler.stop()
        await self.event_bus.stop()
    
    async def emit(self, event_name: str, data: Dict[str, Any] = None, **kwargs) -> str:
        """Emit event"""
        return await self.event_bus.publish(event_name, data, **kwargs)
    
    def on(self, event_pattern: str, handler: Callable, **kwargs) -> str:
        """Subscribe to events"""
        return self.event_bus.subscribe(event_pattern, handler, **kwargs)
    
    def off(self, handler_id: str) -> bool:
        """Unsubscribe from events"""
        return self.event_bus.unsubscribe(handler_id)
    
    async def schedule(self, name: str, event_data: Dict[str, Any], 
                      schedule: str, **kwargs) -> str:
        """Schedule recurring event"""
        return await self.scheduler.schedule_event(name, event_data, schedule, **kwargs)
    
    async def unschedule(self, event_id: str) -> bool:
        """Unschedule event"""
        return await self.scheduler.unschedule_event(event_id)
    
    def automate(self, name: str, trigger_pattern: str, 
                actions: List[Callable], conditions: List[Callable] = None) -> str:
        """Add automation rule"""
        return asyncio.create_task(
            self.automation.add_rule(name, trigger_pattern, actions, conditions)
        )
    
    def _setup_builtin_handlers(self):
        """Setup built-in event handlers"""
        
        # Application lifecycle events
        async def handle_app_error(event: Event):
            """Handle application errors"""
            if self.app.logger:
                await self.app.logger.error_async(
                    f"Application error event: {event.data.get('message', 'Unknown error')}",
                    extra=event.data
                )
        
        self.on('app.error', handle_app_error)
        
        # System health events
        async def handle_health_check(event: Event):
            """Handle health check requests"""
            health_data = {
                'status': 'healthy' if self.app.is_running else 'unhealthy',
                'timestamp': time.time(),
                'uptime': time.time() - (self.app.startup_time or time.time()),
                'components': {}
            }
            
            if hasattr(self.app, 'components'):
                for name, component in self.app.components.items():
                    health_data['components'][name] = {
                        'initialized': component.initialized,
                        'health_status': component.health_status
                    }
            
            await self.emit('health.status', health_data)
        
        self.on('health.check', handle_health_check)
        
        # Configuration change events
        async def handle_config_change(event: Event):
            """Handle configuration changes"""
            key = event.data.get('key')
            old_value = event.data.get('old_value')
            new_value = event.data.get('new_value')
            
            if self.app.logger:
                await self.app.logger.info_async(
                    f"Configuration changed: {key} = {new_value} (was: {old_value})"
                )
        
        self.on('config.changed', handle_config_change)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive event system metrics"""
        return {
            'event_bus': self.event_bus.get_metrics(),
            'scheduled_events': len(self.scheduler.scheduled_events),
            'automation_rules': len(self.automation.rules),
            'active_workflows': len(self.automation.active_workflows)
        }