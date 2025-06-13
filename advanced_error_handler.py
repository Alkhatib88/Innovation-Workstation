#!/usr/bin/env python3
"""
Advanced Error Handler System
Comprehensive error handling, categorization, recovery, and monitoring
"""

import asyncio
import time
import traceback
import threading
import uuid
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import inspect
import sys
import os


class ErrorSeverity(Enum):
    """Error severity levels"""
    TRACE = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5
    FATAL = 6


class ErrorCategory(Enum):
    """Error categories for classification"""
    SYSTEM = auto()
    NETWORK = auto()
    DATABASE = auto()
    AUTHENTICATION = auto()
    AUTHORIZATION = auto()
    VALIDATION = auto()
    BUSINESS_LOGIC = auto()
    INTEGRATION = auto()
    PERFORMANCE = auto()
    SECURITY = auto()
    CONFIGURATION = auto()
    RESOURCE = auto()
    USER_INPUT = auto()
    EXTERNAL_SERVICE = auto()
    UNKNOWN = auto()


class ErrorAction(Enum):
    """Actions to take on errors"""
    LOG_ONLY = auto()
    RETRY = auto()
    FALLBACK = auto()
    ESCALATE = auto()
    SHUTDOWN = auto()
    IGNORE = auto()
    ALERT = auto()
    RECOVER = auto()


@dataclass
class ErrorContext:
    """Extended error context information"""
    component: str
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    request_id: Optional[str] = None
    environment: str = "production"
    version: Optional[str] = None
    build: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorRecord:
    """Comprehensive error record"""
    error_id: str
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    error_type: str
    message: str
    context: ErrorContext
    exception: Optional[Exception] = None
    stack_trace: Optional[str] = None
    call_stack: List[str] = field(default_factory=list)
    system_state: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_details: Optional[str] = None
    resolution_time: Optional[float] = None
    escalated: bool = False
    notifications_sent: List[str] = field(default_factory=list)
    hash: Optional[str] = None
    count: int = 1
    first_occurrence: float = field(default_factory=time.time)
    last_occurrence: float = field(default_factory=time.time)


class ErrorPattern:
    """Error pattern detection"""
    
    def __init__(self, pattern_id: str, description: str):
        self.pattern_id = pattern_id
        self.description = description
        self.conditions = []
        self.actions = []
        self.threshold = 1
        self.time_window = 300  # 5 minutes
        self.cooldown = 3600    # 1 hour
        self.last_triggered = 0
    
    def add_condition(self, condition: Callable[[ErrorRecord], bool]):
        """Add pattern condition"""
        self.conditions.append(condition)
    
    def add_action(self, action: Callable[[List[ErrorRecord]], None]):
        """Add pattern action"""
        self.actions.append(action)
    
    def matches(self, errors: List[ErrorRecord]) -> bool:
        """Check if pattern matches error sequence"""
        if time.time() - self.last_triggered < self.cooldown:
            return False
        
        recent_errors = [
            e for e in errors 
            if time.time() - e.timestamp <= self.time_window
        ]
        
        if len(recent_errors) < self.threshold:
            return False
        
        matching_errors = [
            e for e in recent_errors
            if all(condition(e) for condition in self.conditions)
        ]
        
        return len(matching_errors) >= self.threshold
    
    def trigger(self, errors: List[ErrorRecord]):
        """Trigger pattern actions"""
        self.last_triggered = time.time()
        for action in self.actions:
            try:
                action(errors)
            except Exception as e:
                print(f"Error pattern action failed: {e}")


class ErrorRecoveryStrategy:
    """Error recovery strategy"""
    
    def __init__(self, strategy_id: str, description: str):
        self.strategy_id = strategy_id
        self.description = description
        self.conditions = []
        self.recovery_function = None
        self.max_attempts = 3
        self.retry_delay = 1.0
        self.backoff_multiplier = 2.0
        self.success_rate = 0.0
        self.attempt_count = 0
        self.success_count = 0
    
    def can_recover(self, error_record: ErrorRecord) -> bool:
        """Check if strategy can recover from error"""
        return all(condition(error_record) for condition in self.conditions)
    
    async def attempt_recovery(self, error_record: ErrorRecord, context: Dict[str, Any] = None) -> bool:
        """Attempt error recovery"""
        if not self.recovery_function:
            return False
        
        self.attempt_count += 1
        
        for attempt in range(self.max_attempts):
            try:
                if asyncio.iscoroutinefunction(self.recovery_function):
                    result = await self.recovery_function(error_record, context)
                else:
                    result = self.recovery_function(error_record, context)
                
                if result:
                    self.success_count += 1
                    self.success_rate = self.success_count / self.attempt_count
                    return True
                
            except Exception as e:
                print(f"Recovery attempt {attempt + 1} failed: {e}")
            
            if attempt < self.max_attempts - 1:
                await asyncio.sleep(self.retry_delay * (self.backoff_multiplier ** attempt))
        
        self.success_rate = self.success_count / self.attempt_count
        return False


class ErrorNotificationSystem:
    """Error notification system"""
    
    def __init__(self):
        self.channels = {}
        self.rules = []
        self.rate_limits = {}
        self.notification_history = deque(maxlen=1000)
    
    def add_channel(self, name: str, handler: Callable):
        """Add notification channel"""
        self.channels[name] = handler
    
    def add_rule(self, condition: Callable[[ErrorRecord], bool], 
                 channels: List[str], cooldown: int = 300):
        """Add notification rule"""
        self.rules.append({
            'condition': condition,
            'channels': channels,
            'cooldown': cooldown,
            'last_sent': {}
        })
    
    async def process_error(self, error_record: ErrorRecord):
        """Process error for notifications"""
        for rule in self.rules:
            if rule['condition'](error_record):
                await self._send_notifications(error_record, rule)
    
    async def _send_notifications(self, error_record: ErrorRecord, rule: Dict):
        """Send notifications for error"""
        current_time = time.time()
        
        for channel in rule['channels']:
            if channel not in self.channels:
                continue
            
            # Check rate limit
            last_sent = rule['last_sent'].get(channel, 0)
            if current_time - last_sent < rule['cooldown']:
                continue
            
            try:
                handler = self.channels[channel]
                if asyncio.iscoroutinefunction(handler):
                    await handler(error_record)
                else:
                    handler(error_record)
                
                rule['last_sent'][channel] = current_time
                error_record.notifications_sent.append(channel)
                
                self.notification_history.append({
                    'timestamp': current_time,
                    'error_id': error_record.error_id,
                    'channel': channel,
                    'severity': error_record.severity.name
                })
                
            except Exception as e:
                print(f"Notification failed on channel {channel}: {e}")


class ErrorMetrics:
    """Error metrics collector"""
    
    def __init__(self):
        self.total_errors = 0
        self.errors_by_severity = defaultdict(int)
        self.errors_by_category = defaultdict(int)
        self.errors_by_component = defaultdict(int)
        self.error_rate_history = deque(maxlen=100)
        self.resolution_times = deque(maxlen=100)
        self.recovery_success_rate = 0.0
        self.mttr = 0.0  # Mean Time To Recovery
        self.mtbf = 0.0  # Mean Time Between Failures
        
    def record_error(self, error_record: ErrorRecord):
        """Record error metrics"""
        self.total_errors += 1
        self.errors_by_severity[error_record.severity.name] += 1
        self.errors_by_category[error_record.category.name] += 1
        self.errors_by_component[error_record.context.component] += 1
        
        # Calculate error rate (errors per minute)
        current_time = time.time()
        self.error_rate_history.append(current_time)
        
        # Clean old entries (older than 1 hour)
        cutoff_time = current_time - 3600
        while self.error_rate_history and self.error_rate_history[0] < cutoff_time:
            self.error_rate_history.popleft()
    
    def record_resolution(self, error_record: ErrorRecord):
        """Record error resolution"""
        if error_record.resolution_time:
            resolution_duration = error_record.resolution_time - error_record.timestamp
            self.resolution_times.append(resolution_duration)
            
            # Update MTTR
            if self.resolution_times:
                self.mttr = sum(self.resolution_times) / len(self.resolution_times)
    
    def get_current_error_rate(self) -> float:
        """Get current error rate (errors per minute)"""
        if not self.error_rate_history:
            return 0.0
        
        current_time = time.time()
        minute_ago = current_time - 60
        
        recent_errors = sum(1 for t in self.error_rate_history if t > minute_ago)
        return recent_errors
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        return {
            'total_errors': self.total_errors,
            'current_error_rate': self.get_current_error_rate(),
            'errors_by_severity': dict(self.errors_by_severity),
            'errors_by_category': dict(self.errors_by_category),
            'errors_by_component': dict(self.errors_by_component),
            'mttr': self.mttr,
            'mtbf': self.mtbf,
            'recovery_success_rate': self.recovery_success_rate
        }


class ErrorHandler:
    """Advanced error handler with comprehensive features"""
    
    def __init__(self, app, logger, auto_recovery: bool = True,
                 notification_enabled: bool = True, metrics_enabled: bool = True):
        self.app = app
        self.logger = logger
        self.auto_recovery = auto_recovery
        self.notification_enabled = notification_enabled
        self.metrics_enabled = metrics_enabled
        
        # Error storage and management
        self.errors = {}  # error_id -> ErrorRecord
        self.error_patterns = []
        self.recovery_strategies = []
        self.error_history = deque(maxlen=10000)
        self.error_hashes = {}  # hash -> error_id for deduplication
        
        # Systems
        self.notification_system = ErrorNotificationSystem()
        self.metrics = ErrorMetrics() if metrics_enabled else None
        
        # Threading
        self.lock = threading.Lock()
        
        # Configuration
        self.max_errors_in_memory = 1000
        self.error_persistence_path = None
        
        # Setup default patterns and strategies
        self._setup_default_patterns()
        self._setup_default_recovery_strategies()
        self._setup_default_notifications()
    
    def _setup_default_patterns(self):
        """Setup default error patterns"""
        # High frequency error pattern
        high_freq_pattern = ErrorPattern(
            "high_frequency_errors",
            "High frequency of errors in short time"
        )
        high_freq_pattern.threshold = 10
        high_freq_pattern.time_window = 60
        high_freq_pattern.add_action(lambda errors: self.logger.critical(
            f"High frequency error pattern detected: {len(errors)} errors in 1 minute"
        ))
        self.error_patterns.append(high_freq_pattern)
        
        # Critical error cascade pattern
        cascade_pattern = ErrorPattern(
            "error_cascade",
            "Multiple critical errors in sequence"
        )
        cascade_pattern.add_condition(lambda e: e.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL])
        cascade_pattern.threshold = 3
        cascade_pattern.time_window = 300
        cascade_pattern.add_action(lambda errors: self._trigger_emergency_shutdown(errors))
        self.error_patterns.append(cascade_pattern)
    
    def _setup_default_recovery_strategies(self):
        """Setup default recovery strategies"""
        # Network error recovery
        network_strategy = ErrorRecoveryStrategy(
            "network_retry",
            "Retry network operations with backoff"
        )
        network_strategy.conditions.append(lambda e: e.category == ErrorCategory.NETWORK)
        network_strategy.recovery_function = self._retry_network_operation
        network_strategy.max_attempts = 3
        self.recovery_strategies.append(network_strategy)
        
        # Database connection recovery
        db_strategy = ErrorRecoveryStrategy(
            "database_reconnect",
            "Reconnect to database on connection errors"
        )
        db_strategy.conditions.append(lambda e: e.category == ErrorCategory.DATABASE)
        db_strategy.recovery_function = self._recover_database_connection
        db_strategy.max_attempts = 5
        self.recovery_strategies.append(db_strategy)
    
    def _setup_default_notifications(self):
        """Setup default notification channels and rules"""
        # Console notification
        self.notification_system.add_channel('console', self._console_notification)
        
        # Log notification
        self.notification_system.add_channel('log', self._log_notification)
        
        # Critical error rule
        self.notification_system.add_rule(
            lambda e: e.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL],
            ['console', 'log'],
            cooldown=60
        )
        
        # Security error rule
        self.notification_system.add_rule(
            lambda e: e.category == ErrorCategory.SECURITY,
            ['console', 'log'],
            cooldown=30
        )
    
    def handle_error(self, error: Exception, context: Union[str, ErrorContext],
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    category: ErrorCategory = ErrorCategory.UNKNOWN,
                    auto_recover: bool = None) -> str:
        """Handle error with comprehensive processing"""
        
        # Normalize context
        if isinstance(context, str):
            error_context = ErrorContext(
                component=context,
                operation="unknown"
            )
        else:
            error_context = context
        
        # Create error record
        error_record = self._create_error_record(
            error, error_context, severity, category
        )
        
        # Store error
        with self.lock:
            self._store_error(error_record)
            self.error_history.append(error_record)
        
        # Process error
        asyncio.create_task(self._process_error_async(error_record, auto_recover))
        
        return error_record.error_id
    
    def _create_error_record(self, error: Exception, context: ErrorContext,
                           severity: ErrorSeverity, category: ErrorCategory) -> ErrorRecord:
        """Create comprehensive error record"""
        
        error_id = str(uuid.uuid4())
        
        # Generate error hash for deduplication
        error_signature = f"{type(error).__name__}:{str(error)}:{context.component}:{context.operation}"
        error_hash = hashlib.md5(error_signature.encode()).hexdigest()
        
        # Get system state
        system_state = self._capture_system_state()
        
        # Get call stack
        call_stack = [
            f"{frame.filename}:{frame.lineno} in {frame.name}"
            for frame in inspect.stack()[3:]  # Skip internal frames
        ]
        
        error_record = ErrorRecord(
            error_id=error_id,
            timestamp=time.time(),
            severity=severity,
            category=category,
            error_type=type(error).__name__,
            message=str(error),
            context=context,
            exception=error,
            stack_trace=traceback.format_exc(),
            call_stack=call_stack,
            system_state=system_state,
            hash=error_hash
        )
        
        return error_record
    
    def _store_error(self, error_record: ErrorRecord):
        """Store error with deduplication"""
        
        # Check for duplicate
        if error_record.hash in self.error_hashes:
            existing_id = self.error_hashes[error_record.hash]
            existing_record = self.errors[existing_id]
            existing_record.count += 1
            existing_record.last_occurrence = error_record.timestamp
            return
        
        # Store new error
        self.errors[error_record.error_id] = error_record
        self.error_hashes[error_record.hash] = error_record.error_id
        
        # Cleanup old errors if needed
        if len(self.errors) > self.max_errors_in_memory:
            self._cleanup_old_errors()
    
    def _cleanup_old_errors(self):
        """Clean up old errors to free memory"""
        # Remove oldest 10% of errors
        sorted_errors = sorted(
            self.errors.items(),
            key=lambda x: x[1].timestamp
        )
        
        cleanup_count = len(sorted_errors) // 10
        for error_id, error_record in sorted_errors[:cleanup_count]:
            del self.errors[error_id]
            if error_record.hash in self.error_hashes:
                del self.error_hashes[error_record.hash]
    
    async def _process_error_async(self, error_record: ErrorRecord, auto_recover: bool = None):
        """Process error asynchronously"""
        try:
            # Log error
            await self._log_error(error_record)
            
            # Record metrics
            if self.metrics:
                self.metrics.record_error(error_record)
            
            # Check patterns
            await self._check_error_patterns(error_record)
            
            # Send notifications
            if self.notification_enabled:
                await self.notification_system.process_error(error_record)
            
            # Attempt recovery
            if (auto_recover if auto_recover is not None else self.auto_recovery):
                await self._attempt_recovery(error_record)
            
        except Exception as e:
            # Avoid infinite recursion
            print(f"Error processing error: {e}")
    
    async def _log_error(self, error_record: ErrorRecord):
        """Log error with appropriate level"""
        log_level_map = {
            ErrorSeverity.TRACE: self.logger.trace,
            ErrorSeverity.LOW: self.logger.debug,
            ErrorSeverity.MEDIUM: self.logger.warning,
            ErrorSeverity.HIGH: self.logger.error,
            ErrorSeverity.CRITICAL: self.logger.critical,
            ErrorSeverity.FATAL: self.logger.critical
        }
        
        log_func = log_level_map.get(error_record.severity, self.logger.error)
        
        await log_func(
            f"[{error_record.category.name}] {error_record.message}",
            extra={
                'error_id': error_record.error_id,
                'error_type': error_record.error_type,
                'component': error_record.context.component,
                'operation': error_record.context.operation,
                'user_id': error_record.context.user_id,
                'correlation_id': error_record.context.correlation_id,
                'error_count': error_record.count
            }
        )
    
    async def _check_error_patterns(self, error_record: ErrorRecord):
        """Check for error patterns"""
        recent_errors = [
            e for e in self.error_history
            if time.time() - e.timestamp <= 3600  # Last hour
        ]
        
        for pattern in self.error_patterns:
            if pattern.matches(recent_errors):
                pattern.trigger(recent_errors)
    
    async def _attempt_recovery(self, error_record: ErrorRecord):
        """Attempt error recovery"""
        for strategy in self.recovery_strategies:
            if strategy.can_recover(error_record):
                error_record.recovery_attempted = True
                
                try:
                    success = await strategy.attempt_recovery(error_record)
                    error_record.recovery_successful = success
                    
                    if success:
                        error_record.resolution_time = time.time()
                        await self.logger.info_async(
                            f"Error recovery successful: {error_record.error_id}",
                            extra={'strategy': strategy.strategy_id}
                        )
                        break
                    
                except Exception as e:
                    await self.logger.error_async(
                        f"Recovery strategy failed: {strategy.strategy_id}",
                        extra={'error': str(e)}
                    )
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state"""
        try:
            return {
                'timestamp': time.time(),
                'thread_count': threading.active_count(),
                'process_id': os.getpid(),
                'memory_info': self._get_memory_info(),
                'cpu_info': self._get_cpu_info(),
                'open_files': self._get_open_files_count()
            }
        except Exception:
            return {'capture_failed': True}
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information"""
        try:
            import psutil
            process = psutil.Process()
            return {
                'rss': process.memory_info().rss,
                'vms': process.memory_info().vms,
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'unavailable': 'psutil not installed'}
        except Exception:
            return {'error': 'failed to get memory info'}
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information"""
        try:
            import psutil
            return {
                'cpu_percent': psutil.cpu_percent(),
                'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
        except ImportError:
            return {'unavailable': 'psutil not installed'}
        except Exception:
            return {'error': 'failed to get CPU info'}
    
    def _get_open_files_count(self) -> int:
        """Get open files count"""
        try:
            import psutil
            process = psutil.Process()
            return len(process.open_files())
        except ImportError:
            return -1
        except Exception:
            return -1
    
    # Recovery strategy implementations
    async def _retry_network_operation(self, error_record: ErrorRecord, context: Dict[str, Any] = None) -> bool:
        """Retry network operation"""
        # This would implement actual network retry logic
        await asyncio.sleep(1)  # Simulated retry delay
        return False  # Placeholder
    
    async def _recover_database_connection(self, error_record: ErrorRecord, context: Dict[str, Any] = None) -> bool:
        """Recover database connection"""
        # This would implement actual database recovery logic
        if hasattr(self.app, 'database') and self.app.database:
            try:
                await self.app.database.reconnect_async()
                return True
            except Exception:
                return False
        return False
    
    # Notification handlers
    async def _console_notification(self, error_record: ErrorRecord):
        """Console notification handler"""
        severity_icons = {
            ErrorSeverity.LOW: "🔵",
            ErrorSeverity.MEDIUM: "🟡",
            ErrorSeverity.HIGH: "🟠",
            ErrorSeverity.CRITICAL: "🔴",
            ErrorSeverity.FATAL: "💀"
        }
        
        icon = severity_icons.get(error_record.severity, "⚠️")
        print(f"\n{icon} ERROR ALERT {icon}")
        print(f"ID: {error_record.error_id}")
        print(f"Severity: {error_record.severity.name}")
        print(f"Category: {error_record.category.name}")
        print(f"Component: {error_record.context.component}")
        print(f"Message: {error_record.message}")
        print("-" * 50)
    
    async def _log_notification(self, error_record: ErrorRecord):
        """Log notification handler"""
        await self.logger.critical_async(
            f"ERROR NOTIFICATION: {error_record.message}",
            extra={
                'notification_type': 'error_alert',
                'error_id': error_record.error_id,
                'severity': error_record.severity.name,
                'category': error_record.category.name
            }
        )
    
    def _trigger_emergency_shutdown(self, errors: List[ErrorRecord]):
        """Trigger emergency shutdown"""
        self.logger.critical(
            f"EMERGENCY SHUTDOWN TRIGGERED: {len(errors)} critical errors detected"
        )
        # This would implement actual emergency shutdown logic
        # For now, just log the event
    
    # Public API methods
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary"""
        with self.lock:
            total_errors = len(self.errors)
            recent_errors = [
                e for e in self.error_history
                if time.time() - e.timestamp <= 3600
            ]
            
            summary = {
                'total_errors': total_errors,
                'recent_errors_count': len(recent_errors),
                'recent_errors': [
                    {
                        'id': e.error_id,
                        'timestamp': e.timestamp,
                        'severity': e.severity.name,
                        'category': e.category.name,
                        'message': e.message,
                        'count': e.count
                    }
                    for e in recent_errors[-10:]  # Last 10
                ]
            }
            
            if self.metrics:
                summary.update(self.metrics.get_metrics_summary())
            
            return summary
    
    def get_error_by_id(self, error_id: str) -> Optional[ErrorRecord]:
        """Get error by ID"""
        with self.lock:
            return self.errors.get(error_id)
    
    def clear_errors(self):
        """Clear all errors"""
        with self.lock:
            self.errors.clear()
            self.error_hashes.clear()
            self.error_history.clear()
            if self.metrics:
                self.metrics = ErrorMetrics()
    
    def add_recovery_strategy(self, strategy: ErrorRecoveryStrategy):
        """Add custom recovery strategy"""
        self.recovery_strategies.append(strategy)
    
    def add_error_pattern(self, pattern: ErrorPattern):
        """Add custom error pattern"""
        self.error_patterns.append(pattern)
    
    def set_notification_channel(self, name: str, handler: Callable):
        """Set custom notification channel"""
        self.notification_system.add_channel(name, handler)