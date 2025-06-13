#!/usr/bin/env python3
"""
Advanced Logger System
Enterprise-grade logging with structured output, async support, and comprehensive features
"""

import asyncio
import logging
import logging.handlers
import json
import time
import threading
import queue
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import traceback
from datetime import datetime, timezone
import gzip
import hashlib


class LogLevel(Enum):
    """Enhanced log levels"""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    SECURITY = 60
    AUDIT = 70


class LogFormat(Enum):
    """Log output formats"""
    STANDARD = auto()
    JSON = auto()
    STRUCTURED = auto()
    COMPACT = auto()
    DETAILED = auto()


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: float
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    extra: Dict[str, Any] = field(default_factory=dict)
    exception_info: Optional[str] = None
    stack_trace: Optional[str] = None
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class LogFilter:
    """Advanced log filtering"""
    
    def __init__(self):
        self.level_filters = {}
        self.module_filters = set()
        self.keyword_filters = set()
        self.pattern_filters = []
        
    def add_level_filter(self, level: LogLevel):
        """Add level filter"""
        self.level_filters[level.name] = level.value
    
    def add_module_filter(self, module_name: str):
        """Add module filter"""
        self.module_filters.add(module_name)
    
    def add_keyword_filter(self, keyword: str):
        """Add keyword filter"""
        self.keyword_filters.add(keyword.lower())
    
    def should_log(self, entry: LogEntry) -> bool:
        """Determine if entry should be logged"""
        # Level filtering
        if self.level_filters:
            entry_level = getattr(LogLevel, entry.level.upper(), LogLevel.INFO)
            min_level = min(self.level_filters.values())
            if entry_level.value < min_level:
                return False
        
        # Module filtering
        if self.module_filters and entry.module not in self.module_filters:
            return False
        
        # Keyword filtering
        if self.keyword_filters:
            message_lower = entry.message.lower()
            if not any(keyword in message_lower for keyword in self.keyword_filters):
                return False
        
        return True


class LogFormatter:
    """Advanced log formatters"""
    
    @staticmethod
    def format_standard(entry: LogEntry) -> str:
        """Standard format"""
        timestamp = datetime.fromtimestamp(entry.timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        return f"{timestamp} - {entry.logger_name} - {entry.level} - {entry.message}"
    
    @staticmethod
    def format_json(entry: LogEntry) -> str:
        """JSON format"""
        data = {
            'timestamp': entry.timestamp,
            'datetime': datetime.fromtimestamp(entry.timestamp, timezone.utc).isoformat(),
            'level': entry.level,
            'logger': entry.logger_name,
            'message': entry.message,
            'module': entry.module,
            'function': entry.function,
            'line': entry.line_number,
            'thread': entry.thread_id,
            'process': entry.process_id
        }
        
        if entry.extra:
            data['extra'] = entry.extra
        if entry.exception_info:
            data['exception'] = entry.exception_info
        if entry.correlation_id:
            data['correlation_id'] = entry.correlation_id
        if entry.user_id:
            data['user_id'] = entry.user_id
        if entry.session_id:
            data['session_id'] = entry.session_id
            
        return json.dumps(data, ensure_ascii=False)
    
    @staticmethod
    def format_structured(entry: LogEntry) -> str:
        """Structured format"""
        timestamp = datetime.fromtimestamp(entry.timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        base = f"[{timestamp}] {entry.level:<8} {entry.logger_name:<20} | {entry.message}"
        
        if entry.extra:
            extra_str = " | ".join(f"{k}={v}" for k, v in entry.extra.items())
            base += f" | {extra_str}"
        
        if entry.correlation_id:
            base += f" | corr_id={entry.correlation_id}"
            
        return base
    
    @staticmethod
    def format_compact(entry: LogEntry) -> str:
        """Compact format"""
        timestamp = datetime.fromtimestamp(entry.timestamp).strftime('%H:%M:%S')
        level_short = entry.level[0]  # First letter
        return f"{timestamp} {level_short} {entry.message}"
    
    @staticmethod
    def format_detailed(entry: LogEntry) -> str:
        """Detailed format"""
        timestamp = datetime.fromtimestamp(entry.timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        lines = [
            f"Timestamp: {timestamp}",
            f"Level: {entry.level}",
            f"Logger: {entry.logger_name}",
            f"Message: {entry.message}",
            f"Location: {entry.module}:{entry.function}:{entry.line_number}",
            f"Thread: {entry.thread_id}",
            f"Process: {entry.process_id}"
        ]
        
        if entry.extra:
            lines.append(f"Extra: {json.dumps(entry.extra, indent=2)}")
        
        if entry.exception_info:
            lines.append(f"Exception: {entry.exception_info}")
        
        if entry.stack_trace:
            lines.append(f"Stack Trace:\n{entry.stack_trace}")
        
        return "\n".join(lines) + "\n" + "-" * 80


class AsyncLogHandler:
    """Asynchronous log handler for high-performance logging"""
    
    def __init__(self, handler, queue_size: int = 10000):
        self.handler = handler
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.worker_task = None
        self.running = False
    
    async def start(self):
        """Start async handler"""
        self.running = True
        self.worker_task = asyncio.create_task(self._worker())
    
    async def stop(self):
        """Stop async handler"""
        self.running = False
        if self.worker_task:
            await self.worker_task
    
    async def _worker(self):
        """Worker coroutine"""
        while self.running:
            try:
                # Get log entry with timeout
                entry = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
                # Process entry
                await self._process_entry(entry)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Async log handler error: {e}")
    
    async def _process_entry(self, entry: LogEntry):
        """Process log entry"""
        try:
            if hasattr(self.handler, 'emit_async'):
                await self.handler.emit_async(entry)
            else:
                # Run sync handler in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.handler.emit, entry)
        except Exception as e:
            print(f"Error processing log entry: {e}")
    
    async def emit(self, entry: LogEntry):
        """Emit log entry"""
        try:
            await self.queue.put(entry)
        except asyncio.QueueFull:
            print("Log queue full, dropping entry")


class FileLogHandler:
    """Advanced file log handler with rotation and compression"""
    
    def __init__(self, filename: str, max_size: int = 50*1024*1024, 
                 backup_count: int = 10, compress: bool = True,
                 formatter: Callable = None):
        self.filename = Path(filename)
        self.max_size = max_size
        self.backup_count = backup_count
        self.compress = compress
        self.formatter = formatter or LogFormatter.format_json
        self.current_size = 0
        self.lock = threading.Lock()
        
        # Ensure directory exists
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize current size
        if self.filename.exists():
            self.current_size = self.filename.stat().st_size
    
    def emit(self, entry: LogEntry):
        """Emit log entry to file"""
        formatted = self.formatter(entry)
        
        with self.lock:
            # Check if rotation needed
            if self.current_size + len(formatted.encode()) > self.max_size:
                self._rotate_files()
            
            # Write to file
            with open(self.filename, 'a', encoding='utf-8') as f:
                f.write(formatted + '\n')
                self.current_size += len(formatted.encode()) + 1
    
    def _rotate_files(self):
        """Rotate log files"""
        if not self.filename.exists():
            return
        
        # Rotate existing files
        for i in range(self.backup_count - 1, 0, -1):
            old_name = f"{self.filename}.{i}"
            new_name = f"{self.filename}.{i + 1}"
            
            if self.compress:
                old_name += ".gz"
                new_name += ".gz"
            
            if Path(old_name).exists():
                Path(old_name).rename(new_name)
        
        # Move current file to .1
        backup_name = f"{self.filename}.1"
        if self.compress:
            backup_name += ".gz"
            self._compress_file(self.filename, backup_name)
        else:
            self.filename.rename(backup_name)
        
        # Reset current size
        self.current_size = 0
    
    def _compress_file(self, source: Path, target: str):
        """Compress log file"""
        with open(source, 'rb') as f_in:
            with gzip.open(target, 'wb') as f_out:
                f_out.writelines(f_in)
        source.unlink()


class ConsoleLogHandler:
    """Enhanced console log handler with colors"""
    
    COLORS = {
        'TRACE': '\033[90m',      # Dark gray
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'SECURITY': '\033[41m',   # Red background
        'AUDIT': '\033[44m',      # Blue background
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(self, use_colors: bool = True, formatter: Callable = None):
        self.use_colors = use_colors and sys.stdout.isatty()
        self.formatter = formatter or LogFormatter.format_structured
    
    def emit(self, entry: LogEntry):
        """Emit log entry to console"""
        formatted = self.formatter(entry)
        
        if self.use_colors:
            color = self.COLORS.get(entry.level.upper(), '')
            reset = self.COLORS['RESET']
            formatted = f"{color}{formatted}{reset}"
        
        print(formatted, file=sys.stdout)
        sys.stdout.flush()


class NetworkLogHandler:
    """Network log handler for centralized logging"""
    
    def __init__(self, host: str, port: int, protocol: str = 'tcp'):
        self.host = host
        self.port = port
        self.protocol = protocol
        self.formatter = LogFormatter.format_json
    
    async def emit_async(self, entry: LogEntry):
        """Emit log entry over network"""
        try:
            formatted = self.formatter(entry)
            
            if self.protocol == 'tcp':
                reader, writer = await asyncio.open_connection(self.host, self.port)
                writer.write(formatted.encode() + b'\n')
                await writer.drain()
                writer.close()
                await writer.wait_closed()
            else:
                # UDP implementation would go here
                pass
                
        except Exception as e:
            print(f"Network log handler error: {e}")


class AdvancedLogger:
    """Advanced logger with enterprise features"""
    
    def __init__(self, name: str, app=None):
        self.name = name
        self.app = app
        self.handlers = []
        self.async_handlers = []
        self.filters = []
        self.level = LogLevel.INFO
        self.correlation_id = None
        self.user_id = None
        self.session_id = None
        self.extra_context = {}
        
        # Performance metrics
        self.log_count = 0
        self.error_count = 0
        self.last_log_time = None
        
        # Async support
        self.async_queue = None
        self.async_worker = None
    
    def add_handler(self, handler):
        """Add log handler"""
        if isinstance(handler, AsyncLogHandler):
            self.async_handlers.append(handler)
        else:
            self.handlers.append(handler)
    
    def add_filter(self, log_filter: LogFilter):
        """Add log filter"""
        self.filters.append(log_filter)
    
    def set_level(self, level: Union[LogLevel, str, int]):
        """Set logging level"""
        if isinstance(level, str):
            self.level = LogLevel[level.upper()]
        elif isinstance(level, int):
            for log_level in LogLevel:
                if log_level.value == level:
                    self.level = log_level
                    break
        else:
            self.level = level
    
    def set_context(self, correlation_id: str = None, user_id: str = None, 
                   session_id: str = None, **extra):
        """Set logging context"""
        if correlation_id:
            self.correlation_id = correlation_id
        if user_id:
            self.user_id = user_id
        if session_id:
            self.session_id = session_id
        if extra:
            self.extra_context.update(extra)
    
    def _create_entry(self, level: LogLevel, message: str, **kwargs) -> LogEntry:
        """Create log entry"""
        # Get caller information
        frame = sys._getframe(3)  # Go up the stack
        
        entry = LogEntry(
            timestamp=time.time(),
            level=level.name,
            logger_name=self.name,
            message=message,
            module=frame.f_code.co_filename,
            function=frame.f_code.co_name,
            line_number=frame.f_lineno,
            thread_id=threading.get_ident(),
            process_id=os.getpid(),
            correlation_id=self.correlation_id,
            user_id=self.user_id,
            session_id=self.session_id
        )
        
        # Add extra context
        entry.extra.update(self.extra_context)
        entry.extra.update(kwargs.get('extra', {}))
        
        # Add exception info if present
        if 'exc_info' in kwargs and kwargs['exc_info']:
            entry.exception_info = traceback.format_exc()
            entry.stack_trace = ''.join(traceback.format_stack())
        
        return entry
    
    def _should_log(self, entry: LogEntry) -> bool:
        """Check if entry should be logged"""
        # Level check
        if LogLevel[entry.level].value < self.level.value:
            return False
        
        # Filter checks
        for log_filter in self.filters:
            if not log_filter.should_log(entry):
                return False
        
        return True
    
    def _emit(self, entry: LogEntry):
        """Emit log entry to handlers"""
        if not self._should_log(entry):
            return
        
        # Update metrics
        self.log_count += 1
        self.last_log_time = time.time()
        
        if entry.level in ['ERROR', 'CRITICAL', 'SECURITY']:
            self.error_count += 1
        
        # Emit to synchronous handlers
        for handler in self.handlers:
            try:
                handler.emit(entry)
            except Exception as e:
                print(f"Log handler error: {e}")
        
        # Emit to asynchronous handlers
        for handler in self.async_handlers:
            try:
                asyncio.create_task(handler.emit(entry))
            except Exception as e:
                print(f"Async log handler error: {e}")
    
    # Logging methods
    def trace(self, message: str, **kwargs):
        """Log trace message"""
        entry = self._create_entry(LogLevel.TRACE, message, **kwargs)
        self._emit(entry)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        entry = self._create_entry(LogLevel.DEBUG, message, **kwargs)
        self._emit(entry)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        entry = self._create_entry(LogLevel.INFO, message, **kwargs)
        self._emit(entry)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        entry = self._create_entry(LogLevel.WARNING, message, **kwargs)
        self._emit(entry)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        entry = self._create_entry(LogLevel.ERROR, message, **kwargs)
        self._emit(entry)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        entry = self._create_entry(LogLevel.CRITICAL, message, **kwargs)
        self._emit(entry)
    
    def security(self, message: str, **kwargs):
        """Log security message"""
        entry = self._create_entry(LogLevel.SECURITY, message, **kwargs)
        self._emit(entry)
    
    def audit(self, message: str, **kwargs):
        """Log audit message"""
        entry = self._create_entry(LogLevel.AUDIT, message, **kwargs)
        self._emit(entry)
    
    # Async versions
    async def trace_async(self, message: str, **kwargs):
        """Log trace message asynchronously"""
        self.trace(message, **kwargs)
    
    async def debug_async(self, message: str, **kwargs):
        """Log debug message asynchronously"""
        self.debug(message, **kwargs)
    
    async def info_async(self, message: str, **kwargs):
        """Log info message asynchronously"""
        self.info(message, **kwargs)
    
    async def warning_async(self, message: str, **kwargs):
        """Log warning message asynchronously"""
        self.warning(message, **kwargs)
    
    async def error_async(self, message: str, **kwargs):
        """Log error message asynchronously"""
        self.error(message, **kwargs)
    
    async def critical_async(self, message: str, **kwargs):
        """Log critical message asynchronously"""
        self.critical(message, **kwargs)
    
    async def security_async(self, message: str, **kwargs):
        """Log security message asynchronously"""
        self.security(message, **kwargs)
    
    async def audit_async(self, message: str, **kwargs):
        """Log audit message asynchronously"""
        self.audit(message, **kwargs)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get logging metrics"""
        return {
            'log_count': self.log_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.log_count, 1),
            'last_log_time': self.last_log_time,
            'handlers_count': len(self.handlers) + len(self.async_handlers),
            'filters_count': len(self.filters),
            'current_level': self.level.name
        }
    
    async def cleanup_async(self):
        """Cleanup async resources"""
        for handler in self.async_handlers:
            if hasattr(handler, 'stop'):
                await handler.stop()


class LoggerFactory:
    """Factory for creating configured loggers"""
    
    @staticmethod
    def create_logger(name: str, level: str = "INFO", 
                     console_output: bool = True, file_output: bool = False,
                     file_path: str = None, structured_logging: bool = False,
                     async_logging: bool = False, colors: bool = True,
                     json_format: bool = False, network_logging: bool = False,
                     network_host: str = None, network_port: int = None) -> AdvancedLogger:
        """Create and configure logger"""
        
        logger = AdvancedLogger(name)
        logger.set_level(level)
        
        # Console handler
        if console_output:
            formatter = LogFormatter.format_json if json_format else (
                LogFormatter.format_structured if structured_logging else
                LogFormatter.format_standard
            )
            console_handler = ConsoleLogHandler(use_colors=colors, formatter=formatter)
            
            if async_logging:
                async_handler = AsyncLogHandler(console_handler)
                logger.add_handler(async_handler)
            else:
                logger.add_handler(console_handler)
        
        # File handler
        if file_output and file_path:
            formatter = LogFormatter.format_json if json_format else LogFormatter.format_structured
            file_handler = FileLogHandler(file_path, formatter=formatter)
            
            if async_logging:
                async_handler = AsyncLogHandler(file_handler)
                logger.add_handler(async_handler)
            else:
                logger.add_handler(file_handler)
        
        # Network handler
        if network_logging and network_host and network_port:
            network_handler = NetworkLogHandler(network_host, network_port)
            async_handler = AsyncLogHandler(network_handler)
            logger.add_handler(async_handler)
        
        return logger
    
    @staticmethod
    def create_application_logger(app_name: str, config: Dict[str, Any] = None) -> AdvancedLogger:
        """Create logger for application with configuration"""
        if not config:
            config = {}
        
        return LoggerFactory.create_logger(
            name=f"{app_name}Logger",
            level=config.get('level', 'INFO'),
            console_output=config.get('console', True),
            file_output=config.get('file', True),
            file_path=config.get('file_path', f"logs/{app_name.lower().replace(' ', '_')}.log"),
            structured_logging=config.get('structured', True),
            async_logging=config.get('async', True),
            colors=config.get('colors', True),
            json_format=config.get('json', False),
            network_logging=config.get('network', False),
            network_host=config.get('network_host'),
            network_port=config.get('network_port')
        )