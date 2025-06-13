#!/usr/bin/env python3
"""
Advanced Configuration and Settings Management
Hierarchical configuration, environment management, and dynamic settings
"""

import asyncio
import os
import json
import yaml
import toml
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, ChainMap
import copy
import re
import configparser
from datetime import datetime, timedelta
import weakref


class ConfigFormat(Enum):
    """Configuration file formats"""
    JSON = auto()
    YAML = auto()
    TOML = auto()
    INI = auto()
    ENV = auto()
    XML = auto()


class ConfigScope(Enum):
    """Configuration scope levels"""
    SYSTEM = auto()      # System-wide settings
    APPLICATION = auto() # Application-level settings
    USER = auto()        # User-specific settings
    SESSION = auto()     # Session-specific settings
    TEMPORARY = auto()   # Temporary settings


class ValidationSeverity(Enum):
    """Validation severity levels"""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class ConfigSchema:
    """Configuration schema definition"""
    key: str
    data_type: type
    default_value: Any = None
    required: bool = False
    description: str = ""
    validation_rules: List[Callable] = field(default_factory=list)
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None
    deprecated: bool = False
    migration_path: Optional[str] = None


@dataclass
class ConfigChange:
    """Configuration change record"""
    timestamp: float
    key: str
    old_value: Any
    new_value: Any
    source: str
    user_id: Optional[str] = None
    reason: str = ""


@dataclass
class ValidationResult:
    """Configuration validation result"""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    key: str
    suggested_value: Any = None


class ConfigValidator:
    """Configuration validation system"""
    
    def __init__(self):
        self.schemas = {}  # key -> ConfigSchema
        self.custom_validators = {}  # key -> Callable
    
    def register_schema(self, schema: ConfigSchema):
        """Register configuration schema"""
        self.schemas[schema.key] = schema
    
    def register_validator(self, key: str, validator: Callable[[Any], bool]):
        """Register custom validator"""
        self.custom_validators[key] = validator
    
    def validate_value(self, key: str, value: Any) -> ValidationResult:
        """Validate configuration value"""
        if key not in self.schemas:
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message=f"No schema defined for key: {key}",
                key=key
            )
        
        schema = self.schemas[key]
        
        # Type validation
        if not isinstance(value, schema.data_type):
            try:
                # Try to convert
                if schema.data_type == int:
                    value = int(value)
                elif schema.data_type == float:
                    value = float(value)
                elif schema.data_type == bool:
                    value = str(value).lower() in ('true', '1', 'yes', 'on')
                elif schema.data_type == str:
                    value = str(value)
                else:
                    return ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Invalid type for {key}: expected {schema.data_type.__name__}, got {type(value).__name__}",
                        key=key,
                        suggested_value=schema.default_value
                    )
            except (ValueError, TypeError):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Cannot convert {value} to {schema.data_type.__name__} for key {key}",
                    key=key,
                    suggested_value=schema.default_value
                )
        
        # Range validation
        if schema.min_value is not None and value < schema.min_value:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Value {value} for {key} is below minimum {schema.min_value}",
                key=key,
                suggested_value=schema.min_value
            )
        
        if schema.max_value is not None and value > schema.max_value:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Value {value} for {key} is above maximum {schema.max_value}",
                key=key,
                suggested_value=schema.max_value
            )
        
        # Allowed values validation
        if schema.allowed_values and value not in schema.allowed_values:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Value {value} for {key} not in allowed values: {schema.allowed_values}",
                key=key,
                suggested_value=schema.allowed_values[0] if schema.allowed_values else schema.default_value
            )
        
        # Pattern validation
        if schema.pattern and isinstance(value, str):
            if not re.match(schema.pattern, value):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Value {value} for {key} does not match pattern {schema.pattern}",
                    key=key,
                    suggested_value=schema.default_value
                )
        
        # Custom validation rules
        for rule in schema.validation_rules:
            try:
                if not rule(value):
                    return ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Custom validation failed for {key} with value {value}",
                        key=key,
                        suggested_value=schema.default_value
                    )
            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Validation rule error for {key}: {str(e)}",
                    key=key,
                    suggested_value=schema.default_value
                )
        
        # Custom validator
        if key in self.custom_validators:
            try:
                if not self.custom_validators[key](value):
                    return ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Custom validator failed for {key}",
                        key=key,
                        suggested_value=schema.default_value
                    )
            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Custom validator error for {key}: {str(e)}",
                    key=key,
                    suggested_value=schema.default_value
                )
        
        # Deprecation warning
        if schema.deprecated:
            message = f"Configuration key {key} is deprecated"
            if schema.migration_path:
                message += f". Use {schema.migration_path} instead"
            
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message=message,
                key=key
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message=f"Valid value for {key}",
            key=key
        )


class ConfigurationStore:
    """Configuration storage backend"""
    
    def __init__(self, file_path: Path, format_type: ConfigFormat):
        self.file_path = file_path
        self.format_type = format_type
        self.data = {}
        self.lock = threading.Lock()
        self.last_modified = 0
        
    def load(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if not self.file_path.exists():
                return {}
            
            with self.lock:
                # Check if file was modified
                current_mtime = self.file_path.stat().st_mtime
                if current_mtime <= self.last_modified:
                    return self.data.copy()
                
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if self.format_type == ConfigFormat.JSON:
                    self.data = json.loads(content)
                elif self.format_type == ConfigFormat.YAML:
                    self.data = yaml.safe_load(content) or {}
                elif self.format_type == ConfigFormat.TOML:
                    self.data = toml.loads(content)
                elif self.format_type == ConfigFormat.INI:
                    parser = configparser.ConfigParser()
                    parser.read_string(content)
                    self.data = {section: dict(parser[section]) for section in parser.sections()}
                elif self.format_type == ConfigFormat.ENV:
                    self.data = self._parse_env_format(content)
                
                self.last_modified = current_mtime
                return self.data.copy()
        
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {self.file_path}: {str(e)}")
    
    def save(self, data: Dict[str, Any]):
        """Save configuration to file"""
        try:
            with self.lock:
                # Ensure directory exists
                self.file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Format data
                if self.format_type == ConfigFormat.JSON:
                    content = json.dumps(data, indent=2, ensure_ascii=False)
                elif self.format_type == ConfigFormat.YAML:
                    content = yaml.dump(data, default_flow_style=False, allow_unicode=True)
                elif self.format_type == ConfigFormat.TOML:
                    content = toml.dumps(data)
                elif self.format_type == ConfigFormat.INI:
                    content = self._format_as_ini(data)
                elif self.format_type == ConfigFormat.ENV:
                    content = self._format_as_env(data)
                else:
                    raise ValueError(f"Unsupported format: {self.format_type}")
                
                # Write to temporary file first
                temp_path = self.file_path.with_suffix('.tmp')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # Atomic rename
                temp_path.replace(self.file_path)
                
                self.data = data.copy()
                self.last_modified = self.file_path.stat().st_mtime
        
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration to {self.file_path}: {str(e)}")
    
    def _parse_env_format(self, content: str) -> Dict[str, Any]:
        """Parse environment file format"""
        data = {}
        for line in content.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    data[key] = value
        return data
    
    def _format_as_ini(self, data: Dict[str, Any]) -> str:
        """Format data as INI"""
        lines = []
        for section, values in data.items():
            lines.append(f'[{section}]')
            if isinstance(values, dict):
                for key, value in values.items():
                    lines.append(f'{key} = {value}')
            lines.append('')
        return '\n'.join(lines)
    
    def _format_as_env(self, data: Dict[str, Any]) -> str:
        """Format data as environment file"""
        lines = []
        for key, value in data.items():
            if isinstance(value, str) and (' ' in value or '"' in value):
                value = f'"{value}"'
            lines.append(f'{key}={value}')
        return '\n'.join(lines)


class ConfigurationError(Exception):
    """Configuration-related error"""
    pass


class ConfigManager:
    """Advanced configuration manager"""
    
    def __init__(self, app):
        self.app = app
        self.stores = {}  # scope -> ConfigurationStore
        self.validator = ConfigValidator()
        self.change_history = deque(maxlen=1000)
        self.change_listeners = defaultdict(list)  # key -> List[Callable]
        self.merged_config = {}
        self.lock = threading.Lock()
        
        # Configuration hierarchy (higher priority first)
        self.scope_hierarchy = [
            ConfigScope.TEMPORARY,
            ConfigScope.SESSION,
            ConfigScope.USER,
            ConfigScope.APPLICATION,
            ConfigScope.SYSTEM
        ]
        
        # Auto-reload settings
        self.auto_reload = True
        self.reload_interval = 5.0  # seconds
        self.reload_task = None
        
        self._setup_default_schemas()
    
    def _setup_default_schemas(self):
        """Setup default configuration schemas"""
        schemas = [
            ConfigSchema(
                key="app.name",
                data_type=str,
                default_value="Innovation Workstation",
                required=True,
                description="Application name"
            ),
            ConfigSchema(
                key="app.version",
                data_type=str,
                default_value="1.0.0",
                required=True,
                description="Application version",
                pattern=r'^\d+\.\d+\.\d+$'
            ),
            ConfigSchema(
                key="app.debug",
                data_type=bool,
                default_value=False,
                description="Enable debug mode"
            ),
            ConfigSchema(
                key="logging.level",
                data_type=str,
                default_value="INFO",
                allowed_values=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                description="Logging level"
            ),
            ConfigSchema(
                key="logging.max_file_size",
                data_type=int,
                default_value=10485760,  # 10MB
                min_value=1024,  # 1KB
                max_value=1073741824,  # 1GB
                description="Maximum log file size in bytes"
            ),
            ConfigSchema(
                key="database.host",
                data_type=str,
                default_value="localhost",
                description="Database host"
            ),
            ConfigSchema(
                key="database.port",
                data_type=int,
                default_value=5432,
                min_value=1,
                max_value=65535,
                description="Database port"
            ),
            ConfigSchema(
                key="security.encryption_level",
                data_type=str,
                default_value="HIGH",
                allowed_values=["LOW", "MEDIUM", "HIGH", "MAXIMUM"],
                description="Security encryption level"
            ),
            ConfigSchema(
                key="performance.max_workers",
                data_type=int,
                default_value=4,
                min_value=1,
                max_value=32,
                description="Maximum number of worker threads"
            ),
            ConfigSchema(
                key="network.timeout",
                data_type=float,
                default_value=30.0,
                min_value=1.0,
                max_value=300.0,
                description="Network timeout in seconds"
            )
        ]
        
        for schema in schemas:
            self.validator.register_schema(schema)
    
    async def setup(self) -> bool:
        """Setup configuration manager"""
        try:
            # Initialize configuration stores
            config_dir = self.app.file_manager.config_dir
            
            stores_config = {
                ConfigScope.SYSTEM: (config_dir / "system.json", ConfigFormat.JSON),
                ConfigScope.APPLICATION: (config_dir / "application.yaml", ConfigFormat.YAML),
                ConfigScope.USER: (config_dir / "user.toml", ConfigFormat.TOML),
                ConfigScope.SESSION: (config_dir / "session.json", ConfigFormat.JSON),
                ConfigScope.TEMPORARY: (config_dir / "temp.json", ConfigFormat.JSON)
            }
            
            for scope, (file_path, format_type) in stores_config.items():
                self.stores[scope] = ConfigurationStore(file_path, format_type)
            
            # Load initial configuration
            await self._reload_configuration()
            
            # Start auto-reload if enabled
            if self.auto_reload:
                await self._start_auto_reload()
            
            self.app.logger.info("Configuration manager initialized successfully")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "ConfigManager.setup")
            return False
    
    async def _reload_configuration(self):
        """Reload configuration from all stores"""
        try:
            with self.lock:
                # Load from all stores
                store_data = {}
                for scope in self.scope_hierarchy:
                    if scope in self.stores:
                        store_data[scope] = self.stores[scope].load()
                
                # Merge configurations (higher priority first)
                merged = {}
                for scope in reversed(self.scope_hierarchy):
                    if scope in store_data:
                        merged.update(store_data[scope])
                
                self.merged_config = merged
        
        except Exception as e:
            self.app.error_handler.handle_error(e, "ConfigManager._reload_configuration")
    
    async def _start_auto_reload(self):
        """Start automatic configuration reloading"""
        if self.reload_task:
            return
        
        async def reload_loop():
            while self.auto_reload:
                try:
                    await asyncio.sleep(self.reload_interval)
                    await self._reload_configuration()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.app.error_handler.handle_error(e, "ConfigManager.reload_loop")
        
        self.reload_task = asyncio.create_task(reload_loop())
    
    async def get(self, key: str, default: Any = None, 
                 scope: ConfigScope = None) -> Any:
        """Get configuration value"""
        try:
            if scope:
                # Get from specific scope
                if scope in self.stores:
                    store_data = self.stores[scope].load()
                    return self._get_nested_value(store_data, key, default)
                return default
            else:
                # Get from merged configuration
                return self._get_nested_value(self.merged_config, key, default)
        
        except Exception as e:
            self.app.error_handler.handle_error(e, f"ConfigManager.get({key})")
            return default
    
    async def set(self, key: str, value: Any, scope: ConfigScope = ConfigScope.APPLICATION,
                 validate: bool = True, user_id: str = None, reason: str = "") -> bool:
        """Set configuration value"""
        try:
            # Validate value
            if validate:
                validation_result = self.validator.validate_value(key, value)
                if not validation_result.is_valid:
                    if validation_result.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                        self.app.logger.error(f"Configuration validation failed: {validation_result.message}")
                        return False
                    else:
                        self.app.logger.warning(f"Configuration validation warning: {validation_result.message}")
            
            # Get current value for change tracking
            old_value = await self.get(key)
            
            # Update configuration
            if scope not in self.stores:
                self.app.logger.error(f"Invalid configuration scope: {scope}")
                return False
            
            store_data = self.stores[scope].load()
            self._set_nested_value(store_data, key, value)
            self.stores[scope].save(store_data)
            
            # Reload merged configuration
            await self._reload_configuration()
            
            # Record change
            change = ConfigChange(
                timestamp=time.time(),
                key=key,
                old_value=old_value,
                new_value=value,
                source=scope.name,
                user_id=user_id,
                reason=reason
            )
            
            self.change_history.append(change)
            
            # Notify listeners
            await self._notify_change_listeners(key, old_value, value)
            
            # Emit configuration change event
            if hasattr(self.app, 'event_system'):
                await self.app.event_system.emit('config.changed', {
                    'key': key,
                    'old_value': old_value,
                    'new_value': value,
                    'scope': scope.name,
                    'user_id': user_id,
                    'reason': reason
                })
            
            return True
        
        except Exception as e:
            self.app.error_handler.handle_error(e, f"ConfigManager.set({key})")
            return False
    
    async def delete(self, key: str, scope: ConfigScope = ConfigScope.APPLICATION) -> bool:
        """Delete configuration key"""
        try:
            if scope not in self.stores:
                return False
            
            store_data = self.stores[scope].load()
            
            if self._delete_nested_value(store_data, key):
                self.stores[scope].save(store_data)
                await self._reload_configuration()
                return True
            
            return False
        
        except Exception as e:
            self.app.error_handler.handle_error(e, f"ConfigManager.delete({key})")
            return False
    
    def _get_nested_value(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get nested dictionary value using dot notation"""
        keys = key.split('.')
        current = data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def _set_nested_value(self, data: Dict[str, Any], key: str, value: Any):
        """Set nested dictionary value using dot notation"""
        keys = key.split('.')
        current = data
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def _delete_nested_value(self, data: Dict[str, Any], key: str) -> bool:
        """Delete nested dictionary value using dot notation"""
        keys = key.split('.')
        current = data
        
        for k in keys[:-1]:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return False
        
        if isinstance(current, dict) and keys[-1] in current:
            del current[keys[-1]]
            return True
        
        return False
    
    async def _notify_change_listeners(self, key: str, old_value: Any, new_value: Any):
        """Notify configuration change listeners"""
        try:
            # Notify specific key listeners
            if key in self.change_listeners:
                for listener in self.change_listeners[key]:
                    try:
                        if asyncio.iscoroutinefunction(listener):
                            await listener(key, old_value, new_value)
                        else:
                            listener(key, old_value, new_value)
                    except Exception as e:
                        self.app.error_handler.handle_error(e, f"ConfigManager.change_listener({key})")
            
            # Notify wildcard listeners
            if '*' in self.change_listeners:
                for listener in self.change_listeners['*']:
                    try:
                        if asyncio.iscoroutinefunction(listener):
                            await listener(key, old_value, new_value)
                        else:
                            listener(key, old_value, new_value)
                    except Exception as e:
                        self.app.error_handler.handle_error(e, "ConfigManager.wildcard_listener")
        
        except Exception as e:
            self.app.error_handler.handle_error(e, "ConfigManager._notify_change_listeners")
    
    def add_change_listener(self, key: str, listener: Callable):
        """Add configuration change listener"""
        self.change_listeners[key].append(listener)
    
    def remove_change_listener(self, key: str, listener: Callable):
        """Remove configuration change listener"""
        if key in self.change_listeners and listener in self.change_listeners[key]:
            self.change_listeners[key].remove(listener)
    
    def get_all_keys(self, scope: ConfigScope = None) -> List[str]:
        """Get all configuration keys"""
        try:
            if scope:
                if scope in self.stores:
                    store_data = self.stores[scope].load()
                    return self._flatten_keys(store_data)
                return []
            else:
                return self._flatten_keys(self.merged_config)
        
        except Exception as e:
            self.app.error_handler.handle_error(e, "ConfigManager.get_all_keys")
            return []
    
    def _flatten_keys(self, data: Dict[str, Any], prefix: str = "") -> List[str]:
        """Flatten nested dictionary keys"""
        keys = []
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.append(full_key)
            
            if isinstance(value, dict):
                keys.extend(self._flatten_keys(value, full_key))
        
        return keys
    
    def get_change_history(self, key: str = None, limit: int = 100) -> List[ConfigChange]:
        """Get configuration change history"""
        history = list(self.change_history)
        
        if key:
            history = [change for change in history if change.key == key]
        
        return history[-limit:]
    
    def validate_all(self) -> List[ValidationResult]:
        """Validate all configuration values"""
        results = []
        
        for key in self.get_all_keys():
            value = asyncio.run(self.get(key))
            if value is not None:
                result = self.validator.validate_value(key, value)
                results.append(result)
        
        return results
    
    def export_configuration(self, scope: ConfigScope = None, 
                           format_type: ConfigFormat = ConfigFormat.JSON) -> str:
        """Export configuration to string"""
        try:
            if scope:
                if scope in self.stores:
                    data = self.stores[scope].load()
                else:
                    data = {}
            else:
                data = self.merged_config.copy()
            
            if format_type == ConfigFormat.JSON:
                return json.dumps(data, indent=2, ensure_ascii=False)
            elif format_type == ConfigFormat.YAML:
                return yaml.dump(data, default_flow_style=False, allow_unicode=True)
            elif format_type == ConfigFormat.TOML:
                return toml.dumps(data)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
        
        except Exception as e:
            self.app.error_handler.handle_error(e, "ConfigManager.export_configuration")
            return ""
    
    async def import_configuration(self, content: str, format_type: ConfigFormat,
                                  scope: ConfigScope = ConfigScope.APPLICATION,
                                  merge: bool = True) -> bool:
        """Import configuration from string"""
        try:
            # Parse content
            if format_type == ConfigFormat.JSON:
                data = json.loads(content)
            elif format_type == ConfigFormat.YAML:
                data = yaml.safe_load(content) or {}
            elif format_type == ConfigFormat.TOML:
                data = toml.loads(content)
            else:
                raise ValueError(f"Unsupported import format: {format_type}")
            
            if merge:
                # Merge with existing configuration
                if scope in self.stores:
                    existing_data = self.stores[scope].load()
                    existing_data.update(data)
                    data = existing_data
            
            # Save to store
            if scope in self.stores:
                self.stores[scope].save(data)
                await self._reload_configuration()
                return True
            
            return False
        
        except Exception as e:
            self.app.error_handler.handle_error(e, "ConfigManager.import_configuration")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get configuration statistics"""
        return {
            'total_keys': len(self.get_all_keys()),
            'scopes': len(self.stores),
            'change_history_size': len(self.change_history),
            'listeners': sum(len(listeners) for listeners in self.change_listeners.values()),
            'schemas': len(self.validator.schemas),
            'auto_reload': self.auto_reload,
            'reload_interval': self.reload_interval
        }
    
    async def cleanup(self):
        """Cleanup configuration manager"""
        try:
            # Stop auto-reload
            self.auto_reload = False
            if self.reload_task:
                self.reload_task.cancel()
                try:
                    await self.reload_task
                except asyncio.CancelledError:
                    pass
            
            # Save any pending changes
            for scope, store in self.stores.items():
                if hasattr(store, 'pending_changes'):
                    store.save(store.data)
            
            self.app.logger.info("Configuration manager cleanup completed")
        
        except Exception as e:
            self.app.error_handler.handle_error(e, "ConfigManager.cleanup")


class EnvironmentManager:
    """Advanced environment variable management"""
    
    def __init__(self, app):
        self.app = app
        self.env_files = []  # List of .env files to load
        self.env_cache = {}
        self.env_watchers = {}  # file -> watcher
        self.custom_env = {}
        self.lock = threading.Lock()
    
    async def setup(self) -> bool:
        """Setup environment manager"""
        try:
            # Load environment files
            env_files = [
                self.app.file_manager.config_dir / ".env",
                self.app.file_manager.config_dir / ".env.local",
                self.app.file_manager.config_dir / f".env.{os.getenv('ENVIRONMENT', 'development')}"
            ]
            
            for env_file in env_files:
                if env_file.exists():
                    await self._load_env_file(env_file)
            
            # Load system environment variables
            self._load_system_env()
            
            self.app.logger.info("Environment manager initialized successfully")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "EnvironmentManager.setup")
            return False
    
    async def _load_env_file(self, file_path: Path):
        """Load environment file"""
        try:
            content = await self.app.file_manager.read_text_async(file_path)
            if content:
                with self.lock:
                    for line in content.strip().split('\n'):
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"\'')
                            
                            # Expand variables
                            value = self._expand_variables(value)
                            
                            self.env_cache[key] = value
                            
                            # Set in os.environ if not already set
                            if key not in os.environ:
                                os.environ[key] = value
            
            self.env_files.append(file_path)
        
        except Exception as e:
            self.app.error_handler.handle_error(e, f"EnvironmentManager._load_env_file({file_path})")
    
    def _load_system_env(self):
        """Load system environment variables"""
        with self.lock:
            for key, value in os.environ.items():
                if key not in self.env_cache:
                    self.env_cache[key] = value
    
    def _expand_variables(self, value: str) -> str:
        """Expand environment variables in value"""
        import re
        
        # Replace ${VAR} and $VAR patterns
        def replace_var(match):
            var_name = match.group(1) or match.group(2)
            return os.getenv(var_name, match.group(0))
        
        # Handle ${VAR} format
        value = re.sub(r'\$\{([^}]+)\}', replace_var, value)
        
        # Handle $VAR format
        value = re.sub(r'\$([A-Za-z_][A-Za-z0-9_]*)', replace_var, value)
        
        return value
    
    def get(self, key: str, default: str = None) -> Optional[str]:
        """Get environment variable"""
        with self.lock:
            return self.env_cache.get(key, default)
    
    def set(self, key: str, value: str, persist: bool = False):
        """Set environment variable"""
        with self.lock:
            self.env_cache[key] = value
            os.environ[key] = value
            
            if persist:
                self.custom_env[key] = value
    
    def delete(self, key: str):
        """Delete environment variable"""
        with self.lock:
            self.env_cache.pop(key, None)
            os.environ.pop(key, None)
            self.custom_env.pop(key, None)
    
    def list_all(self) -> Dict[str, str]:
        """List all environment variables"""
        with self.lock:
            return self.env_cache.copy()
    
    def save_custom_env(self) -> bool:
        """Save custom environment variables to file"""
        try:
            if not self.custom_env:
                return True
            
            env_file = self.app.file_manager.config_dir / ".env.custom"
            lines = []
            
            for key, value in self.custom_env.items():
                # Escape value if it contains spaces or special characters
                if ' ' in value or '"' in value or '\n' in value:
                    value = f'"{value.replace('"', '\\"')}"'
                lines.append(f"{key}={value}")
            
            content = '\n'.join(lines)
            return asyncio.run(self.app.file_manager.write_text_async(env_file, content))
        
        except Exception as e:
            self.app.error_handler.handle_error(e, "EnvironmentManager.save_custom_env")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get environment statistics"""
        with self.lock:
            return {
                'total_variables': len(self.env_cache),
                'custom_variables': len(self.custom_env),
                'env_files_loaded': len(self.env_files),
                'system_variables': len(os.environ)
            }