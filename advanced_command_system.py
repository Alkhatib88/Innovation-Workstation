#!/usr/bin/env python3
"""
Advanced Command System
Comprehensive command processing with subcommands, auto-completion, and advanced features
"""

import asyncio
import shlex
import re
import time
import json
import inspect
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import argparse
import difflib


class CommandType(Enum):
    """Command types"""
    SYSTEM = auto()
    USER = auto()
    ADMIN = auto()
    DEBUG = auto()
    API = auto()


class PermissionLevel(Enum):
    """Permission levels"""
    PUBLIC = 1
    USER = 2
    MODERATOR = 3
    ADMIN = 4
    SUPERUSER = 5


class CommandStatus(Enum):
    """Command execution status"""
    SUCCESS = auto()
    FAILED = auto()
    PARTIAL = auto()
    UNAUTHORIZED = auto()
    NOT_FOUND = auto()
    INVALID_ARGS = auto()


@dataclass
class CommandArgument:
    """Command argument definition"""
    name: str
    arg_type: type = str
    required: bool = True
    default: Any = None
    description: str = ""
    choices: Optional[List[Any]] = None
    validation_func: Optional[Callable] = None
    help_text: str = ""


@dataclass
class CommandOption:
    """Command option definition"""
    name: str
    short_name: Optional[str] = None
    arg_type: type = bool
    default: Any = None
    description: str = ""
    help_text: str = ""


@dataclass
class CommandResult:
    """Command execution result"""
    status: CommandStatus
    message: str = ""
    data: Any = None
    execution_time: float = 0.0
    error: Optional[Exception] = None
    suggestions: List[str] = field(default_factory=list)


@dataclass
class CommandMetadata:
    """Command metadata"""
    name: str
    description: str
    usage: str = ""
    examples: List[str] = field(default_factory=list)
    category: str = "general"
    command_type: CommandType = CommandType.USER
    permission_level: PermissionLevel = PermissionLevel.PUBLIC
    aliases: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    deprecated: bool = False
    version_added: str = "1.0.0"
    author: str = ""
    see_also: List[str] = field(default_factory=list)


class CommandParser:
    """Advanced command argument parser"""
    
    def __init__(self, command_name: str):
        self.command_name = command_name
        self.arguments = {}
        self.options = {}
        self.subparsers = {}
        self.allow_unknown = False
        
    def add_argument(self, arg: CommandArgument):
        """Add command argument"""
        self.arguments[arg.name] = arg
    
    def add_option(self, option: CommandOption):
        """Add command option"""
        self.options[option.name] = option
    
    def add_subparser(self, name: str, parser: 'CommandParser'):
        """Add subcommand parser"""
        self.subparsers[name] = parser
    
    def parse(self, args: List[str]) -> Tuple[Dict[str, Any], List[str]]:
        """Parse command arguments"""
        parsed_args = {}
        remaining_args = args.copy()
        unknown_args = []
        
        # Check for subcommand first
        if remaining_args and remaining_args[0] in self.subparsers:
            subcommand = remaining_args.pop(0)
            subparser = self.subparsers[subcommand]
            sub_parsed, sub_unknown = subparser.parse(remaining_args)
            parsed_args['_subcommand'] = subcommand
            parsed_args.update(sub_parsed)
            unknown_args.extend(sub_unknown)
            return parsed_args, unknown_args
        
        # Parse options (flags starting with -)
        i = 0
        while i < len(remaining_args):
            arg = remaining_args[i]
            
            if arg.startswith('--'):
                # Long option
                option_name = arg[2:]
                if '=' in option_name:
                    option_name, value = option_name.split('=', 1)
                else:
                    value = None
                
                if option_name in self.options:
                    option = self.options[option_name]
                    if option.arg_type == bool:
                        parsed_args[option_name] = True
                    elif value is not None:
                        parsed_args[option_name] = self._convert_type(value, option.arg_type)
                    elif i + 1 < len(remaining_args) and not remaining_args[i + 1].startswith('-'):
                        i += 1
                        parsed_args[option_name] = self._convert_type(remaining_args[i], option.arg_type)
                    else:
                        parsed_args[option_name] = option.default
                else:
                    if self.allow_unknown:
                        unknown_args.append(arg)
                    else:
                        raise ValueError(f"Unknown option: --{option_name}")
                        
            elif arg.startswith('-') and len(arg) > 1:
                # Short option(s)
                short_opts = arg[1:]
                for short_opt in short_opts:
                    option = self._find_option_by_short_name(short_opt)
                    if option:
                        if option.arg_type == bool:
                            parsed_args[option.name] = True
                        elif i + 1 < len(remaining_args) and not remaining_args[i + 1].startswith('-'):
                            i += 1
                            parsed_args[option.name] = self._convert_type(remaining_args[i], option.arg_type)
                        else:
                            parsed_args[option.name] = option.default
                    else:
                        if self.allow_unknown:
                            unknown_args.append(f"-{short_opt}")
                        else:
                            raise ValueError(f"Unknown option: -{short_opt}")
            else:
                # Positional argument
                unknown_args.append(arg)
            
            i += 1
        
        # Parse positional arguments
        positional_args = [arg for arg in unknown_args if not arg.startswith('-')]
        unknown_args = [arg for arg in unknown_args if arg.startswith('-')]
        
        arg_names = list(self.arguments.keys())
        for i, value in enumerate(positional_args):
            if i < len(arg_names):
                arg_name = arg_names[i]
                arg_def = self.arguments[arg_name]
                parsed_args[arg_name] = self._convert_type(value, arg_def.arg_type)
        
        # Set defaults for missing arguments
        for arg_name, arg_def in self.arguments.items():
            if arg_name not in parsed_args:
                if arg_def.required and arg_def.default is None:
                    raise ValueError(f"Required argument missing: {arg_name}")
                parsed_args[arg_name] = arg_def.default
        
        # Set defaults for missing options
        for option_name, option_def in self.options.items():
            if option_name not in parsed_args:
                parsed_args[option_name] = option_def.default
        
        return parsed_args, unknown_args
    
    def _find_option_by_short_name(self, short_name: str) -> Optional[CommandOption]:
        """Find option by short name"""
        for option in self.options.values():
            if option.short_name == short_name:
                return option
        return None
    
    def _convert_type(self, value: str, target_type: type) -> Any:
        """Convert string value to target type"""
        if target_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type == list:
            return value.split(',')
        else:
            return str(value)
    
    def generate_help(self) -> str:
        """Generate help text"""
        help_text = [f"Usage: {self.command_name}"]
        
        # Add options to usage
        if self.options:
            help_text[0] += " [OPTIONS]"
        
        # Add arguments to usage
        for arg in self.arguments.values():
            if arg.required:
                help_text[0] += f" <{arg.name}>"
            else:
                help_text[0] += f" [{arg.name}]"
        
        # Add subcommands to usage
        if self.subparsers:
            help_text[0] += " <subcommand>"
        
        help_text.append("")
        
        # Add arguments section
        if self.arguments:
            help_text.append("Arguments:")
            for arg in self.arguments.values():
                required_text = " (required)" if arg.required else " (optional)"
                help_text.append(f"  {arg.name:<15} {arg.description}{required_text}")
            help_text.append("")
        
        # Add options section
        if self.options:
            help_text.append("Options:")
            for option in self.options.values():
                short_text = f"-{option.short_name}, " if option.short_name else "    "
                help_text.append(f"  {short_text}--{option.name:<12} {option.description}")
            help_text.append("")
        
        # Add subcommands section
        if self.subparsers:
            help_text.append("Subcommands:")
            for sub_name in self.subparsers.keys():
                help_text.append(f"  {sub_name}")
            help_text.append("")
        
        return "\n".join(help_text)


class Command:
    """Individual command definition"""
    
    def __init__(self, metadata: CommandMetadata, handler: Callable,
                 parser: Optional[CommandParser] = None):
        self.metadata = metadata
        self.handler = handler
        self.parser = parser or CommandParser(metadata.name)
        self.subcommands = {}
        self.middleware = []
        
        # Statistics
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.last_executed = None
        self.error_count = 0
    
    def add_subcommand(self, name: str, command: 'Command'):
        """Add subcommand"""
        self.subcommands[name] = command
        self.parser.add_subparser(name, command.parser)
    
    def add_middleware(self, middleware: Callable):
        """Add middleware function"""
        self.middleware.append(middleware)
    
    async def execute(self, args: List[str], context: Dict[str, Any] = None) -> CommandResult:
        """Execute command with middleware and error handling"""
        start_time = time.time()
        result = CommandResult(CommandStatus.SUCCESS)
        
        try:
            # Parse arguments
            parsed_args, unknown_args = self.parser.parse(args)
            
            # Check for subcommand
            if '_subcommand' in parsed_args:
                subcommand_name = parsed_args.pop('_subcommand')
                if subcommand_name in self.subcommands:
                    return await self.subcommands[subcommand_name].execute(args[1:], context)
                else:
                    result.status = CommandStatus.NOT_FOUND
                    result.message = f"Unknown subcommand: {subcommand_name}"
                    return result
            
            # Apply middleware
            for middleware_func in self.middleware:
                if asyncio.iscoroutinefunction(middleware_func):
                    middleware_result = await middleware_func(parsed_args, context)
                else:
                    middleware_result = middleware_func(parsed_args, context)
                
                if middleware_result is False:
                    result.status = CommandStatus.UNAUTHORIZED
                    result.message = "Command execution blocked by middleware"
                    return result
            
            # Execute handler
            if asyncio.iscoroutinefunction(self.handler):
                result.data = await self.handler(parsed_args, context)
            else:
                result.data = self.handler(parsed_args, context)
            
            if isinstance(result.data, CommandResult):
                result = result.data
            else:
                result.message = "Command executed successfully"
            
            self.execution_count += 1
            self.last_executed = time.time()
            
        except ValueError as e:
            result.status = CommandStatus.INVALID_ARGS
            result.message = str(e)
            result.error = e
            self.error_count += 1
            
        except Exception as e:
            result.status = CommandStatus.FAILED
            result.message = f"Command execution failed: {str(e)}"
            result.error = e
            self.error_count += 1
        
        finally:
            result.execution_time = time.time() - start_time
            self.total_execution_time += result.execution_time
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get command statistics"""
        avg_execution_time = (
            self.total_execution_time / self.execution_count
            if self.execution_count > 0 else 0
        )
        
        return {
            'execution_count': self.execution_count,
            'error_count': self.error_count,
            'success_rate': (self.execution_count - self.error_count) / max(self.execution_count, 1),
            'average_execution_time': avg_execution_time,
            'total_execution_time': self.total_execution_time,
            'last_executed': self.last_executed
        }


class SubCommandRegistry:
    """Registry for managing subcommands"""
    
    def __init__(self):
        self.registry = defaultdict(dict)  # parent_command -> {sub_name -> Command}
        self.reverse_registry = {}  # command_path -> Command
    
    def register(self, parent_path: str, subcommand_name: str, command: Command):
        """Register subcommand"""
        self.registry[parent_path][subcommand_name] = command
        full_path = f"{parent_path}.{subcommand_name}"
        self.reverse_registry[full_path] = command
    
    def get_subcommands(self, parent_path: str) -> Dict[str, Command]:
        """Get subcommands for parent"""
        return self.registry.get(parent_path, {})
    
    def find_command(self, path: str) -> Optional[Command]:
        """Find command by path"""
        return self.reverse_registry.get(path)
    
    def list_all_paths(self) -> List[str]:
        """List all command paths"""
        return list(self.reverse_registry.keys())


class CommandHistory:
    """Command execution history"""
    
    def __init__(self, max_size: int = 1000):
        self.history = deque(maxlen=max_size)
        self.session_id = None
        self.user_id = None
    
    def add_entry(self, command: str, args: List[str], result: CommandResult,
                  user_id: str = None, session_id: str = None):
        """Add history entry"""
        entry = {
            'timestamp': time.time(),
            'command': command,
            'args': args,
            'status': result.status.name,
            'execution_time': result.execution_time,
            'user_id': user_id or self.user_id,
            'session_id': session_id or self.session_id,
            'error': str(result.error) if result.error else None
        }
        self.history.append(entry)
    
    def get_recent(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent history entries"""
        return list(self.history)[-count:]
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search history"""
        results = []
        for entry in self.history:
            if (query.lower() in entry['command'].lower() or
                any(query.lower() in arg.lower() for arg in entry['args'])):
                results.append(entry)
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get history statistics"""
        if not self.history:
            return {'total_commands': 0}
        
        command_counts = defaultdict(int)
        status_counts = defaultdict(int)
        total_execution_time = 0
        
        for entry in self.history:
            command_counts[entry['command']] += 1
            status_counts[entry['status']] += 1
            total_execution_time += entry['execution_time']
        
        return {
            'total_commands': len(self.history),
            'unique_commands': len(command_counts),
            'most_used_commands': dict(command_counts.most_common(10)),
            'status_distribution': dict(status_counts),
            'total_execution_time': total_execution_time,
            'average_execution_time': total_execution_time / len(self.history)
        }


class CommandAutoComplete:
    """Command auto-completion system"""
    
    def __init__(self, command_system):
        self.command_system = command_system
        self.completion_cache = {}
    
    def complete(self, text: str, line: str, start_idx: int, end_idx: int) -> List[str]:
        """Complete command or arguments"""
        parts = line[:start_idx].split()
        
        if not parts:
            # Complete command names
            return self._complete_commands(text)
        
        command_name = parts[0]
        if command_name not in self.command_system.commands:
            return self._complete_commands(text)
        
        command = self.command_system.commands[command_name]
        
        # Determine what to complete
        if len(parts) == 1:
            # Complete subcommands or options
            completions = []
            
            # Add subcommands
            completions.extend(command.subcommands.keys())
            
            # Add options
            for option in command.parser.options.values():
                completions.append(f"--{option.name}")
                if option.short_name:
                    completions.append(f"-{option.short_name}")
            
            return [c for c in completions if c.startswith(text)]
        
        else:
            # Complete arguments or option values
            return self._complete_arguments(command, parts[1:], text)
    
    def _complete_commands(self, text: str) -> List[str]:
        """Complete command names"""
        commands = list(self.command_system.commands.keys())
        matches = [cmd for cmd in commands if cmd.startswith(text)]
        
        # Add fuzzy matches if no exact matches
        if not matches:
            matches = difflib.get_close_matches(text, commands, n=5, cutoff=0.6)
        
        return matches
    
    def _complete_arguments(self, command: Command, args: List[str], text: str) -> List[str]:
        """Complete command arguments"""
        # This is a simplified implementation
        # In practice, this would be more sophisticated
        completions = []
        
        # Check if completing an option value
        if len(args) >= 2 and args[-2].startswith('-'):
            option_name = args[-2].lstrip('-')
            option = command.parser.options.get(option_name)
            if option and option.choices:
                return [choice for choice in option.choices if str(choice).startswith(text)]
        
        # Check if completing a subcommand
        if args[0] in command.subcommands:
            subcommand = command.subcommands[args[0]]
            return self._complete_arguments(subcommand, args[1:], text)
        
        return completions


class CommandValidator:
    """Command validation system"""
    
    @staticmethod
    def validate_command_name(name: str) -> bool:
        """Validate command name"""
        if not name:
            return False
        
        # Command names should be alphanumeric with hyphens/underscores
        pattern = r'^[a-zA-Z][a-zA-Z0-9_-]*$'
        return bool(re.match(pattern, name))
    
    @staticmethod
    def validate_arguments(args: Dict[str, Any], parser: CommandParser) -> List[str]:
        """Validate command arguments"""
        errors = []
        
        # Check required arguments
        for arg_name, arg_def in parser.arguments.items():
            if arg_def.required and args.get(arg_name) is None:
                errors.append(f"Required argument '{arg_name}' is missing")
            
            # Validate choices
            value = args.get(arg_name)
            if value is not None and arg_def.choices and value not in arg_def.choices:
                errors.append(f"Invalid value for '{arg_name}': {value}. Choices: {arg_def.choices}")
            
            # Custom validation
            if value is not None and arg_def.validation_func:
                try:
                    if not arg_def.validation_func(value):
                        errors.append(f"Validation failed for argument '{arg_name}'")
                except Exception as e:
                    errors.append(f"Validation error for '{arg_name}': {str(e)}")
        
        return errors


class CommandSystem:
    """Advanced command system with comprehensive features"""
    
    def __init__(self, app):
        self.app = app
        self.commands = {}
        self.aliases = {}
        self.categories = defaultdict(list)
        self.subcommand_registry = SubCommandRegistry()
        self.history = CommandHistory()
        self.autocomplete = CommandAutoComplete(self)
        self.middleware = []
        
        # Configuration
        self.case_sensitive = False
        self.allow_unknown_commands = False
        self.command_prefix = ""
        
        # Statistics
        self.total_executions = 0
        self.total_execution_time = 0.0
        
        # Setup default commands
        self._register_default_commands()
    
    def _register_default_commands(self):
        """Register default system commands"""
        
        # Help command
        help_metadata = CommandMetadata(
            name="help",
            description="Show help information",
            category="system",
            command_type=CommandType.SYSTEM,
            examples=["help", "help status", "help --verbose"]
        )
        help_parser = CommandParser("help")
        help_parser.add_argument(CommandArgument("command", str, False, description="Command name"))
        help_parser.add_option(CommandOption("verbose", "v", bool, False, "Show detailed help"))
        help_parser.add_option(CommandOption("category", "c", str, None, "Filter by category"))
        
        help_command = Command(help_metadata, self._cmd_help, help_parser)
        self.register_command(help_command)
        
        # Status command with subcommands
        status_metadata = CommandMetadata(
            name="status",
            description="Show system status",
            category="system",
            examples=["status", "status --detailed", "status components"]
        )
        status_parser = CommandParser("status")
        status_parser.add_option(CommandOption("detailed", "d", bool, False, "Show detailed status"))
        status_parser.add_option(CommandOption("json", "j", bool, False, "Output in JSON format"))
        
        status_command = Command(status_metadata, self._cmd_status, status_parser)
        
        # Add status subcommands
        components_metadata = CommandMetadata(
            name="components",
            description="Show component status",
            category="system"
        )
        components_command = Command(components_metadata, self._cmd_status_components)
        status_command.add_subcommand("components", components_command)
        
        metrics_metadata = CommandMetadata(
            name="metrics",
            description="Show system metrics",
            category="system"
        )
        metrics_command = Command(metrics_metadata, self._cmd_status_metrics)
        status_command.add_subcommand("metrics", metrics_command)
        
        self.register_command(status_command)
        
        # History command
        history_metadata = CommandMetadata(
            name="history",
            description="Show command history",
            category="system",
            examples=["history", "history --count 20", "history --search error"]
        )
        history_parser = CommandParser("history")
        history_parser.add_option(CommandOption("count", "n", int, 10, "Number of entries to show"))
        history_parser.add_option(CommandOption("search", "s", str, None, "Search history"))
        history_parser.add_option(CommandOption("stats", None, bool, False, "Show statistics"))
        
        history_command = Command(history_metadata, self._cmd_history, history_parser)
        self.register_command(history_command)
        
        # Commands command (list all commands)
        commands_metadata = CommandMetadata(
            name="commands",
            description="List all available commands",
            category="system",
            aliases=["cmd", "cmds"]
        )
        commands_parser = CommandParser("commands")
        commands_parser.add_option(CommandOption("category", "c", str, None, "Filter by category"))
        commands_parser.add_option(CommandOption("type", "t", str, None, "Filter by type"))
        commands_parser.add_option(CommandOption("verbose", "v", bool, False, "Show detailed info"))
        
        commands_command = Command(commands_metadata, self._cmd_commands, commands_parser)
        self.register_command(commands_command)
        
        # Config command
        config_metadata = CommandMetadata(
            name="config",
            description="Manage configuration",
            category="system",
            examples=["config get log_level", "config set debug true", "config list"]
        )
        config_parser = CommandParser("config")
        
        # Config subcommands
        get_metadata = CommandMetadata(name="get", description="Get configuration value")
        get_parser = CommandParser("get")
        get_parser.add_argument(CommandArgument("key", str, True, description="Configuration key"))
        get_command = Command(get_metadata, self._cmd_config_get, get_parser)
        
        set_metadata = CommandMetadata(name="set", description="Set configuration value")
        set_parser = CommandParser("set")
        set_parser.add_argument(CommandArgument("key", str, True, description="Configuration key"))
        set_parser.add_argument(CommandArgument("value", str, True, description="Configuration value"))
        set_command = Command(set_metadata, self._cmd_config_set, set_parser)
        
        list_metadata = CommandMetadata(name="list", description="List all configuration")
        list_command = Command(list_metadata, self._cmd_config_list)
        
        config_command = Command(config_metadata, self._cmd_config, config_parser)
        config_command.add_subcommand("get", get_command)
        config_command.add_subcommand("set", set_command)
        config_command.add_subcommand("list", list_command)
        
        self.register_command(config_command)
        
        # Add aliases
        self.add_alias("h", "help")
        self.add_alias("s", "status")
        self.add_alias("quit", "exit")
    
    def register_command(self, command: Command):
        """Register a command"""
        if not CommandValidator.validate_command_name(command.metadata.name):
            raise ValueError(f"Invalid command name: {command.metadata.name}")
        
        if command.metadata.name in self.commands:
            raise ValueError(f"Command already exists: {command.metadata.name}")
        
        self.commands[command.metadata.name] = command
        self.categories[command.metadata.category].append(command.metadata.name)
        
        # Register aliases
        for alias in command.metadata.aliases:
            self.add_alias(alias, command.metadata.name)
    
    def add_alias(self, alias: str, command_name: str):
        """Add command alias"""
        if alias in self.aliases:
            raise ValueError(f"Alias already exists: {alias}")
        self.aliases[alias] = command_name
    
    def add_global_middleware(self, middleware: Callable):
        """Add global middleware"""
        self.middleware.append(middleware)
    
    async def execute_async(self, command_line: str, context: Dict[str, Any] = None) -> CommandResult:
        """Execute command asynchronously"""
        start_time = time.time()
        context = context or {}
        
        try:
            # Parse command line
            if not command_line.strip():
                return CommandResult(CommandStatus.SUCCESS, "No command entered")
            
            # Apply command prefix
            if self.command_prefix and not command_line.startswith(self.command_prefix):
                command_line = self.command_prefix + command_line
            
            # Parse arguments
            try:
                args = shlex.split(command_line)
            except ValueError:
                args = command_line.split()
            
            if not args:
                return CommandResult(CommandStatus.SUCCESS, "No command entered")
            
            command_name = args[0]
            command_args = args[1:]
            
            # Handle case sensitivity
            if not self.case_sensitive:
                command_name = command_name.lower()
            
            # Resolve aliases
            if command_name in self.aliases:
                command_name = self.aliases[command_name]
            
            # Find command
            if command_name not in self.commands:
                result = CommandResult(CommandStatus.NOT_FOUND)
                result.message = f"Unknown command: {command_name}"
                
                # Suggest similar commands
                suggestions = difflib.get_close_matches(
                    command_name, list(self.commands.keys()), n=3, cutoff=0.6
                )
                if suggestions:
                    result.suggestions = suggestions
                    result.message += f". Did you mean: {', '.join(suggestions)}?"
                
                return result
            
            command = self.commands[command_name]
            
            # Apply global middleware
            for middleware_func in self.middleware:
                if asyncio.iscoroutinefunction(middleware_func):
                    middleware_result = await middleware_func(command_name, command_args, context)
                else:
                    middleware_result = middleware_func(command_name, command_args, context)
                
                if middleware_result is False:
                    return CommandResult(
                        CommandStatus.UNAUTHORIZED,
                        "Command blocked by security middleware"
                    )
            
            # Execute command
            result = await command.execute(command_args, context)
            
            # Update statistics
            self.total_executions += 1
            self.total_execution_time += result.execution_time
            
            # Add to history
            self.history.add_entry(
                command_name, command_args, result,
                context.get('user_id'), context.get('session_id')
            )
            
            return result
            
        except Exception as e:
            result = CommandResult(CommandStatus.FAILED)
            result.message = f"Command execution error: {str(e)}"
            result.error = e
            result.execution_time = time.time() - start_time
            return result
    
    def execute(self, command_line: str, context: Dict[str, Any] = None) -> CommandResult:
        """Execute command synchronously"""
        return asyncio.run(self.execute_async(command_line, context))
    
    # Default command handlers
    async def _cmd_help(self, args: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        """Help command handler"""
        command_name = args.get('command')
        verbose = args.get('verbose', False)
        category = args.get('category')
        
        if command_name:
            # Show help for specific command
            if command_name in self.commands:
                command = self.commands[command_name]
                help_text = self._generate_command_help(command, verbose)
                return CommandResult(CommandStatus.SUCCESS, help_text)
            else:
                return CommandResult(
                    CommandStatus.NOT_FOUND,
                    f"Unknown command: {command_name}"
                )
        else:
            # Show general help
            help_text = self._generate_general_help(category, verbose)
            return CommandResult(CommandStatus.SUCCESS, help_text)
    
    async def _cmd_status(self, args: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        """Status command handler"""
        detailed = args.get('detailed', False)
        json_output = args.get('json', False)
        
        if hasattr(self.app, 'show_advanced_status'):
            self.app.show_advanced_status()
        else:
            self.app.show_status()
        
        return CommandResult(CommandStatus.SUCCESS, "Status displayed")
    
    async def _cmd_status_components(self, args: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        """Status components subcommand handler"""
        if hasattr(self.app, 'components'):
            component_info = []
            for name, component in self.app.components.items():
                component_info.append(f"  {'✅' if component.initialized else '❌'} {name}")
            
            message = "Component Status:\n" + "\n".join(component_info)
            return CommandResult(CommandStatus.SUCCESS, message)
        else:
            return CommandResult(CommandStatus.SUCCESS, "No component information available")
    
    async def _cmd_status_metrics(self, args: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        """Status metrics subcommand handler"""
        metrics = {
            'total_commands_executed': self.total_executions,
            'total_execution_time': self.total_execution_time,
            'average_execution_time': self.total_execution_time / max(self.total_executions, 1),
            'registered_commands': len(self.commands),
            'command_categories': len(self.categories)
        }
        
        message = "Command System Metrics:\n"
        for key, value in metrics.items():
            message += f"  {key}: {value}\n"
        
        return CommandResult(CommandStatus.SUCCESS, message)
    
    async def _cmd_history(self, args: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        """History command handler"""
        count = args.get('count', 10)
        search = args.get('search')
        stats = args.get('stats', False)
        
        if stats:
            stats_data = self.history.get_statistics()
            message = "Command History Statistics:\n"
            for key, value in stats_data.items():
                message += f"  {key}: {value}\n"
        elif search:
            entries = self.history.search(search)
            message = f"History search results for '{search}':\n"
            for entry in entries[-count:]:
                timestamp = time.strftime('%H:%M:%S', time.localtime(entry['timestamp']))
                message += f"  [{timestamp}] {entry['command']} {' '.join(entry['args'])}\n"
        else:
            entries = self.history.get_recent(count)
            message = f"Recent command history ({len(entries)} entries):\n"
            for i, entry in enumerate(entries, 1):
                timestamp = time.strftime('%H:%M:%S', time.localtime(entry['timestamp']))
                status_icon = "✅" if entry['status'] == 'SUCCESS' else "❌"
                message += f"  {i:2d}. [{timestamp}] {status_icon} {entry['command']} {' '.join(entry['args'])}\n"
        
        return CommandResult(CommandStatus.SUCCESS, message)
    
    async def _cmd_commands(self, args: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        """Commands command handler"""
        category = args.get('category')
        command_type = args.get('type')
        verbose = args.get('verbose', False)
        
        filtered_commands = []
        
        for name, command in self.commands.items():
            if category and command.metadata.category != category:
                continue
            if command_type and command.metadata.command_type.name.lower() != command_type.lower():
                continue
            filtered_commands.append((name, command))
        
        if not filtered_commands:
            return CommandResult(CommandStatus.SUCCESS, "No commands found matching criteria")
        
        # Sort by category and name
        filtered_commands.sort(key=lambda x: (x[1].metadata.category, x[0]))
        
        message = f"Available Commands ({len(filtered_commands)}):\n\n"
        
        current_category = None
        for name, command in filtered_commands:
            if command.metadata.category != current_category:
                current_category = command.metadata.category
                message += f"📁 {current_category.title()}:\n"
            
            aliases_text = f" (aliases: {', '.join(command.metadata.aliases)})" if command.metadata.aliases else ""
            
            if verbose:
                message += f"  {name:<15} - {command.metadata.description}{aliases_text}\n"
                if command.metadata.examples:
                    message += f"                  Examples: {', '.join(command.metadata.examples[:2])}\n"
            else:
                message += f"  {name:<15} - {command.metadata.description}{aliases_text}\n"
        
        return CommandResult(CommandStatus.SUCCESS, message)
    
    async def _cmd_config(self, args: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        """Config command handler"""
        return CommandResult(
            CommandStatus.SUCCESS,
            "Use subcommands: get, set, list. Type 'help config' for details."
        )
    
    async def _cmd_config_get(self, args: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        """Config get subcommand handler"""
        key = args.get('key')
        
        if hasattr(self.app, 'config') and self.app.config:
            value = self.app.config.get(key)
            if value is not None:
                return CommandResult(CommandStatus.SUCCESS, f"{key} = {value}")
            else:
                return CommandResult(CommandStatus.NOT_FOUND, f"Configuration key '{key}' not found")
        else:
            return CommandResult(CommandStatus.FAILED, "Configuration system not available")
    
    async def _cmd_config_set(self, args: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        """Config set subcommand handler"""
        key = args.get('key')
        value = args.get('value')
        
        if hasattr(self.app, 'config') and self.app.config:
            # Type conversion
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '', 1).isdigit():
                value = float(value)
            
            self.app.config.set(key, value)
            return CommandResult(CommandStatus.SUCCESS, f"Set {key} = {value}")
        else:
            return CommandResult(CommandStatus.FAILED, "Configuration system not available")
    
    async def _cmd_config_list(self, args: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        """Config list subcommand handler"""
        if hasattr(self.app, 'config') and self.app.config:
            message = "Configuration:\n"
            for key, value in self.app.config.config.items():
                message += f"  {key}: {value}\n"
            return CommandResult(CommandStatus.SUCCESS, message)
        else:
            return CommandResult(CommandStatus.FAILED, "Configuration system not available")
    
    def _generate_command_help(self, command: Command, verbose: bool = False) -> str:
        """Generate help text for a specific command"""
        help_text = [f"Command: {command.metadata.name}"]
        help_text.append(f"Description: {command.metadata.description}")
        
        if command.metadata.aliases:
            help_text.append(f"Aliases: {', '.join(command.metadata.aliases)}")
        
        help_text.append("")
        help_text.append(command.parser.generate_help())
        
        if verbose:
            if command.metadata.examples:
                help_text.append("Examples:")
                for example in command.metadata.examples:
                    help_text.append(f"  {example}")
                help_text.append("")
            
            if command.subcommands:
                help_text.append("Subcommands:")
                for sub_name, sub_command in command.subcommands.items():
                    help_text.append(f"  {sub_name:<15} - {sub_command.metadata.description}")
                help_text.append("")
            
            # Statistics
            stats = command.get_statistics()
            help_text.append("Statistics:")
            help_text.append(f"  Execution count: {stats['execution_count']}")
            help_text.append(f"  Success rate: {stats['success_rate']:.2%}")
            help_text.append(f"  Average execution time: {stats['average_execution_time']:.3f}s")
        
        return "\n".join(help_text)
    
    def _generate_general_help(self, category: str = None, verbose: bool = False) -> str:
        """Generate general help text"""
        help_text = [f"🚀 {self.app.app_name} Command System"]
        help_text.append("=" * 50)
        help_text.append("")
        
        # Filter commands by category if specified
        commands_to_show = []
        for name, command in self.commands.items():
            if category is None or command.metadata.category == category:
                commands_to_show.append((name, command))
        
        # Group by category
        categories = defaultdict(list)
        for name, command in commands_to_show:
            categories[command.metadata.category].append((name, command))
        
        # Display by category
        for cat_name in sorted(categories.keys()):
            help_text.append(f"📁 {cat_name.title()} Commands:")
            
            commands_in_cat = sorted(categories[cat_name], key=lambda x: x[0])
            for name, command in commands_in_cat:
                aliases_text = f" ({', '.join(command.metadata.aliases)})" if command.metadata.aliases else ""
                help_text.append(f"  {name:<15} - {command.metadata.description}{aliases_text}")
            
            help_text.append("")
        
        # Usage information
        help_text.append("Usage:")
        help_text.append("  <command> [arguments] [options]")
        help_text.append("  help <command>  - Show detailed help for a command")
        help_text.append("  commands        - List all available commands")
        help_text.append("")
        
        if verbose:
            # System statistics
            help_text.append("System Statistics:")
            help_text.append(f"  Total commands: {len(self.commands)}")
            help_text.append(f"  Total categories: {len(self.categories)}")
            help_text.append(f"  Total executions: {self.total_executions}")
            help_text.append(f"  Total aliases: {len(self.aliases)}")
        
        return "\n".join(help_text)