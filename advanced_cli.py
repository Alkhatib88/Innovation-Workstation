#!/usr/bin/env python3
"""
Advanced CLI System
Comprehensive command-line interface with auto-completion, colors, history, and advanced features
"""

import asyncio
import sys
import os
import tty
import termios
import signal
import threading
import time
import json
import readline
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import difflib
import re
import shutil


class ColorCode(Enum):
    """ANSI color codes"""
    # Standard colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'
    
    # Styles
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    STRIKETHROUGH = '\033[9m'


class PromptStyle(Enum):
    """Prompt display styles"""
    SIMPLE = auto()
    DETAILED = auto()
    MINIMAL = auto()
    POWERLINE = auto()
    CUSTOM = auto()


@dataclass
class ColorTheme:
    """Color theme configuration"""
    name: str = "default"
    prompt: str = ColorCode.BRIGHT_GREEN.value
    command: str = ColorCode.WHITE.value
    argument: str = ColorCode.CYAN.value
    option: str = ColorCode.YELLOW.value
    error: str = ColorCode.RED.value
    warning: str = ColorCode.YELLOW.value
    info: str = ColorCode.BLUE.value
    success: str = ColorCode.GREEN.value
    highlight: str = ColorCode.BRIGHT_WHITE.value
    secondary: str = ColorCode.BRIGHT_BLACK.value
    reset: str = ColorCode.RESET.value


class CLITheme:
    """CLI theme manager"""
    
    THEMES = {
        'default': ColorTheme(),
        'dark': ColorTheme(
            name="dark",
            prompt=ColorCode.BRIGHT_CYAN.value,
            command=ColorCode.BRIGHT_WHITE.value,
            argument=ColorCode.BRIGHT_BLUE.value,
            option=ColorCode.BRIGHT_YELLOW.value,
            error=ColorCode.BRIGHT_RED.value,
            warning=ColorCode.BRIGHT_YELLOW.value,
            info=ColorCode.BRIGHT_BLUE.value,
            success=ColorCode.BRIGHT_GREEN.value,
            highlight=ColorCode.WHITE.value,
            secondary=ColorCode.BRIGHT_BLACK.value
        ),
        'light': ColorTheme(
            name="light",
            prompt=ColorCode.BLUE.value,
            command=ColorCode.BLACK.value,
            argument=ColorCode.CYAN.value,
            option=ColorCode.MAGENTA.value,
            error=ColorCode.RED.value,
            warning=ColorCode.YELLOW.value,
            info=ColorCode.BLUE.value,
            success=ColorCode.GREEN.value,
            highlight=ColorCode.BLACK.value,
            secondary=ColorCode.BLACK.value
        ),
        'matrix': ColorTheme(
            name="matrix",
            prompt=ColorCode.BRIGHT_GREEN.value,
            command=ColorCode.GREEN.value,
            argument=ColorCode.BRIGHT_GREEN.value,
            option=ColorCode.GREEN.value,
            error=ColorCode.RED.value,
            warning=ColorCode.YELLOW.value,
            info=ColorCode.GREEN.value,
            success=ColorCode.BRIGHT_GREEN.value,
            highlight=ColorCode.BRIGHT_WHITE.value,
            secondary=ColorCode.BLACK.value
        )
    }
    
    def __init__(self, theme_name: str = 'default'):
        self.current_theme = self.THEMES.get(theme_name, self.THEMES['default'])
        self.colors_enabled = self._check_color_support()
    
    def _check_color_support(self) -> bool:
        """Check if terminal supports colors"""
        return (hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and
                os.getenv('TERM', '').lower() != 'dumb')
    
    def colorize(self, text: str, color: str, bold: bool = False, 
                underline: bool = False) -> str:
        """Apply color and styles to text"""
        if not self.colors_enabled:
            return text
        
        codes = [color]
        if bold:
            codes.append(ColorCode.BOLD.value)
        if underline:
            codes.append(ColorCode.UNDERLINE.value)
        
        return f"{''.join(codes)}{text}{self.current_theme.reset}"
    
    def error(self, text: str) -> str:
        """Format error text"""
        return self.colorize(text, self.current_theme.error, bold=True)
    
    def warning(self, text: str) -> str:
        """Format warning text"""
        return self.colorize(text, self.current_theme.warning)
    
    def info(self, text: str) -> str:
        """Format info text"""
        return self.colorize(text, self.current_theme.info)
    
    def success(self, text: str) -> str:
        """Format success text"""
        return self.colorize(text, self.current_theme.success, bold=True)
    
    def highlight(self, text: str) -> str:
        """Format highlighted text"""
        return self.colorize(text, self.current_theme.highlight, bold=True)
    
    def command(self, text: str) -> str:
        """Format command text"""
        return self.colorize(text, self.current_theme.command, bold=True)
    
    def argument(self, text: str) -> str:
        """Format argument text"""
        return self.colorize(text, self.current_theme.argument)
    
    def option(self, text: str) -> str:
        """Format option text"""
        return self.colorize(text, self.current_theme.option)
    
    def set_theme(self, theme_name: str):
        """Set color theme"""
        if theme_name in self.THEMES:
            self.current_theme = self.THEMES[theme_name]


class AutoComplete:
    """Advanced auto-completion system"""
    
    def __init__(self, command_system):
        self.command_system = command_system
        self.completion_cache = {}
        self.file_completion_enabled = True
        self.history_completion_enabled = True
        self.fuzzy_matching = True
        
        # Setup readline
        self._setup_readline()
    
    def _setup_readline(self):
        """Setup readline for auto-completion"""
        try:
            readline.set_completer(self.complete)
            readline.parse_and_bind("tab: complete")
            readline.set_completer_delims(' \t\n`!@#$%^&*()=+[{]}\\|;:\'",<>?')
            
            # Enable history
            readline.set_history_length(1000)
            
            # Load history file
            history_file = Path.home() / '.workstation_history'
            if history_file.exists():
                readline.read_history_file(str(history_file))
        except ImportError:
            # Readline not available
            pass
    
    def complete(self, text: str, state: int) -> Optional[str]:
        """Main completion function for readline"""
        if state == 0:
            # First call - generate completions
            line = readline.get_line_buffer()
            start_idx = readline.get_begidx()
            end_idx = readline.get_endidx()
            
            self.completions = self.get_completions(text, line, start_idx, end_idx)
        
        try:
            return self.completions[state]
        except IndexError:
            return None
    
    def get_completions(self, text: str, line: str, start_idx: int, end_idx: int) -> List[str]:
        """Get completions for current context"""
        # Parse the command line
        parts = line[:start_idx].split()
        
        if not parts and not text:
            # Complete command names at the beginning
            return self._complete_commands(text)
        
        if not parts:
            # Still completing the command name
            return self._complete_commands(text)
        
        command_name = parts[0]
        
        # Check if command exists
        if command_name not in self.command_system.commands:
            # Suggest similar commands
            return self._complete_commands(text)
        
        command = self.command_system.commands[command_name]
        
        # Determine completion context
        if len(parts) == 1 and start_idx > len(command_name):
            # Complete subcommands or options
            return self._complete_command_parts(command, text)
        elif len(parts) > 1:
            # Complete arguments or subcommand parts
            return self._complete_arguments(command, parts[1:], text)
        else:
            return []
    
    def _complete_commands(self, text: str) -> List[str]:
        """Complete command names"""
        commands = list(self.command_system.commands.keys())
        aliases = list(self.command_system.aliases.keys())
        all_commands = commands + aliases
        
        # Exact prefix matches
        matches = [cmd for cmd in all_commands if cmd.startswith(text)]
        
        # Fuzzy matches if enabled and no exact matches
        if not matches and self.fuzzy_matching and len(text) >= 2:
            fuzzy_matches = difflib.get_close_matches(
                text, all_commands, n=10, cutoff=0.4
            )
            matches.extend(fuzzy_matches)
        
        return sorted(set(matches))
    
    def _complete_command_parts(self, command, text: str) -> List[str]:
        """Complete subcommands and options"""
        completions = []
        
        # Add subcommands
        for subcommand_name in command.subcommands.keys():
            if subcommand_name.startswith(text):
                completions.append(subcommand_name)
        
        # Add options
        for option_name in command.parser.options.keys():
            long_option = f"--{option_name}"
            if long_option.startswith(text):
                completions.append(long_option)
        
        for option in command.parser.options.values():
            if option.short_name:
                short_option = f"-{option.short_name}"
                if short_option.startswith(text):
                    completions.append(short_option)
        
        return sorted(completions)
    
    def _complete_arguments(self, command, args: List[str], text: str) -> List[str]:
        """Complete command arguments"""
        completions = []
        
        # Check if we're in a subcommand
        if args and args[0] in command.subcommands:
            subcommand = command.subcommands[args[0]]
            return self._complete_arguments(subcommand, args[1:], text)
        
        # Check if completing an option value
        if len(args) >= 2 and args[-2].startswith('-'):
            option_name = args[-2].lstrip('-')
            if option_name in command.parser.options:
                option = command.parser.options[option_name]
                if hasattr(option, 'choices') and option.choices:
                    completions = [
                        choice for choice in option.choices 
                        if str(choice).startswith(text)
                    ]
        
        # File completion for appropriate arguments
        if self.file_completion_enabled and not completions:
            file_completions = self._complete_files(text)
            completions.extend(file_completions)
        
        # History-based completion
        if self.history_completion_enabled and not completions:
            history_completions = self._complete_from_history(text)
            completions.extend(history_completions)
        
        return sorted(set(completions))
    
    def _complete_files(self, text: str) -> List[str]:
        """Complete file and directory names"""
        try:
            if not text:
                # Complete files in current directory
                path = Path('.')
                pattern = '*'
            elif '/' in text:
                # Complete with directory path
                path = Path(text).parent
                pattern = Path(text).name + '*'
            else:
                # Complete files starting with text
                path = Path('.')
                pattern = text + '*'
            
            if path.exists():
                matches = []
                for item in path.glob(pattern):
                    if item.is_dir():
                        matches.append(str(item) + '/')
                    else:
                        matches.append(str(item))
                return matches[:20]  # Limit to 20 matches
        except Exception:
            pass
        
        return []
    
    def _complete_from_history(self, text: str) -> List[str]:
        """Complete from command history"""
        try:
            history_completions = []
            history_length = readline.get_current_history_length()
            
            for i in range(1, min(history_length + 1, 100)):  # Last 100 commands
                line = readline.get_history_item(i)
                if line and text in line:
                    # Extract relevant part
                    parts = line.split()
                    for part in parts:
                        if part.startswith(text) and part != text:
                            history_completions.append(part)
            
            return list(set(history_completions))[:10]  # Limit to 10
        except:
            return []
    
    def save_history(self):
        """Save command history"""
        try:
            history_file = Path.home() / '.workstation_history'
            readline.write_history_file(str(history_file))
        except:
            pass


class CLIPrompt:
    """Advanced CLI prompt manager"""
    
    def __init__(self, app, theme: CLITheme):
        self.app = app
        self.theme = theme
        self.style = PromptStyle.DETAILED
        self.show_time = True
        self.show_user = True
        self.show_status = True
        self.custom_elements = []
    
    def generate_prompt(self) -> str:
        """Generate prompt string"""
        if self.style == PromptStyle.SIMPLE:
            return self._simple_prompt()
        elif self.style == PromptStyle.DETAILED:
            return self._detailed_prompt()
        elif self.style == PromptStyle.MINIMAL:
            return self._minimal_prompt()
        elif self.style == PromptStyle.POWERLINE:
            return self._powerline_prompt()
        else:
            return self._custom_prompt()
    
    def _simple_prompt(self) -> str:
        """Simple prompt: AppName> """
        app_name = self.theme.colorize(self.app.app_name, self.theme.current_theme.prompt, bold=True)
        return f"{app_name}> "
    
    def _detailed_prompt(self) -> str:
        """Detailed prompt with status and time"""
        elements = []
        
        # Time
        if self.show_time:
            current_time = time.strftime("%H:%M:%S")
            time_element = self.theme.colorize(f"[{current_time}]", self.theme.current_theme.secondary)
            elements.append(time_element)
        
        # App name and status
        status_icon = "✅" if self.app.is_running else "❌"
        app_element = self.theme.colorize(
            f"{status_icon} {self.app.app_name}",
            self.theme.current_theme.prompt,
            bold=True
        )
        elements.append(app_element)
        
        # User info
        if self.show_user:
            user = os.getenv('USER', 'user')
            user_element = self.theme.colorize(f"({user})", self.theme.current_theme.secondary)
            elements.append(user_element)
        
        # Custom elements
        for element in self.custom_elements:
            elements.append(element())
        
        prompt_line = " ".join(elements)
        return f"{prompt_line}\n> "
    
    def _minimal_prompt(self) -> str:
        """Minimal prompt: > """
        return self.theme.colorize("> ", self.theme.current_theme.prompt)
    
    def _powerline_prompt(self) -> str:
        """Powerline-style prompt"""
        # This would implement a powerline-style prompt
        # For now, using detailed style
        return self._detailed_prompt()
    
    def _custom_prompt(self) -> str:
        """Custom prompt with user-defined elements"""
        elements = [element() for element in self.custom_elements]
        prompt_line = " ".join(elements)
        return f"{prompt_line}> "
    
    def add_custom_element(self, element_func: Callable[[], str]):
        """Add custom prompt element"""
        self.custom_elements.append(element_func)


class CLIHistory:
    """Enhanced command history management"""
    
    def __init__(self, max_size: int = 10000):
        self.history = deque(maxlen=max_size)
        self.session_history = deque(maxlen=1000)
        self.history_file = Path.home() / '.workstation_cli_history'
        self.search_mode = False
        self.search_results = []
        self.current_search = ""
        
        self.load_history()
    
    def add_command(self, command: str):
        """Add command to history"""
        if command.strip():
            timestamp = time.time()
            entry = {
                'command': command,
                'timestamp': timestamp,
                'session': True
            }
            
            self.history.append(entry)
            self.session_history.append(entry)
    
    def search_history(self, query: str) -> List[Dict[str, Any]]:
        """Search command history"""
        results = []
        for entry in reversed(list(self.history)):
            if query.lower() in entry['command'].lower():
                results.append(entry)
                if len(results) >= 50:  # Limit results
                    break
        return results
    
    def get_recent(self, count: int = 20) -> List[Dict[str, Any]]:
        """Get recent commands"""
        return list(self.history)[-count:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get history statistics"""
        if not self.history:
            return {'total_commands': 0}
        
        commands = [entry['command'].split()[0] for entry in self.history if entry['command']]
        command_counts = {}
        for cmd in commands:
            command_counts[cmd] = command_counts.get(cmd, 0) + 1
        
        most_used = sorted(command_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_commands': len(self.history),
            'session_commands': len(self.session_history),
            'unique_commands': len(command_counts),
            'most_used': most_used,
            'oldest_entry': self.history[0]['timestamp'] if self.history else None,
            'newest_entry': self.history[-1]['timestamp'] if self.history else None
        }
    
    def load_history(self):
        """Load history from file"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    for entry in data:
                        entry['session'] = False
                        self.history.append(entry)
        except Exception as e:
            print(f"Error loading history: {e}")
    
    def save_history(self):
        """Save history to file"""
        try:
            # Save all non-session history plus recent session history
            to_save = []
            
            # Add existing non-session history
            for entry in self.history:
                if not entry.get('session', False):
                    to_save.append({
                        'command': entry['command'],
                        'timestamp': entry['timestamp']
                    })
            
            # Add recent session history
            recent_session = list(self.session_history)[-100:]  # Last 100 from session
            for entry in recent_session:
                to_save.append({
                    'command': entry['command'],
                    'timestamp': entry['timestamp']
                })
            
            with open(self.history_file, 'w') as f:
                json.dump(to_save, f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")


class CLITools:
    """Collection of CLI utility tools"""
    
    def __init__(self, app):
        self.app = app
        self.terminal_size = self._get_terminal_size()
        self.pager_enabled = True
        self.progress_bars = {}
    
    def _get_terminal_size(self) -> Tuple[int, int]:
        """Get terminal size (width, height)"""
        try:
            size = shutil.get_terminal_size()
            return size.columns, size.lines
        except:
            return 80, 24  # Default size
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_table(self, headers: List[str], rows: List[List[str]], 
                   theme: CLITheme):
        """Print formatted table"""
        if not rows:
            return
        
        # Calculate column widths
        col_widths = []
        for i, header in enumerate(headers):
            max_width = len(header)
            for row in rows:
                if i < len(row):
                    max_width = max(max_width, len(str(row[i])))
            col_widths.append(min(max_width, 30))  # Max 30 chars per column
        
        # Print header
        header_line = " | ".join(
            theme.highlight(header.ljust(col_widths[i]))
            for i, header in enumerate(headers)
        )
        print(header_line)
        print("-" * sum(col_widths) + "-" * (len(headers) - 1) * 3)
        
        # Print rows
        for row in rows:
            row_line = " | ".join(
                str(row[i] if i < len(row) else "").ljust(col_widths[i])
                for i in range(len(headers))
            )
            print(row_line)
    
    def print_with_pager(self, text: str):
        """Print text with pager if it's too long"""
        if not self.pager_enabled:
            print(text)
            return
        
        lines = text.split('\n')
        terminal_height = self.terminal_size[1]
        
        if len(lines) <= terminal_height - 2:
            print(text)
        else:
            # Use pager
            self._page_text(lines, terminal_height)
    
    def _page_text(self, lines: List[str], page_size: int):
        """Page through text"""
        current_line = 0
        
        while current_line < len(lines):
            # Print page
            page_end = min(current_line + page_size - 1, len(lines))
            for i in range(current_line, page_end):
                print(lines[i])
            
            current_line = page_end
            
            # Check if more pages
            if current_line >= len(lines):
                break
            
            # Prompt for next page
            try:
                response = input("\nPress Enter for next page, 'q' to quit: ").strip().lower()
                if response == 'q':
                    break
            except (EOFError, KeyboardInterrupt):
                break
    
    def create_progress_bar(self, task_id: str, total: int, description: str = ""):
        """Create progress bar"""
        self.progress_bars[task_id] = {
            'total': total,
            'current': 0,
            'description': description,
            'start_time': time.time()
        }
    
    def update_progress(self, task_id: str, increment: int = 1):
        """Update progress bar"""
        if task_id not in self.progress_bars:
            return
        
        progress = self.progress_bars[task_id]
        progress['current'] = min(progress['current'] + increment, progress['total'])
        
        # Calculate percentage and ETA
        percentage = (progress['current'] / progress['total']) * 100
        elapsed = time.time() - progress['start_time']
        
        if progress['current'] > 0:
            eta = (elapsed / progress['current']) * (progress['total'] - progress['current'])
            eta_str = f" ETA: {eta:.1f}s"
        else:
            eta_str = ""
        
        # Create progress bar
        bar_width = 30
        filled = int((percentage / 100) * bar_width)
        bar = '█' * filled + '░' * (bar_width - filled)
        
        # Print progress
        print(f"\r{progress['description']} [{bar}] {percentage:.1f}%{eta_str}", end='', flush=True)
        
        if progress['current'] >= progress['total']:
            print()  # New line when complete
            del self.progress_bars[task_id]
    
    def print_banner(self, text: str, theme: CLITheme, width: Optional[int] = None):
        """Print decorative banner"""
        if width is None:
            width = self.terminal_size[0]
        
        border = "=" * width
        padding = (width - len(text) - 2) // 2
        
        print(theme.highlight(border))
        print(theme.highlight(f"|{' ' * padding}{text}{' ' * padding}|"))
        print(theme.highlight(border))
    
    def print_box(self, content: List[str], theme: CLITheme, title: str = ""):
        """Print content in a box"""
        if not content:
            return
        
        # Calculate box width
        max_width = max(len(line) for line in content)
        if title:
            max_width = max(max_width, len(title))
        
        box_width = min(max_width + 4, self.terminal_size[0])
        content_width = box_width - 4
        
        # Top border
        if title:
            title_padding = (box_width - len(title) - 2) // 2
            top_line = f"┌{title_padding * '─'} {title} {title_padding * '─'}┐"
            print(theme.highlight(top_line))
        else:
            print(theme.highlight(f"┌{'─' * (box_width - 2)}┐"))
        
        # Content
        for line in content:
            if len(line) <= content_width:
                padding = content_width - len(line)
                print(f"│ {line}{' ' * padding} │")
            else:
                # Wrap long lines
                wrapped = [line[i:i+content_width] for i in range(0, len(line), content_width)]
                for wrapped_line in wrapped:
                    padding = content_width - len(wrapped_line)
                    print(f"│ {wrapped_line}{' ' * padding} │")
        
        # Bottom border
        print(theme.highlight(f"└{'─' * (box_width - 2)}┘"))


class AdvancedCLI:
    """Advanced CLI with comprehensive features"""
    
    def __init__(self, app):
        self.app = app
        self.theme = CLITheme('default')
        self.prompt = CLIPrompt(app, self.theme)
        self.autocomplete = AutoComplete(app.command_system)
        self.history = CLIHistory()
        self.tools = CLITools(app)
        
        # Configuration
        self.running = False
        self.current_line = ""
        self.cursor_position = 0
        self.input_mode = "normal"  # normal, search, command
        
        # Keyboard handling
        self.key_bindings = {}
        self._setup_key_bindings()
        
        # Status
        self.last_command_time = None
        self.command_count = 0
        
    def _setup_key_bindings(self):
        """Setup keyboard shortcuts"""
        self.key_bindings = {
            '\x03': self._handle_ctrl_c,      # Ctrl+C
            '\x04': self._handle_ctrl_d,      # Ctrl+D
            '\x0c': self._handle_ctrl_l,      # Ctrl+L
            '\x12': self._handle_ctrl_r,      # Ctrl+R (reverse search)
            '\x15': self._handle_ctrl_u,      # Ctrl+U (clear line)
            '\x17': self._handle_ctrl_w,      # Ctrl+W (delete word)
        }
    
    async def run_async(self):
        """Run CLI asynchronously"""
        self.running = True
        
        # Welcome message
        self._print_welcome()
        
        try:
            while self.running:
                try:
                    # Generate and display prompt
                    prompt_text = self.prompt.generate_prompt()
                    
                    # Get user input
                    try:
                        user_input = input(prompt_text).strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\nGoodbye!")
                        break
                    
                    if not user_input:
                        continue
                    
                    # Add to history
                    self.history.add_command(user_input)
                    self.command_count += 1
                    self.last_command_time = time.time()
                    
                    # Process command
                    await self._process_command(user_input)
                    
                except KeyboardInterrupt:
                    print("\nUse 'quit' or 'exit' to exit gracefully.")
                    continue
                except Exception as e:
                    self.theme.error(f"CLI error: {e}")
                    if self.app.logger:
                        await self.app.logger.error_async(f"CLI error: {e}")
        
        finally:
            self._cleanup()
    
    def run(self):
        """Run CLI synchronously"""
        return asyncio.run(self.run_async())
    
    async def _process_command(self, command: str):
        """Process command input"""
        start_time = time.time()
        
        try:
            # Handle built-in CLI commands
            if command.startswith(':'):
                await self._handle_cli_command(command[1:])
                return
            
            # Execute through command system
            if hasattr(self.app, 'command_system'):
                result = await self.app.command_system.execute_async(command)
                
                # Display result
                if result.status.name == 'SUCCESS':
                    if result.message and result.message != "Command executed successfully":
                        print(result.message)
                    if result.data:
                        self._display_result(result.data)
                elif result.status.name == 'NOT_FOUND':
                    print(self.theme.error(result.message))
                    if result.suggestions:
                        print(f"Did you mean: {', '.join(result.suggestions)}?")
                elif result.status.name == 'INVALID_ARGS':
                    print(self.theme.error(f"Invalid arguments: {result.message}"))
                    print("Use 'help <command>' for usage information.")
                else:
                    print(self.theme.error(f"Command failed: {result.message}"))
            else:
                print(self.theme.error("Command system not available"))
        
        except Exception as e:
            print(self.theme.error(f"Error executing command: {e}"))
            if self.app.error_handler:
                self.app.error_handler.handle_error(e, "CLI Command Execution")
        
        finally:
            execution_time = time.time() - start_time
            if execution_time > 1.0:  # Show time for slow commands
                print(self.theme.info(f"Execution time: {execution_time:.2f}s"))
    
    async def _handle_cli_command(self, command: str):
        """Handle CLI-specific commands"""
        parts = command.split()
        if not parts:
            return
        
        cmd = parts[0].lower()
        args = parts[1:]
        
        if cmd == 'theme':
            if args:
                self.theme.set_theme(args[0])
                print(self.theme.success(f"Theme set to: {args[0]}"))
            else:
                themes = list(CLITheme.THEMES.keys())
                print(f"Available themes: {', '.join(themes)}")
                print(f"Current theme: {self.theme.current_theme.name}")
        
        elif cmd == 'history':
            count = int(args[0]) if args and args[0].isdigit() else 20
            history = self.history.get_recent(count)
            
            print(f"\nRecent commands ({len(history)}):")
            for i, entry in enumerate(history, 1):
                timestamp = time.strftime('%H:%M:%S', time.localtime(entry['timestamp']))
                print(f"  {i:2d}. [{timestamp}] {entry['command']}")
        
        elif cmd == 'search':
            if args:
                query = ' '.join(args)
                results = self.history.search_history(query)
                
                print(f"\nHistory search results for '{query}':")
                for entry in results[:20]:  # Limit to 20 results
                    timestamp = time.strftime('%H:%M:%S', time.localtime(entry['timestamp']))
                    print(f"  [{timestamp}] {entry['command']}")
            else:
                print("Usage: :search <query>")
        
        elif cmd == 'clear':
            self.tools.clear_screen()
        
        elif cmd == 'stats':
            stats = self.history.get_statistics()
            cli_stats = {
                'CLI Session Commands': self.command_count,
                'Last Command Time': time.strftime('%H:%M:%S', time.localtime(self.last_command_time)) if self.last_command_time else 'Never'
            }
            
            print("\nCLI Statistics:")
            for key, value in cli_stats.items():
                print(f"  {key}: {value}")
            
            print("\nHistory Statistics:")
            for key, value in stats.items():
                if key == 'most_used':
                    print(f"  {key}:")
                    for cmd, count in value:
                        print(f"    {cmd}: {count}")
                else:
                    print(f"  {key}: {value}")
        
        elif cmd == 'help':
            self._print_cli_help()
        
        else:
            print(self.theme.error(f"Unknown CLI command: {cmd}"))
            print("Use ':help' for available CLI commands")
    
    def _display_result(self, data: Any):
        """Display command result data"""
        if isinstance(data, dict):
            # Display as formatted key-value pairs
            for key, value in data.items():
                print(f"  {key}: {value}")
        elif isinstance(data, (list, tuple)):
            # Display as numbered list
            for i, item in enumerate(data, 1):
                print(f"  {i}. {item}")
        else:
            # Display as string
            print(str(data))
    
    def _print_welcome(self):
        """Print welcome message"""
        welcome_text = f"Welcome to {self.app.app_name} CLI"
        self.tools.print_banner(welcome_text, self.theme)
        
        print(self.theme.info("Type 'help' for available commands"))
        print(self.theme.info("Type ':help' for CLI-specific commands"))
        print(self.theme.info("Use Tab for auto-completion"))
        print()
    
    def _print_cli_help(self):
        """Print CLI help"""
        help_content = [
            "CLI Commands (prefix with ':'):",
            "",
            ":theme [name]     - Change color theme",
            ":history [count]  - Show command history",
            ":search <query>   - Search command history",
            ":clear           - Clear screen",
            ":stats           - Show CLI statistics",
            ":help            - Show this help",
            "",
            "Keyboard Shortcuts:",
            "Ctrl+C           - Interrupt current command",
            "Ctrl+D           - Exit CLI",
            "Ctrl+L           - Clear screen",
            "Ctrl+R           - Reverse history search",
            "Tab              - Auto-complete",
            "↑/↓              - Navigate history"
        ]
        
        self.tools.print_box(help_content, self.theme, "CLI Help")
    
    # Key binding handlers
    def _handle_ctrl_c(self):
        """Handle Ctrl+C"""
        raise KeyboardInterrupt()
    
    def _handle_ctrl_d(self):
        """Handle Ctrl+D"""
        self.running = False
        return False
    
    def _handle_ctrl_l(self):
        """Handle Ctrl+L"""
        self.tools.clear_screen()
        return True
    
    def _handle_ctrl_r(self):
        """Handle Ctrl+R (reverse search)"""
        # This would implement reverse search
        # For now, just return True
        return True
    
    def _handle_ctrl_u(self):
        """Handle Ctrl+U (clear line)"""
        self.current_line = ""
        self.cursor_position = 0
        return True
    
    def _handle_ctrl_w(self):
        """Handle Ctrl+W (delete word)"""
        # This would implement word deletion
        # For now, just return True
        return True
    
    def _cleanup(self):
        """Cleanup CLI resources"""
        # Save history
        self.history.save_history()
        
        # Save auto-completion history
        self.autocomplete.save_history()
        
        print(self.theme.success("\nCLI session ended. History saved."))