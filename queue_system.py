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

#!/usr/bin/env python3
"""
Advanced File and Directory Management Systems
Comprehensive file operations, monitoring, versioning, and security
"""

import asyncio
import os
import shutil
import hashlib
import mimetypes
import time
import json
import pickle
import gzip
import tarfile
import zipfile
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import stat
import fnmatch
import tempfile
import uuid
from datetime import datetime, timedelta
import watchdog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class FileType(Enum):
    """File type classification"""
    TEXT = auto()
    BINARY = auto()
    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()
    DOCUMENT = auto()
    ARCHIVE = auto()
    EXECUTABLE = auto()
    CONFIG = auto()
    LOG = auto()
    DATABASE = auto()
    TEMPORARY = auto()
    UNKNOWN = auto()


class FileOperation(Enum):
    """File operations"""
    CREATE = auto()
    READ = auto()
    WRITE = auto()
    UPDATE = auto()
    DELETE = auto()
    MOVE = auto()
    COPY = auto()
    RENAME = auto()
    COMPRESS = auto()
    DECOMPRESS = auto()
    ENCRYPT = auto()
    DECRYPT = auto()


class FilePermission(Enum):
    """File permissions"""
    READ = 'r'
    WRITE = 'w'
    EXECUTE = 'x'
    READ_WRITE = 'rw'
    READ_EXECUTE = 'rx'
    WRITE_EXECUTE = 'wx'
    ALL = 'rwx'


@dataclass
class FileMetadata:
    """Enhanced file metadata"""
    path: Path
    size: int = 0
    created_time: float = 0
    modified_time: float = 0
    accessed_time: float = 0
    file_type: FileType = FileType.UNKNOWN
    mime_type: str = ""
    checksum: str = ""
    permissions: str = ""
    owner: str = ""
    group: str = ""
    is_symlink: bool = False
    target_path: Optional[Path] = None
    encoding: Optional[str] = None
    line_count: Optional[int] = None
    version: int = 1
    tags: Set[str] = field(default_factory=set)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileVersion:
    """File version information"""
    version: int
    timestamp: float
    size: int
    checksum: str
    author: str = ""
    comment: str = ""
    backup_path: Optional[Path] = None


class FileIndex:
    """File indexing and search system"""
    
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.index = {}  # path -> FileMetadata
        self.checksum_index = {}  # checksum -> List[path]
        self.type_index = defaultdict(list)  # file_type -> List[path]
        self.tag_index = defaultdict(list)  # tag -> List[path]
        self.size_index = defaultdict(list)  # size_range -> List[path]
        self.lock = threading.Lock()
        
        self.load_index()
    
    def add_file(self, metadata: FileMetadata):
        """Add file to index"""
        with self.lock:
            path_str = str(metadata.path)
            self.index[path_str] = metadata
            
            # Update secondary indexes
            if metadata.checksum:
                if metadata.checksum not in self.checksum_index:
                    self.checksum_index[metadata.checksum] = []
                self.checksum_index[metadata.checksum].append(path_str)
            
            self.type_index[metadata.file_type].append(path_str)
            
            for tag in metadata.tags:
                self.tag_index[tag].append(path_str)
            
            # Size index (categorize by size ranges)
            size_range = self._get_size_range(metadata.size)
            self.size_index[size_range].append(path_str)
    
    def remove_file(self, path: Path):
        """Remove file from index"""
        with self.lock:
            path_str = str(path)
            if path_str not in self.index:
                return
            
            metadata = self.index[path_str]
            
            # Remove from secondary indexes
            if metadata.checksum and metadata.checksum in self.checksum_index:
                if path_str in self.checksum_index[metadata.checksum]:
                    self.checksum_index[metadata.checksum].remove(path_str)
                if not self.checksum_index[metadata.checksum]:
                    del self.checksum_index[metadata.checksum]
            
            if path_str in self.type_index[metadata.file_type]:
                self.type_index[metadata.file_type].remove(path_str)
            
            for tag in metadata.tags:
                if path_str in self.tag_index[tag]:
                    self.tag_index[tag].remove(path_str)
            
            size_range = self._get_size_range(metadata.size)
            if path_str in self.size_index[size_range]:
                self.size_index[size_range].remove(path_str)
            
            del self.index[path_str]
    
    def search_files(self, query: str = "", file_type: FileType = None,
                    tags: List[str] = None, size_range: Tuple[int, int] = None,
                    modified_after: float = None) -> List[FileMetadata]:
        """Search files with various criteria"""
        with self.lock:
            results = []
            
            # Start with all files if no specific criteria
            candidates = set(self.index.keys())
            
            # Filter by file type
            if file_type:
                type_files = set(self.type_index.get(file_type, []))
                candidates &= type_files
            
            # Filter by tags
            if tags:
                for tag in tags:
                    tag_files = set(self.tag_index.get(tag, []))
                    candidates &= tag_files
            
            # Filter by size range
            if size_range:
                min_size, max_size = size_range
                size_files = set()
                for size_cat, files in self.size_index.items():
                    if size_cat[0] <= max_size and size_cat[1] >= min_size:
                        size_files.update(files)
                candidates &= size_files
            
            # Filter by modification time
            if modified_after:
                time_candidates = set()
                for path_str in candidates:
                    metadata = self.index[path_str]
                    if metadata.modified_time >= modified_after:
                        time_candidates.add(path_str)
                candidates = time_candidates
            
            # Text search in filename
            if query:
                query_lower = query.lower()
                text_candidates = set()
                for path_str in candidates:
                    if query_lower in Path(path_str).name.lower():
                        text_candidates.add(path_str)
                candidates = text_candidates
            
            # Convert to metadata objects
            for path_str in candidates:
                results.append(self.index[path_str])
            
            return results
    
    def find_duplicates(self) -> List[List[FileMetadata]]:
        """Find duplicate files by checksum"""
        with self.lock:
            duplicates = []
            for checksum, paths in self.checksum_index.items():
                if len(paths) > 1:
                    duplicate_group = [self.index[path] for path in paths]
                    duplicates.append(duplicate_group)
            return duplicates
    
    def _get_size_range(self, size: int) -> Tuple[int, int]:
        """Get size range category"""
        if size < 1024:  # < 1KB
            return (0, 1023)
        elif size < 1024 * 1024:  # < 1MB
            return (1024, 1024 * 1024 - 1)
        elif size < 1024 * 1024 * 1024:  # < 1GB
            return (1024 * 1024, 1024 * 1024 * 1024 - 1)
        else:  # >= 1GB
            return (1024 * 1024 * 1024, float('inf'))
    
    def save_index(self):
        """Save index to disk"""
        try:
            with self.lock:
                # Convert to serializable format
                index_data = {}
                for path_str, metadata in self.index.items():
                    index_data[path_str] = {
                        'size': metadata.size,
                        'created_time': metadata.created_time,
                        'modified_time': metadata.modified_time,
                        'accessed_time': metadata.accessed_time,
                        'file_type': metadata.file_type.name,
                        'mime_type': metadata.mime_type,
                        'checksum': metadata.checksum,
                        'permissions': metadata.permissions,
                        'owner': metadata.owner,
                        'group': metadata.group,
                        'is_symlink': metadata.is_symlink,
                        'target_path': str(metadata.target_path) if metadata.target_path else None,
                        'encoding': metadata.encoding,
                        'line_count': metadata.line_count,
                        'version': metadata.version,
                        'tags': list(metadata.tags),
                        'custom_metadata': metadata.custom_metadata
                    }
                
                with open(self.index_path, 'w') as f:
                    json.dump(index_data, f, indent=2)
        except Exception as e:
            print(f"Error saving file index: {e}")
    
    def load_index(self):
        """Load index from disk"""
        try:
            if self.index_path.exists():
                with open(self.index_path, 'r') as f:
                    index_data = json.load(f)
                
                with self.lock:
                    for path_str, data in index_data.items():
                        metadata = FileMetadata(
                            path=Path(path_str),
                            size=data['size'],
                            created_time=data['created_time'],
                            modified_time=data['modified_time'],
                            accessed_time=data['accessed_time'],
                            file_type=FileType[data['file_type']],
                            mime_type=data['mime_type'],
                            checksum=data['checksum'],
                            permissions=data['permissions'],
                            owner=data['owner'],
                            group=data['group'],
                            is_symlink=data['is_symlink'],
                            target_path=Path(data['target_path']) if data['target_path'] else None,
                            encoding=data['encoding'],
                            line_count=data['line_count'],
                            version=data['version'],
                            tags=set(data['tags']),
                            custom_metadata=data['custom_metadata']
                        )
                        self.add_file(metadata)
        except Exception as e:
            print(f"Error loading file index: {e}")


class FileWatcher(FileSystemEventHandler):
    """File system event watcher"""
    
    def __init__(self, file_manager):
        super().__init__()
        self.file_manager = file_manager
        self.event_queue = asyncio.Queue()
    
    def on_any_event(self, event):
        """Handle any file system event"""
        try:
            asyncio.create_task(self.event_queue.put({
                'type': event.event_type,
                'path': event.src_path,
                'is_directory': event.is_directory,
                'timestamp': time.time()
            }))
        except:
            pass  # Queue might be full or event loop not running
    
    async def process_events(self):
        """Process file system events"""
        while True:
            try:
                event = await self.event_queue.get()
                await self.file_manager._handle_fs_event(event)
            except Exception as e:
                print(f"Error processing file system event: {e}")


class FileManager:
    """Advanced file management system"""
    
    def __init__(self, app):
        self.app = app
        self.base_path = Path.cwd()
        self.data_dir = self.base_path / "data"
        self.temp_dir = self.base_path / "temp"
        self.backup_dir = self.base_path / "backups"
        self.index_dir = self.base_path / "indexes"
        
        # File operations
        self.operation_history = deque(maxlen=1000)
        self.pending_operations = []
        self.batch_operations = {}
        
        # File monitoring
        self.file_index = None
        self.file_watcher = None
        self.observer = None
        self.watched_paths = set()
        
        # Versioning
        self.versioning_enabled = True
        self.max_versions = 10
        self.version_storage = {}  # file_path -> List[FileVersion]
        
        # Security and validation
        self.allowed_extensions = set()
        self.blocked_extensions = {'.exe', '.bat', '.cmd', '.scr', '.vbs'}
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.virus_scan_enabled = False
        
        # Performance
        self.cache = {}
        self.cache_max_size = 1000
        self.enable_compression = True
        
        # Statistics
        self.stats = {
            'files_created': 0,
            'files_read': 0,
            'files_written': 0,
            'files_deleted': 0,
            'bytes_read': 0,
            'bytes_written': 0,
            'operations_count': 0,
            'errors_count': 0
        }
    
    async def setup(self) -> bool:
        """Setup file manager"""
        try:
            # Create directories
            directories = [
                self.data_dir,
                self.temp_dir,
                self.backup_dir,
                self.index_dir
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
            
            # Initialize file index
            index_path = self.index_dir / "file_index.json"
            self.file_index = FileIndex(index_path)
            
            # Setup file watcher
            self.file_watcher = FileWatcher(self)
            self.observer = Observer()
            
            # Start monitoring base directories
            for directory in directories:
                if directory.exists():
                    self.observer.schedule(self.file_watcher, str(directory), recursive=True)
                    self.watched_paths.add(directory)
            
            self.observer.start()
            
            # Load configuration
            await self._load_configuration()
            
            self.app.logger.info("File manager initialized successfully")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "FileManager.setup")
            return False
    
    async def _load_configuration(self):
        """Load file manager configuration"""
        try:
            config_file = self.app.config.get('file_manager.config_file', 'config/file_manager.json')
            config_path = Path(config_file)
            
            if config_path.exists():
                content = await self.read_text_async(config_path)
                if content:
                    config = json.loads(content)
                    
                    self.max_file_size = config.get('max_file_size', self.max_file_size)
                    self.max_versions = config.get('max_versions', self.max_versions)
                    self.versioning_enabled = config.get('versioning_enabled', self.versioning_enabled)
                    self.allowed_extensions = set(config.get('allowed_extensions', []))
                    self.blocked_extensions.update(config.get('blocked_extensions', []))
                    self.virus_scan_enabled = config.get('virus_scan_enabled', False)
        except Exception as e:
            self.app.logger.warning(f"Could not load file manager configuration: {e}")
    
    # Core file operations
    async def read_text_async(self, file_path: Union[str, Path], encoding: str = 'utf-8') -> Optional[str]:
        """Read text file asynchronously"""
        try:
            path = Path(file_path)
            
            # Security check
            if not await self._validate_file_access(path, FileOperation.READ):
                return None
            
            # Check cache
            cache_key = f"read_text:{path}:{path.stat().st_mtime}"
            if cache_key in self.cache:
                self.stats['files_read'] += 1
                return self.cache[cache_key]
            
            # Read file
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, lambda: path.read_text(encoding=encoding))
            
            # Update cache
            if len(self.cache) < self.cache_max_size:
                self.cache[cache_key] = content
            
            # Update statistics
            self.stats['files_read'] += 1
            self.stats['bytes_read'] += len(content.encode())
            
            # Update file index
            if self.file_index:
                metadata = await self._generate_file_metadata(path)
                self.file_index.add_file(metadata)
            
            # Record operation
            await self._record_operation(FileOperation.READ, path, success=True)
            
            return content
            
        except Exception as e:
            await self._record_operation(FileOperation.READ, Path(file_path), success=False, error=str(e))
            self.app.error_handler.handle_error(e, f"FileManager.read_text_async({file_path})")
            return None
    
    async def write_text_async(self, file_path: Union[str, Path], content: str, 
                              encoding: str = 'utf-8', create_backup: bool = True) -> bool:
        """Write text file asynchronously"""
        try:
            path = Path(file_path)
            
            # Security check
            if not await self._validate_file_access(path, FileOperation.WRITE):
                return False
            
            # Create backup if file exists and versioning is enabled
            if create_backup and self.versioning_enabled and path.exists():
                await self._create_backup(path)
            
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: path.write_text(content, encoding=encoding))
            
            # Update statistics
            self.stats['files_written'] += 1
            self.stats['bytes_written'] += len(content.encode())
            
            if not path.existed_before_write:
                self.stats['files_created'] += 1
            
            # Update file index
            if self.file_index:
                metadata = await self._generate_file_metadata(path)
                self.file_index.add_file(metadata)
            
            # Clear cache
            self._clear_file_cache(path)
            
            # Record operation
            await self._record_operation(FileOperation.WRITE, path, success=True)
            
            return True
            
        except Exception as e:
            await self._record_operation(FileOperation.WRITE, Path(file_path), success=False, error=str(e))
            self.app.error_handler.handle_error(e, f"FileManager.write_text_async({file_path})")
            return False
    
    async def read_binary_async(self, file_path: Union[str, Path]) -> Optional[bytes]:
        """Read binary file asynchronously"""
        try:
            path = Path(file_path)
            
            # Security check
            if not await self._validate_file_access(path, FileOperation.READ):
                return None
            
            # Read file
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, lambda: path.read_bytes())
            
            # Update statistics
            self.stats['files_read'] += 1
            self.stats['bytes_read'] += len(content)
            
            # Record operation
            await self._record_operation(FileOperation.READ, path, success=True)
            
            return content
            
        except Exception as e:
            await self._record_operation(FileOperation.READ, Path(file_path), success=False, error=str(e))
            self.app.error_handler.handle_error(e, f"FileManager.read_binary_async({file_path})")
            return None
    
    async def write_binary_async(self, file_path: Union[str, Path], content: bytes,
                                create_backup: bool = True) -> bool:
        """Write binary file asynchronously"""
        try:
            path = Path(file_path)
            
            # Security check
            if not await self._validate_file_access(path, FileOperation.WRITE):
                return False
            
            # Create backup if file exists and versioning is enabled
            if create_backup and self.versioning_enabled and path.exists():
                await self._create_backup(path)
            
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: path.write_bytes(content))
            
            # Update statistics
            self.stats['files_written'] += 1
            self.stats['bytes_written'] += len(content)
            
            # Update file index
            if self.file_index:
                metadata = await self._generate_file_metadata(path)
                self.file_index.add_file(metadata)
            
            # Record operation
            await self._record_operation(FileOperation.WRITE, path, success=True)
            
            return True
            
        except Exception as e:
            await self._record_operation(FileOperation.WRITE, Path(file_path), success=False, error=str(e))
            self.app.error_handler.handle_error(e, f"FileManager.write_binary_async({file_path})")
            return False
    
    async def delete_async(self, file_path: Union[str, Path], secure_delete: bool = False) -> bool:
        """Delete file asynchronously"""
        try:
            path = Path(file_path)
            
            # Security check
            if not await self._validate_file_access(path, FileOperation.DELETE):
                return False
            
            if not path.exists():
                return True
            
            # Create backup before deletion if versioning is enabled
            if self.versioning_enabled:
                await self._create_backup(path)
            
            # Secure deletion (overwrite with random data)
            if secure_delete and path.is_file():
                await self._secure_delete(path)
            else:
                # Regular deletion
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
            
            # Update statistics
            self.stats['files_deleted'] += 1
            
            # Remove from index
            if self.file_index:
                self.file_index.remove_file(path)
            
            # Clear cache
            self._clear_file_cache(path)
            
            # Record operation
            await self._record_operation(FileOperation.DELETE, path, success=True)
            
            return True
            
        except Exception as e:
            await self._record_operation(FileOperation.DELETE, Path(file_path), success=False, error=str(e))
            self.app.error_handler.handle_error(e, f"FileManager.delete_async({file_path})")
            return False
    
    async def copy_async(self, source: Union[str, Path], destination: Union[str, Path],
                        preserve_metadata: bool = True) -> bool:
        """Copy file or directory asynchronously"""
        try:
            src_path = Path(source)
            dst_path = Path(destination)
            
            # Security checks
            if not await self._validate_file_access(src_path, FileOperation.READ):
                return False
            if not await self._validate_file_access(dst_path, FileOperation.WRITE):
                return False
            
            # Ensure destination directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy operation
            loop = asyncio.get_event_loop()
            if src_path.is_file():
                await loop.run_in_executor(None, lambda: shutil.copy2(src_path, dst_path) if preserve_metadata else shutil.copy(src_path, dst_path))
            elif src_path.is_dir():
                await loop.run_in_executor(None, lambda: shutil.copytree(src_path, dst_path, dirs_exist_ok=True))
            
            # Update file index
            if self.file_index and dst_path.is_file():
                metadata = await self._generate_file_metadata(dst_path)
                self.file_index.add_file(metadata)
            
            # Record operation
            await self._record_operation(FileOperation.COPY, src_path, success=True, extra_info={'destination': str(dst_path)})
            
            return True
            
        except Exception as e:
            await self._record_operation(FileOperation.COPY, Path(source), success=False, error=str(e))
            self.app.error_handler.handle_error(e, f"FileManager.copy_async({source} -> {destination})")
            return False
    
    async def move_async(self, source: Union[str, Path], destination: Union[str, Path]) -> bool:
        """Move file or directory asynchronously"""
        try:
            src_path = Path(source)
            dst_path = Path(destination)
            
            # Security checks
            if not await self._validate_file_access(src_path, FileOperation.READ):
                return False
            if not await self._validate_file_access(dst_path, FileOperation.WRITE):
                return False
            
            # Ensure destination directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move operation
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: shutil.move(str(src_path), str(dst_path)))
            
            # Update file index
            if self.file_index:
                self.file_index.remove_file(src_path)
                if dst_path.is_file():
                    metadata = await self._generate_file_metadata(dst_path)
                    self.file_index.add_file(metadata)
            
            # Clear cache
            self._clear_file_cache(src_path)
            
            # Record operation
            await self._record_operation(FileOperation.MOVE, src_path, success=True, extra_info={'destination': str(dst_path)})
            
            return True
            
        except Exception as e:
            await self._record_operation(FileOperation.MOVE, Path(source), success=False, error=str(e))
            self.app.error_handler.handle_error(e, f"FileManager.move_async({source} -> {destination})")
            return False
    
    # Compression and archiving
    async def compress_file_async(self, file_path: Union[str, Path], 
                                 compression_type: str = 'gzip') -> Optional[Path]:
        """Compress file asynchronously"""
        try:
            path = Path(file_path)
            
            if not path.exists() or not path.is_file():
                return None
            
            # Determine output path
            if compression_type == 'gzip':
                output_path = path.with_suffix(path.suffix + '.gz')
                
                # Compress
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._gzip_compress, path, output_path)
            else:
                raise ValueError(f"Unsupported compression type: {compression_type}")
            
            # Record operation
            await self._record_operation(FileOperation.COMPRESS, path, success=True, 
                                       extra_info={'output': str(output_path), 'type': compression_type})
            
            return output_path
            
        except Exception as e:
            await self._record_operation(FileOperation.COMPRESS, Path(file_path), success=False, error=str(e))
            self.app.error_handler.handle_error(e, f"FileManager.compress_file_async({file_path})")
            return None
    
    def _gzip_compress(self, input_path: Path, output_path: Path):
        """Compress file using gzip"""
        with open(input_path, 'rb') as f_in:
            with gzip.open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    # File validation and security
    async def _validate_file_access(self, path: Path, operation: FileOperation) -> bool:
        """Validate file access for security"""
        try:
            # Check if path is within allowed directories
            resolved_path = path.resolve()
            base_resolved = self.base_path.resolve()
            
            if not str(resolved_path).startswith(str(base_resolved)):
                self.app.logger.warning(f"Access denied: {path} is outside base directory")
                return False
            
            # Check file extension for write operations
            if operation in [FileOperation.WRITE, FileOperation.CREATE]:
                if path.suffix.lower() in self.blocked_extensions:
                    self.app.logger.warning(f"Access denied: {path.suffix} extension is blocked")
                    return False
                
                if self.allowed_extensions and path.suffix.lower() not in self.allowed_extensions:
                    self.app.logger.warning(f"Access denied: {path.suffix} extension is not allowed")
                    return False
            
            # Check file size for write operations
            if operation == FileOperation.WRITE and path.exists():
                if path.stat().st_size > self.max_file_size:
                    self.app.logger.warning(f"Access denied: {path} exceeds maximum file size")
                    return False
            
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"FileManager._validate_file_access({path}, {operation})")
            return False
    
    async def _generate_file_metadata(self, path: Path) -> FileMetadata:
        """Generate comprehensive file metadata"""
        try:
            stat_info = path.stat()
            
            metadata = FileMetadata(
                path=path,
                size=stat_info.st_size,
                created_time=stat_info.st_ctime,
                modified_time=stat_info.st_mtime,
                accessed_time=stat_info.st_atime,
                permissions=stat.filemode(stat_info.st_mode),
                is_symlink=path.is_symlink()
            )
            
            if path.is_symlink():
                metadata.target_path = path.readlink()
            
            # Determine file type
            metadata.file_type = self._classify_file_type(path)
            
            # Get MIME type
            metadata.mime_type = mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
            
            # Calculate checksum for regular files
            if path.is_file() and stat_info.st_size < 10 * 1024 * 1024:  # Only for files < 10MB
                loop = asyncio.get_event_loop()
                metadata.checksum = await loop.run_in_executor(None, self._calculate_checksum, path)
            
            # Get encoding for text files
            if metadata.file_type == FileType.TEXT:
                metadata.encoding = await self._detect_encoding(path)
                
                # Count lines for small text files
                if stat_info.st_size < 1024 * 1024:  # < 1MB
                    metadata.line_count = await self._count_lines(path)
            
            # Get owner information (Unix systems)
            try:
                import pwd
                metadata.owner = pwd.getpwuid(stat_info.st_uid).pw_name
            except (ImportError, KeyError):
                metadata.owner = str(stat_info.st_uid)
            
            try:
                import grp
                metadata.group = grp.getgrgid(stat_info.st_gid).gr_name
            except (ImportError, KeyError):
                metadata.group = str(stat_info.st_gid)
            
            return metadata
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"FileManager._generate_file_metadata({path})")
            # Return basic metadata on error
            return FileMetadata(path=path)
    
    def _classify_file_type(self, path: Path) -> FileType:
        """Classify file type based on extension and content"""
        suffix = path.suffix.lower()
        
        # Text files
        text_extensions = {'.txt', '.md', '.rst', '.log', '.cfg', '.ini', '.json', '.xml', '.yaml', '.yml', '.csv', '.tsv'}
        if suffix in text_extensions:
            return FileType.TEXT
        
        # Configuration files
        config_extensions = {'.conf', '.config', '.ini', '.cfg', '.properties', '.toml'}
        if suffix in config_extensions:
            return FileType.CONFIG
        
        # Images
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg', '.webp'}
        if suffix in image_extensions:
            return FileType.IMAGE
        
        # Videos
        video_extensions = {'.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv'}
        if suffix in video_extensions:
            return FileType.VIDEO
        
        # Audio
        audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}
        if suffix in audio_extensions:
            return FileType.AUDIO
        
        # Documents
        doc_extensions = {'.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.odt', '.ods', '.odp'}
        if suffix in doc_extensions:
            return FileType.DOCUMENT
        
        # Archives
        archive_extensions = {'.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar'}
        if suffix in archive_extensions:
            return FileType.ARCHIVE
        
        # Executables
        exec_extensions = {'.exe', '.bat', '.cmd', '.sh', '.bin', '.app', '.deb', '.rpm'}
        if suffix in exec_extensions:
            return FileType.EXECUTABLE
        
        # Databases
        db_extensions = {'.db', '.sqlite', '.sqlite3', '.mdb', '.accdb'}
        if suffix in db_extensions:
            return FileType.DATABASE
        
        # Temporary files
        temp_extensions = {'.tmp', '.temp', '.bak', '.swp', '.swo'}
        if suffix in temp_extensions or path.name.startswith('.'):
            return FileType.TEMPORARY
        
        # Default to binary for unknown types
        return FileType.BINARY
    
    def _calculate_checksum(self, path: Path, algorithm: str = 'sha256') -> str:
        """Calculate file checksum"""
        hash_obj = hashlib.new(algorithm)
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    async def _detect_encoding(self, path: Path) -> Optional[str]:
        """Detect text file encoding"""
        try:
            # Try common encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
            
            for encoding in encodings:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        f.read(1024)  # Read first 1KB
                    return encoding
                except UnicodeDecodeError:
                    continue
            
            return None
        except Exception:
            return None
    
    async def _count_lines(self, path: Path) -> Optional[int]:
        """Count lines in text file"""
        try:
            loop = asyncio.get_event_loop()
            count = await loop.run_in_executor(None, self._count_lines_sync, path)
            return count
        except Exception:
            return None
    
    def _count_lines_sync(self, path: Path) -> int:
        """Count lines synchronously"""
        with open(path, 'rb') as f:
            count = sum(1 for _ in f)
        return count
    
    # Backup and versioning
    async def _create_backup(self, path: Path):
        """Create backup of file"""
        try:
            if not path.exists():
                return
            
            # Generate backup path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{path.name}.{timestamp}.backup"
            backup_path = self.backup_dir / path.parent.relative_to(self.base_path) / backup_name
            
            # Ensure backup directory exists
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file to backup location
            shutil.copy2(path, backup_path)
            
            # Update version storage
            file_str = str(path)
            if file_str not in self.version_storage:
                self.version_storage[file_str] = []
            
            version = FileVersion(
                version=len(self.version_storage[file_str]) + 1,
                timestamp=time.time(),
                size=path.stat().st_size,
                checksum=self._calculate_checksum(path),
                backup_path=backup_path
            )
            
            self.version_storage[file_str].append(version)
            
            # Cleanup old versions
            if len(self.version_storage[file_str]) > self.max_versions:
                old_version = self.version_storage[file_str].pop(0)
                if old_version.backup_path and old_version.backup_path.exists():
                    old_version.backup_path.unlink()
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"FileManager._create_backup({path})")
    
    async def _secure_delete(self, path: Path):
        """Securely delete file by overwriting with random data"""
        try:
            if not path.is_file():
                return
            
            file_size = path.stat().st_size
            
            # Overwrite with random data multiple times
            for _ in range(3):
                with open(path, 'wb') as f:
                    while f.tell() < file_size:
                        chunk_size = min(8192, file_size - f.tell())
                        random_data = os.urandom(chunk_size)
                        f.write(random_data)
                f.flush()
                os.fsync(f.fileno())
            
            # Finally delete the file
            path.unlink()
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"FileManager._secure_delete({path})")
    
    # Event handling
    async def _handle_fs_event(self, event: Dict[str, Any]):
        """Handle file system event"""
        try:
            path = Path(event['path'])
            event_type = event['type']
            
            if event['is_directory']:
                return  # Skip directory events for now
            
            # Update file index based on event type
            if self.file_index:
                if event_type in ['created', 'modified']:
                    if path.exists():
                        metadata = await self._generate_file_metadata(path)
                        self.file_index.add_file(metadata)
                elif event_type == 'deleted':
                    self.file_index.remove_file(path)
            
            # Clear cache for affected file
            self._clear_file_cache(path)
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"FileManager._handle_fs_event({event})")
    
    # Utility methods
    def _clear_file_cache(self, path: Path):
        """Clear cache entries for a file"""
        path_str = str(path)
        keys_to_remove = [key for key in self.cache.keys() if path_str in key]
        for key in keys_to_remove:
            del self.cache[key]
    
    async def _record_operation(self, operation: FileOperation, path: Path, 
                               success: bool, error: str = None, extra_info: Dict = None):
        """Record file operation for auditing"""
        record = {
            'timestamp': time.time(),
            'operation': operation.name,
            'path': str(path),
            'success': success,
            'error': error,
            'extra_info': extra_info or {}
        }
        
        self.operation_history.append(record)
        self.stats['operations_count'] += 1
        
        if not success:
            self.stats['errors_count'] += 1
    
    # Public API methods
    def get_file_info(self, file_path: Union[str, Path]) -> Optional[FileMetadata]:
        """Get file information from index"""
        if self.file_index:
            path_str = str(Path(file_path))
            return self.file_index.index.get(path_str)
        return None
    
    def search_files(self, **criteria) -> List[FileMetadata]:
        """Search files using various criteria"""
        if self.file_index:
            return self.file_index.search_files(**criteria)
        return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get file manager statistics"""
        stats = self.stats.copy()
        
        if self.file_index:
            stats['indexed_files'] = len(self.file_index.index)
            stats['unique_checksums'] = len(self.file_index.checksum_index)
        
        stats['watched_paths'] = len(self.watched_paths)
        stats['cached_files'] = len(self.cache)
        stats['operation_history_size'] = len(self.operation_history)
        
        return stats
    
    async def cleanup(self):
        """Cleanup file manager resources"""
        try:
            # Stop file watcher
            if self.observer:
                self.observer.stop()
                self.observer.join()
            
            # Save file index
            if self.file_index:
                self.file_index.save_index()
            
            # Clear cache
            self.cache.clear()
            
            self.app.logger.info("File manager cleanup completed")
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "FileManager.cleanup")


class DirectoryManager:
    """Advanced directory management system"""
    
    def __init__(self, app):
        self.app = app
        self.base_path = Path.cwd()
        self.monitored_directories = {}
        self.directory_stats = {}
        self.sync_tasks = {}
        
    async def setup(self) -> bool:
        """Setup directory manager"""
        try:
            # Initialize monitoring for key directories
            key_dirs = ['data', 'logs', 'config', 'temp', 'backups']
            
            for dir_name in key_dirs:
                dir_path = self.base_path / dir_name
                if dir_path.exists():
                    await self.monitor_directory(dir_path)
            
            self.app.logger.info("Directory manager initialized successfully")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "DirectoryManager.setup")
            return False
    
    async def monitor_directory(self, path: Path, recursive: bool = True):
        """Start monitoring a directory"""
        try:
            path_str = str(path)
            
            if path_str in self.monitored_directories:
                return  # Already monitoring
            
            # Create monitoring entry
            self.monitored_directories[path_str] = {
                'path': path,
                'recursive': recursive,
                'start_time': time.time(),
                'file_count': 0,
                'total_size': 0,
                'last_scan': None
            }
            
            # Perform initial scan
            await self._scan_directory(path, recursive)
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"DirectoryManager.monitor_directory({path})")
    
    async def _scan_directory(self, path: Path, recursive: bool = True):
        """Scan directory and update statistics"""
        try:
            file_count = 0
            total_size = 0
            
            if recursive:
                for item in path.rglob('*'):
                    if item.is_file():
                        file_count += 1
                        total_size += item.stat().st_size
            else:
                for item in path.iterdir():
                    if item.is_file():
                        file_count += 1
                        total_size += item.stat().st_size
            
            # Update monitoring data
            path_str = str(path)
            if path_str in self.monitored_directories:
                self.monitored_directories[path_str].update({
                    'file_count': file_count,
                    'total_size': total_size,
                    'last_scan': time.time()
                })
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"DirectoryManager._scan_directory({path})")
    
    def get_directory_stats(self, path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Get directory statistics"""
        path_str = str(Path(path))
        return self.monitored_directories.get(path_str)
    
    def get_all_monitored_directories(self) -> Dict[str, Dict[str, Any]]:
        """Get all monitored directories"""
        return self.monitored_directories.copy()
    
    async def cleanup(self):
        """Cleanup directory manager resources"""
        try:
            # Stop any running sync tasks
            for task in self.sync_tasks.values():
                task.cancel()
            
            if self.sync_tasks:
                await asyncio.gather(*self.sync_tasks.values(), return_exceptions=True)
            
            self.app.logger.info("Directory manager cleanup completed")
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "DirectoryManager.cleanup")

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

#!/usr/bin/env python3
"""
Innovation Workstation - Advanced Enterprise Application Framework
A comprehensive digital management platform with enterprise-grade features
"""

import os
import sys
import signal
import atexit
import time
import asyncio
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json

# Import advanced modules
from advanced_logger import AdvancedLogger, LoggerFactory
from advanced_error_handler import ErrorHandler, ErrorSeverity, ErrorCategory
from advanced_file_manager import FileManager, DirectoryManager
from advanced_config_manager import ConfigManager, SettingsManager, EnvironmentManager
from advanced_command_system import CommandSystem, SubCommandRegistry
from event_system import EventSystem, EventBus, EventScheduler
from advanced_cli import AdvancedCLI, CLITools, AutoComplete, ColorTheme
from encryption_system import EncryptionManager, SecurityManager
from system_info import SystemInfoManager, PerformanceMonitor
from time_system import TimeManager, SchedulerManager
from queue_system import QueueManager, MessageBroker
from database_manager import PostgreSQLManager, DatabasePool
from thread_system import ThreadManager, ProcessManager
from asyncio_system import AsyncIOManager, AsyncTaskManager
from worker_system import WorkerManager, TaskManager, TaskScheduler
from pool_system import PoolManager, ResourcePool
from base_manager import BaseManager, ServiceManager
from storage_system import StorageManager, CacheManager
from resource_manager import ResourceManager, MemoryManager
from ipc_system import IPCManager, CommunicationBridge
from load_balancer import LoadBalancer, RequestDistributor


class InitPhase(Enum):
    """Comprehensive initialization phases"""
    APP_START = auto()
    LOGGER_CONSOLE = auto()
    ERROR_SYSTEMS = auto()
    FILE_SYSTEMS = auto()
    ENCRYPTION_SECURITY = auto()
    CONFIG_SYSTEMS = auto()
    LOGGER_UPGRADE = auto()
    SYSTEM_INFO_TIME = auto()
    DATABASE_SYSTEMS = auto()
    EVENT_SYSTEMS = auto()
    QUEUE_MESSAGE_SYSTEMS = auto()
    THREAD_PROCESS_SYSTEMS = auto()
    ASYNCIO_SYSTEMS = auto()
    WORKER_TASK_SYSTEMS = auto()
    POOL_RESOURCE_SYSTEMS = auto()
    STORAGE_CACHE_SYSTEMS = auto()
    IPC_COMMUNICATION = auto()
    LOAD_BALANCING = auto()
    COMMAND_SYSTEMS = auto()
    CLI_SYSTEMS = auto()
    BASE_MANAGEMENT = auto()
    SERVICE_REGISTRATION = auto()
    ADVANCED_FEATURES = auto()


@dataclass
class ComponentStatus:
    """Enhanced component status tracking"""
    name: str
    initialized: bool = False
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    initialization_time: Optional[float] = None
    health_status: str = "unknown"
    metrics: Dict[str, Any] = field(default_factory=dict)
    last_health_check: Optional[float] = None


class InnovationWorkstation:
    """
    Advanced Enterprise Innovation Workstation
    A comprehensive platform for digital management, automation, and development
    """
    
    def __init__(self, app_name: str = "Innovation Workstation Enterprise"):
        # Core application properties
        self.app_name = app_name
        self.version = "2.0.0"
        self.build = "enterprise"
        self.is_running = False
        self.is_initialized = False
        self.startup_time = None
        self.current_phase = InitPhase.APP_START
        
        # Advanced configuration
        self.mode = "production"  # development, testing, production
        self.debug_enabled = False
        self.performance_monitoring = True
        self.auto_recovery = True
        
        # Component tracking
        self.components = {}
        self.services = {}
        self.health_status = "initializing"
        
        # Core Systems - Layer 1
        self.logger = None
        self.error_handler = None
        self.file_manager = None
        self.dir_manager = None
        self.encryption = None
        self.security = None
        
        # Configuration Systems - Layer 2
        self.config = None
        self.settings = None
        self.env = None
        
        # Information Systems - Layer 3
        self.system_info = None
        self.performance_monitor = None
        self.time_manager = None
        self.scheduler = None
        
        # Database Systems - Layer 4
        self.database = None
        self.db_pool = None
        
        # Event & Communication Systems - Layer 5
        self.event_system = None
        self.event_bus = None
        self.event_scheduler = None
        self.queue_manager = None
        self.message_broker = None
        
        # Execution Systems - Layer 6
        self.thread_manager = None
        self.process_manager = None
        self.asyncio_manager = None
        self.async_task_manager = None
        
        # Worker & Task Systems - Layer 7
        self.worker_manager = None
        self.task_manager = None
        self.task_scheduler = None
        
        # Resource Systems - Layer 8
        self.pool_manager = None
        self.resource_pool = None
        self.storage_manager = None
        self.cache_manager = None
        self.resource_manager = None
        self.memory_manager = None
        
        # Communication Systems - Layer 9
        self.ipc_manager = None
        self.communication_bridge = None
        self.load_balancer = None
        self.request_distributor = None
        
        # Interface Systems - Layer 10
        self.command_system = None
        self.subcommand_registry = None
        self.cli = None
        self.cli_tools = None
        self.autocomplete = None
        self.color_theme = None
        
        # Management Systems - Layer 11
        self.base_manager = None
        self.service_manager = None
        
        # Advanced event loop
        self.event_loop = None
        self.background_tasks = set()
        
        # Setup signal handlers and cleanup
        self._setup_signal_handlers()
        atexit.register(self.cleanup)
    
    def _setup_signal_handlers(self):
        """Setup comprehensive signal handlers"""
        def signal_handler(signum, frame):
            signal_name = {
                2: "SIGINT",
                15: "SIGTERM",
                9: "SIGKILL"
            }.get(signum, f"Signal-{signum}")
            
            print(f"\n🛑 Received {signal_name}, initiating graceful shutdown...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
    
    def _register_component(self, name: str, dependencies: List[str] = None):
        """Register component with enhanced tracking"""
        self.components[name] = ComponentStatus(
            name=name,
            dependencies=dependencies or [],
            initialization_time=time.time()
        )
    
    def _mark_component_initialized(self, name: str, success: bool = True, 
                                  error: str = None, metrics: Dict = None):
        """Mark component initialization status"""
        if name in self.components:
            component = self.components[name]
            component.initialized = success
            component.error = error
            component.health_status = "healthy" if success else "error"
            component.last_health_check = time.time()
            if metrics:
                component.metrics.update(metrics)
            
            if component.initialization_time:
                init_duration = time.time() - component.initialization_time
                component.metrics['initialization_duration'] = init_duration
    
    async def start_and_initialize_async(self) -> bool:
        """Asynchronous initialization for maximum performance"""
        try:
            self.startup_time = time.time()
            print(f"🚀 Starting {self.app_name} v{self.version} ({self.build})")
            print("=" * 80)
            
            # Get event loop
            self.event_loop = asyncio.get_event_loop()
            
            # Initialize in phases
            phases = [
                (InitPhase.APP_START, self._phase_1_app_start),
                (InitPhase.LOGGER_CONSOLE, self._phase_2_logger_console),
                (InitPhase.ERROR_SYSTEMS, self._phase_3_error_systems),
                (InitPhase.FILE_SYSTEMS, self._phase_4_file_systems),
                (InitPhase.ENCRYPTION_SECURITY, self._phase_5_encryption_security),
                (InitPhase.CONFIG_SYSTEMS, self._phase_6_config_systems),
                (InitPhase.LOGGER_UPGRADE, self._phase_7_logger_upgrade),
                (InitPhase.SYSTEM_INFO_TIME, self._phase_8_system_info_time),
                (InitPhase.DATABASE_SYSTEMS, self._phase_9_database_systems),
                (InitPhase.EVENT_SYSTEMS, self._phase_10_event_systems),
                (InitPhase.QUEUE_MESSAGE_SYSTEMS, self._phase_11_queue_message_systems),
                (InitPhase.THREAD_PROCESS_SYSTEMS, self._phase_12_thread_process_systems),
                (InitPhase.ASYNCIO_SYSTEMS, self._phase_13_asyncio_systems),
                (InitPhase.WORKER_TASK_SYSTEMS, self._phase_14_worker_task_systems),
                (InitPhase.POOL_RESOURCE_SYSTEMS, self._phase_15_pool_resource_systems),
                (InitPhase.STORAGE_CACHE_SYSTEMS, self._phase_16_storage_cache_systems),
                (InitPhase.IPC_COMMUNICATION, self._phase_17_ipc_communication),
                (InitPhase.LOAD_BALANCING, self._phase_18_load_balancing),
                (InitPhase.COMMAND_SYSTEMS, self._phase_19_command_systems),
                (InitPhase.CLI_SYSTEMS, self._phase_20_cli_systems),
                (InitPhase.BASE_MANAGEMENT, self._phase_21_base_management),
                (InitPhase.SERVICE_REGISTRATION, self._phase_22_service_registration),
                (InitPhase.ADVANCED_FEATURES, self._phase_23_advanced_features)
            ]
            
            success = True
            for phase, handler in phases:
                self.current_phase = phase
                phase_start = time.time()
                
                if asyncio.iscoroutinefunction(handler):
                    result = await handler()
                else:
                    result = handler()
                
                if not result:
                    success = False
                    break
                
                phase_duration = time.time() - phase_start
                print(f"   ⏱️  Phase completed in {phase_duration:.3f}s")
            
            if success:
                self.is_initialized = True
                self.is_running = True
                self.health_status = "operational"
                
                startup_duration = time.time() - self.startup_time
                component_count = len([c for c in self.components.values() if c.initialized])
                total_components = len(self.components)
                
                print("=" * 80)
                print(f"✅ {self.app_name} initialized successfully!")
                print(f"⏱️  Total startup time: {startup_duration:.3f} seconds")
                print(f"📊 Components: {component_count}/{total_components} initialized")
                print(f"🏥 Health Status: {self.health_status}")
                print(f"🔧 Mode: {self.mode}")
                print("=" * 80)
                
                if self.logger:
                    await self.logger.info_async(f"{self.app_name} startup completed in {startup_duration:.3f}s")
                
                # Start health monitoring
                if self.performance_monitoring:
                    asyncio.create_task(self._health_monitor_loop())
                
                return True
            else:
                self.health_status = "failed"
                print(f"❌ {self.app_name} initialization failed at phase {self.current_phase.name}")
                return False
                
        except Exception as e:
            self.health_status = "error"
            print(f"💥 Critical error during initialization: {e}")
            traceback.print_exc()
            return False
    
    def start_and_initialize(self) -> bool:
        """Synchronous wrapper for initialization"""
        return asyncio.run(self.start_and_initialize_async())
    
    # Phase implementations (showing structure - full implementations in individual modules)
    def _phase_1_app_start(self) -> bool:
        """Phase 1: Application startup and core setup"""
        try:
            print("🔄 Phase 1: Application startup and core setup...")
            
            self._register_component('app_core')
            
            # Set application mode based on environment
            if os.getenv('WORKSTATION_MODE'):
                self.mode = os.getenv('WORKSTATION_MODE')
            
            if os.getenv('WORKSTATION_DEBUG'):
                self.debug_enabled = True
            
            print(f"  ✓ Application mode: {self.mode}")
            print(f"  ✓ Debug mode: {'enabled' if self.debug_enabled else 'disabled'}")
            print("  ✓ Signal handlers configured")
            print("  ✓ Component tracking initialized")
            print("  ✓ Cleanup handlers registered")
            
            self._mark_component_initialized('app_core', True)
            print("✅ Phase 1: Core setup complete")
            return True
            
        except Exception as e:
            print(f"❌ Phase 1 failed: {e}")
            return False
    
    def _phase_2_logger_console(self) -> bool:
        """Phase 2: Advanced logger initialization"""
        try:
            print("🔄 Phase 2: Advanced logging system initialization...")
            
            self._register_component('logger')
            
            # Initialize advanced logger factory
            logger_factory = LoggerFactory()
            self.logger = logger_factory.create_logger(
                name=f"{self.app_name.replace(' ', '')}Logger",
                level="INFO",
                console_output=True,
                file_output=False,  # Will be enabled in upgrade phase
                structured_logging=True,
                async_logging=True
            )
            
            print("  ✓ Advanced logger factory created")
            print("  ✓ Structured logging enabled")
            print("  ✓ Async logging configured")
            print("  ✓ Console output active")
            
            self.logger.info("Advanced logging system initialized")
            
            self._mark_component_initialized('logger', True, metrics={
                'log_level': 'INFO',
                'async_enabled': True,
                'structured': True
            })
            
            print("✅ Phase 2: Advanced logging active")
            return True
            
        except Exception as e:
            print(f"❌ Phase 2 failed: {e}")
            return False
    
    def _phase_3_error_systems(self) -> bool:
        """Phase 3: Comprehensive error handling systems"""
        try:
            print("🔄 Phase 3: Advanced error handling and monitoring...")
            
            self._register_component('error_handler', ['logger'])
            
            self.error_handler = ErrorHandler(
                app=self,
                logger=self.logger,
                auto_recovery=self.auto_recovery,
                notification_enabled=True,
                metrics_enabled=True
            )
            
            print("  ✓ Advanced error handler initialized")
            print("  ✓ Error categorization enabled")
            print("  ✓ Auto-recovery mechanisms active")
            print("  ✓ Error metrics collection enabled")
            print("  ✓ Notification system configured")
            
            self.logger.info("Advanced error handling system initialized")
            
            self._mark_component_initialized('error_handler', True, metrics={
                'auto_recovery': self.auto_recovery,
                'categories_supported': len(ErrorCategory),
                'severity_levels': len(ErrorSeverity)
            })
            
            print("✅ Phase 3: Error systems operational")
            return True
            
        except Exception as e:
            print(f"❌ Phase 3 failed: {e}")
            return False
    
    # ... Continue with all other phases
    # (I'll implement the key phases and provide structure for others)
    
    def _phase_4_file_systems(self) -> bool:
        """Phase 4: Advanced file and directory management"""
        try:
            print("🔄 Phase 4: Advanced file and directory systems...")
            
            self._register_component('file_manager', ['logger', 'error_handler'])
            self._register_component('dir_manager', ['logger', 'error_handler'])
            
            self.file_manager = FileManager(self)
            self.dir_manager = DirectoryManager(self)
            
            # Setup with advanced features
            if self.file_manager.setup() and self.dir_manager.setup():
                print("  ✓ Advanced file manager initialized")
                print("  ✓ Directory manager with monitoring")
                print("  ✓ File versioning system enabled")
                print("  ✓ Automated backup configured")
                print("  ✓ File integrity checking active")
                
                self._mark_component_initialized('file_manager', True)
                self._mark_component_initialized('dir_manager', True)
            else:
                return False
            
            print("✅ Phase 4: File systems operational")
            return True
            
        except Exception as e:
            self.error_handler.handle_error(e, "Phase 4", ErrorSeverity.HIGH)
            return False
    
    # ... (Additional phases would be implemented similarly)
    
    async def run_async(self):
        """Advanced asynchronous main run method"""
        try:
            # Initialize everything
            if not await self.start_and_initialize_async():
                print("❌ Failed to initialize application")
                return False
            
            print(f"\n🎯 {self.app_name} is fully operational!")
            print("Advanced enterprise features enabled.")
            print("Type 'help' for available commands, 'quit' to exit.")
            print("-" * 60)
            
            # Start background services
            background_tasks = []
            
            if self.performance_monitoring:
                background_tasks.append(
                    asyncio.create_task(self._performance_monitor_loop())
                )
            
            if self.event_scheduler:
                background_tasks.append(
                    asyncio.create_task(self.event_scheduler.run_async())
                )
            
            # Main CLI loop
            cli_task = asyncio.create_task(self._cli_loop_async())
            background_tasks.append(cli_task)
            
            # Wait for completion
            await asyncio.gather(*background_tasks, return_exceptions=True)
            
            return True
            
        except Exception as e:
            if self.logger:
                await self.logger.error_async(f"Application run error: {e}")
            else:
                print(f"Application run error: {e}")
            return False
        finally:
            await self.shutdown_async()
    
    def run(self):
        """Synchronous wrapper for main run method"""
        return asyncio.run(self.run_async())
    
    async def _cli_loop_async(self):
        """Asynchronous CLI loop"""
        while self.is_running:
            try:
                # Use advanced CLI with async support
                if self.cli:
                    await self.cli.run_async()
                else:
                    # Fallback to basic input
                    user_input = input(f"{self.app_name}> ").strip()
                    if not user_input:
                        continue
                    
                    if self.command_system:
                        continue_running = await self.command_system.execute_async(user_input)
                        if not continue_running:
                            break
                    elif user_input.lower() in ['quit', 'exit']:
                        break
                        
            except (EOFError, KeyboardInterrupt):
                print("\nReceived interrupt signal. Shutting down...")
                break
            except Exception as e:
                if self.error_handler:
                    self.error_handler.handle_error(e, "CLI Loop", ErrorSeverity.MEDIUM)
                else:
                    print(f"CLI error: {e}")
    
    async def _health_monitor_loop(self):
        """Continuous health monitoring"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check component health
                unhealthy_components = []
                for name, component in self.components.items():
                    if component.initialized and component.health_status != "healthy":
                        unhealthy_components.append(name)
                
                if unhealthy_components:
                    await self.logger.warning_async(
                        f"Unhealthy components detected: {', '.join(unhealthy_components)}"
                    )
                
                # Update overall health status
                if len(unhealthy_components) == 0:
                    self.health_status = "excellent"
                elif len(unhealthy_components) < len(self.components) * 0.1:
                    self.health_status = "good"
                elif len(unhealthy_components) < len(self.components) * 0.3:
                    self.health_status = "fair"
                else:
                    self.health_status = "poor"
                
            except Exception as e:
                if self.error_handler:
                    self.error_handler.handle_error(e, "Health Monitor", ErrorSeverity.LOW)
    
    async def _performance_monitor_loop(self):
        """Performance monitoring loop"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                if self.performance_monitor:
                    metrics = await self.performance_monitor.get_metrics_async()
                    
                    # Log performance metrics
                    await self.logger.info_async(
                        "Performance metrics",
                        extra={
                            'cpu_usage': metrics.get('cpu_usage'),
                            'memory_usage': metrics.get('memory_usage'),
                            'disk_usage': metrics.get('disk_usage'),
                            'active_threads': metrics.get('active_threads'),
                            'queue_sizes': metrics.get('queue_sizes')
                        }
                    )
                
            except Exception as e:
                if self.error_handler:
                    self.error_handler.handle_error(e, "Performance Monitor", ErrorSeverity.LOW)
    
    def show_advanced_status(self):
        """Show comprehensive application status"""
        print("\n" + "=" * 90)
        print(f"📊 {self.app_name} v{self.version} ({self.build}) - Advanced Status")
        print("=" * 90)
        
        # Basic status
        print(f"🏃 Running: {'✅ Yes' if self.is_running else '❌ No'}")
        print(f"🔧 Initialized: {'✅ Yes' if self.is_initialized else '❌ No'}")
        print(f"🏥 Health: {self.health_status.upper()}")
        print(f"🎯 Mode: {self.mode}")
        print(f"🐛 Debug: {'Enabled' if self.debug_enabled else 'Disabled'}")
        
        if self.startup_time:
            uptime = time.time() - self.startup_time
            hours, remainder = divmod(int(uptime), 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"⏰ Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Component status by layer
        layers = {
            "Core Systems": ['logger', 'error_handler', 'file_manager', 'dir_manager', 'encryption', 'security'],
            "Configuration": ['config', 'settings', 'env'],
            "Information": ['system_info', 'performance_monitor', 'time_manager', 'scheduler'],
            "Database": ['database', 'db_pool'],
            "Events & Queues": ['event_system', 'event_bus', 'queue_manager', 'message_broker'],
            "Execution": ['thread_manager', 'process_manager', 'asyncio_manager'],
            "Workers & Tasks": ['worker_manager', 'task_manager', 'task_scheduler'],
            "Resources": ['pool_manager', 'storage_manager', 'cache_manager', 'resource_manager'],
            "Communication": ['ipc_manager', 'load_balancer'],
            "Interface": ['command_system', 'cli', 'cli_tools'],
            "Management": ['base_manager', 'service_manager']
        }
        
        for layer_name, component_names in layers.items():
            print(f"\n🏗️  {layer_name}:")
            for name in component_names:
                if name in self.components:
                    component = self.components[name]
                    status_icon = "✅" if component.initialized else "❌"
                    health_icon = {
                        "healthy": "💚", "good": "💛", "fair": "🧡", 
                        "poor": "❤️", "error": "💔", "unknown": "❓"
                    }.get(component.health_status, "❓")
                    
                    metrics_info = ""
                    if component.metrics:
                        key_metrics = list(component.metrics.keys())[:2]
                        if key_metrics:
                            metrics_info = f" [{', '.join(key_metrics)}]"
                    
                    print(f"    {status_icon} {health_icon} {name.replace('_', ' ').title()}{metrics_info}")
                else:
                    print(f"    ⚪ ❓ {name.replace('_', ' ').title()} (not registered)")
        
        print("=" * 90)
    
    async def shutdown_async(self):
        """Advanced asynchronous shutdown"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.health_status = "shutting_down"
        
        print(f"\n🔄 Shutting down {self.app_name}...")
        
        if self.logger:
            await self.logger.info_async(f"Initiating shutdown of {self.app_name}")
        
        try:
            # Shutdown in reverse order of initialization
            shutdown_tasks = []
            
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            # Shutdown components
            if self.service_manager:
                shutdown_tasks.append(self.service_manager.shutdown_all_async())
            
            if self.load_balancer:
                shutdown_tasks.append(self.load_balancer.shutdown_async())
            
            if self.ipc_manager:
                shutdown_tasks.append(self.ipc_manager.shutdown_async())
            
            if self.worker_manager:
                shutdown_tasks.append(self.worker_manager.shutdown_async())
            
            if self.task_manager:
                shutdown_tasks.append(self.task_manager.shutdown_async())
            
            if self.thread_manager:
                shutdown_tasks.append(self.thread_manager.shutdown_async())
            
            if self.process_manager:
                shutdown_tasks.append(self.process_manager.shutdown_async())
            
            if self.database:
                shutdown_tasks.append(self.database.close_async())
            
            # Execute shutdown tasks
            if shutdown_tasks:
                await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            
            # Final cleanup
            if self.config and hasattr(self.config, 'save_async'):
                await self.config.save_async()
            
            if self.settings and hasattr(self.settings, 'save_async'):
                await self.settings.save_async()
            
            if self.logger:
                await self.logger.info_async("Shutdown completed")
                await self.logger.cleanup_async()
        
        except Exception as e:
            print(f"⚠️  Error during shutdown: {e}")
        
        if self.startup_time:
            total_runtime = time.time() - self.startup_time
            hours, remainder = divmod(int(total_runtime), 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"✅ {self.app_name} shutdown complete. Runtime: {hours:02d}:{minutes:02d}:{seconds:02d}")
        else:
            print(f"✅ {self.app_name} shutdown complete")
    
    def shutdown(self):
        """Synchronous wrapper for shutdown"""
        if self.event_loop and self.event_loop.is_running():
            asyncio.create_task(self.shutdown_async())
        else:
            asyncio.run(self.shutdown_async())
    
    def cleanup(self):
        """Final cleanup (called by atexit)"""
        if self.is_running:
            self.shutdown()


async def main_async():
    """Advanced asynchronous main entry point"""
    app = InnovationWorkstation("Innovation Workstation Enterprise")
    
    try:
        success = await app.run_async()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        return 1


def main():
    """Synchronous main entry point"""
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Advanced Encryption and Security System
Comprehensive cryptographic operations, security management, and access control
"""

import asyncio
import os
import secrets
import hashlib
import hmac
import base64
import time
import json
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import jwt
import bcrypt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
import ssl
import socket


class EncryptionType(Enum):
    """Encryption algorithm types"""
    AES_256_GCM = auto()
    AES_256_CBC = auto()
    CHACHA20_POLY1305 = auto()
    FERNET = auto()
    RSA_2048 = auto()
    RSA_4096 = auto()


class HashAlgorithm(Enum):
    """Hash algorithm types"""
    SHA256 = 'sha256'
    SHA512 = 'sha512'
    SHA3_256 = 'sha3_256'
    SHA3_512 = 'sha3_512'
    BLAKE2B = 'blake2b'
    BLAKE2S = 'blake2s'


class SecurityLevel(Enum):
    """Security levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    MAXIMUM = 4


@dataclass
class CryptoKey:
    """Cryptographic key information"""
    key_id: str
    key_type: EncryptionType
    key_data: bytes
    created_at: float
    expires_at: Optional[float] = None
    usage_count: int = 0
    max_usage: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    name: str
    min_password_length: int = 12
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digits: bool = True
    require_special_chars: bool = True
    max_login_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes
    session_timeout: int = 3600  # 1 hour
    require_2fa: bool = False
    allowed_ip_ranges: List[str] = field(default_factory=list)
    blocked_ip_ranges: List[str] = field(default_factory=list)
    encryption_level: SecurityLevel = SecurityLevel.HIGH


@dataclass
class SecurityEvent:
    """Security event record"""
    event_id: str
    event_type: str
    timestamp: float
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    severity: str = "INFO"
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class KeyDerivation:
    """Key derivation functions"""
    
    @staticmethod
    def pbkdf2(password: bytes, salt: bytes, iterations: int = 100000, 
               key_length: int = 32, hash_algorithm=hashes.SHA256()) -> bytes:
        """Derive key using PBKDF2"""
        kdf = PBKDF2HMAC(
            algorithm=hash_algorithm,
            length=key_length,
            salt=salt,
            iterations=iterations,
            backend=default_backend()
        )
        return kdf.derive(password)
    
    @staticmethod
    def scrypt(password: bytes, salt: bytes, n: int = 2**14, r: int = 8, 
               p: int = 1, key_length: int = 32) -> bytes:
        """Derive key using Scrypt"""
        kdf = Scrypt(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=salt,
            n=n,
            r=r,
            p=p,
            backend=default_backend()
        )
        return kdf.derive(password)


class SymmetricEncryption:
    """Symmetric encryption operations"""
    
    def __init__(self):
        self.backend = default_backend()
    
    def encrypt_aes_gcm(self, plaintext: bytes, key: bytes, 
                       associated_data: bytes = None) -> Tuple[bytes, bytes, bytes]:
        """Encrypt using AES-256-GCM"""
        iv = os.urandom(12)  # 96-bit IV for GCM
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        
        if associated_data:
            encryptor.authenticate_additional_data(associated_data)
        
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        return ciphertext, iv, encryptor.tag
    
    def decrypt_aes_gcm(self, ciphertext: bytes, key: bytes, iv: bytes, 
                       tag: bytes, associated_data: bytes = None) -> bytes:
        """Decrypt using AES-256-GCM"""
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv, tag),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        
        if associated_data:
            decryptor.authenticate_additional_data(associated_data)
        
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    def encrypt_chacha20_poly1305(self, plaintext: bytes, key: bytes, 
                                 associated_data: bytes = None) -> Tuple[bytes, bytes]:
        """Encrypt using ChaCha20-Poly1305"""
        nonce = os.urandom(12)
        cipher = Cipher(
            algorithms.ChaCha20(key, nonce),
            modes.Poly1305(),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        
        if associated_data:
            encryptor.authenticate_additional_data(associated_data)
        
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        return ciphertext + encryptor.tag, nonce
    
    def decrypt_chacha20_poly1305(self, ciphertext_with_tag: bytes, key: bytes, 
                                 nonce: bytes, associated_data: bytes = None) -> bytes:
        """Decrypt using ChaCha20-Poly1305"""
        ciphertext = ciphertext_with_tag[:-16]
        tag = ciphertext_with_tag[-16:]
        
        cipher = Cipher(
            algorithms.ChaCha20(key, nonce),
            modes.Poly1305(tag),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        
        if associated_data:
            decryptor.authenticate_additional_data(associated_data)
        
        return decryptor.update(ciphertext) + decryptor.finalize()


class AsymmetricEncryption:
    """Asymmetric encryption operations"""
    
    def __init__(self):
        self.backend = default_backend()
    
    def generate_rsa_keypair(self, key_size: int = 2048) -> Tuple[bytes, bytes]:
        """Generate RSA key pair"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=self.backend
        )
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def encrypt_rsa(self, plaintext: bytes, public_key_pem: bytes) -> bytes:
        """Encrypt using RSA public key"""
        public_key = serialization.load_pem_public_key(
            public_key_pem, backend=self.backend
        )
        
        ciphertext = public_key.encrypt(
            plaintext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return ciphertext
    
    def decrypt_rsa(self, ciphertext: bytes, private_key_pem: bytes) -> bytes:
        """Decrypt using RSA private key"""
        private_key = serialization.load_pem_private_key(
            private_key_pem, password=None, backend=self.backend
        )
        
        plaintext = private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return plaintext
    
    def sign_rsa(self, message: bytes, private_key_pem: bytes) -> bytes:
        """Sign message using RSA private key"""
        private_key = serialization.load_pem_private_key(
            private_key_pem, password=None, backend=self.backend
        )
        
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature
    
    def verify_rsa(self, message: bytes, signature: bytes, public_key_pem: bytes) -> bool:
        """Verify RSA signature"""
        try:
            public_key = serialization.load_pem_public_key(
                public_key_pem, backend=self.backend
            )
            
            public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False


class HashManager:
    """Hash and HMAC operations"""
    
    @staticmethod
    def hash_data(data: bytes, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
        """Hash data using specified algorithm"""
        if algorithm == HashAlgorithm.SHA256:
            return hashlib.sha256(data).hexdigest()
        elif algorithm == HashAlgorithm.SHA512:
            return hashlib.sha512(data).hexdigest()
        elif algorithm == HashAlgorithm.SHA3_256:
            return hashlib.sha3_256(data).hexdigest()
        elif algorithm == HashAlgorithm.SHA3_512:
            return hashlib.sha3_512(data).hexdigest()
        elif algorithm == HashAlgorithm.BLAKE2B:
            return hashlib.blake2b(data).hexdigest()
        elif algorithm == HashAlgorithm.BLAKE2S:
            return hashlib.blake2s(data).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    @staticmethod
    def hmac_sign(data: bytes, key: bytes, algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
        """Create HMAC signature"""
        if algorithm == HashAlgorithm.SHA256:
            return hmac.new(key, data, hashlib.sha256).hexdigest()
        elif algorithm == HashAlgorithm.SHA512:
            return hmac.new(key, data, hashlib.sha512).hexdigest()
        else:
            raise ValueError(f"Unsupported HMAC algorithm: {algorithm}")
    
    @staticmethod
    def verify_hmac(data: bytes, signature: str, key: bytes, 
                   algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> bool:
        """Verify HMAC signature"""
        try:
            expected = HashManager.hmac_sign(data, key, algorithm)
            return hmac.compare_digest(signature, expected)
        except Exception:
            return False
    
    @staticmethod
    def hash_password(password: str, rounds: int = 12) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt(rounds=rounds)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception:
            return False


class SecureStorage:
    """Secure storage for sensitive data"""
    
    def __init__(self, storage_path: Path, master_key: bytes):
        self.storage_path = storage_path
        self.master_key = master_key
        self.fernet = Fernet(base64.urlsafe_b64encode(master_key[:32].ljust(32)[:32]))
        self.storage = {}
        self.lock = threading.Lock()
        
        self.load_storage()
    
    def store(self, key: str, data: Union[str, bytes, Dict, List]) -> bool:
        """Store encrypted data"""
        try:
            with self.lock:
                # Serialize data
                if isinstance(data, (dict, list)):
                    serialized = json.dumps(data).encode('utf-8')
                elif isinstance(data, str):
                    serialized = data.encode('utf-8')
                else:
                    serialized = data
                
                # Encrypt data
                encrypted = self.fernet.encrypt(serialized)
                
                # Store with metadata
                self.storage[key] = {
                    'data': base64.b64encode(encrypted).decode('utf-8'),
                    'timestamp': time.time(),
                    'type': type(data).__name__
                }
                
                self.save_storage()
                return True
        except Exception:
            return False
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve and decrypt data"""
        try:
            with self.lock:
                if key not in self.storage:
                    return None
                
                entry = self.storage[key]
                encrypted = base64.b64decode(entry['data'])
                
                # Decrypt data
                decrypted = self.fernet.decrypt(encrypted)
                
                # Deserialize based on original type
                if entry['type'] in ['dict', 'list']:
                    return json.loads(decrypted.decode('utf-8'))
                elif entry['type'] == 'str':
                    return decrypted.decode('utf-8')
                else:
                    return decrypted
        except Exception:
            return None
    
    def delete(self, key: str) -> bool:
        """Delete stored data"""
        try:
            with self.lock:
                if key in self.storage:
                    del self.storage[key]
                    self.save_storage()
                    return True
                return False
        except Exception:
            return False
    
    def list_keys(self) -> List[str]:
        """List all storage keys"""
        with self.lock:
            return list(self.storage.keys())
    
    def save_storage(self):
        """Save storage to disk"""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(self.storage, f, indent=2)
        except Exception:
            pass
    
    def load_storage(self):
        """Load storage from disk"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    self.storage = json.load(f)
        except Exception:
            self.storage = {}


class TokenManager:
    """JWT token management"""
    
    def __init__(self, secret_key: bytes, issuer: str = "InnovationWorkstation"):
        self.secret_key = secret_key
        self.issuer = issuer
        self.algorithm = 'HS256'
    
    def generate_token(self, payload: Dict[str, Any], expires_in: int = 3600) -> str:
        """Generate JWT token"""
        now = time.time()
        token_payload = {
            'iss': self.issuer,
            'iat': now,
            'exp': now + expires_in,
            **payload
        }
        
        return jwt.encode(token_payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token, self.secret_key, 
                algorithms=[self.algorithm],
                options={'verify_exp': True}
            )
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def refresh_token(self, token: str, expires_in: int = 3600) -> Optional[str]:
        """Refresh JWT token"""
        payload = self.verify_token(token)
        if payload:
            # Remove standard claims
            for claim in ['iss', 'iat', 'exp']:
                payload.pop(claim, None)
            
            return self.generate_token(payload, expires_in)
        return None


class SecurityAudit:
    """Security auditing and monitoring"""
    
    def __init__(self, max_events: int = 10000):
        self.events = deque(maxlen=max_events)
        self.failed_attempts = defaultdict(list)  # ip -> List[timestamp]
        self.blocked_ips = set()
        self.lock = threading.Lock()
    
    def log_event(self, event_type: str, severity: str = "INFO", 
                  source_ip: str = None, user_id: str = None, 
                  description: str = "", **details):
        """Log security event"""
        with self.lock:
            event = SecurityEvent(
                event_id=secrets.token_hex(16),
                event_type=event_type,
                timestamp=time.time(),
                source_ip=source_ip,
                user_id=user_id,
                severity=severity,
                description=description,
                details=details
            )
            
            self.events.append(event)
            
            # Track failed login attempts
            if event_type == "login_failed" and source_ip:
                self.failed_attempts[source_ip].append(time.time())
                
                # Clean old attempts (older than 1 hour)
                cutoff = time.time() - 3600
                self.failed_attempts[source_ip] = [
                    t for t in self.failed_attempts[source_ip] if t > cutoff
                ]
                
                # Check if IP should be blocked
                if len(self.failed_attempts[source_ip]) >= 5:
                    self.blocked_ips.add(source_ip)
                    self.log_event("ip_blocked", "WARNING", source_ip=source_ip,
                                 description=f"IP blocked due to {len(self.failed_attempts[source_ip])} failed attempts")
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked"""
        with self.lock:
            return ip in self.blocked_ips
    
    def unblock_ip(self, ip: str):
        """Unblock IP address"""
        with self.lock:
            self.blocked_ips.discard(ip)
            if ip in self.failed_attempts:
                del self.failed_attempts[ip]
    
    def get_recent_events(self, count: int = 100, 
                         event_type: str = None, severity: str = None) -> List[SecurityEvent]:
        """Get recent security events"""
        with self.lock:
            events = list(self.events)
            
            # Filter by event type
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            # Filter by severity
            if severity:
                events = [e for e in events if e.severity == severity]
            
            # Return most recent
            return events[-count:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get security statistics"""
        with self.lock:
            event_types = defaultdict(int)
            severities = defaultdict(int)
            
            for event in self.events:
                event_types[event.event_type] += 1
                severities[event.severity] += 1
            
            return {
                'total_events': len(self.events),
                'blocked_ips': len(self.blocked_ips),
                'failed_attempts_tracking': len(self.failed_attempts),
                'event_types': dict(event_types),
                'severities': dict(severities)
            }


class EncryptionManager:
    """Main encryption manager"""
    
    def __init__(self, app):
        self.app = app
        self.symmetric_crypto = SymmetricEncryption()
        self.asymmetric_crypto = AsymmetricEncryption()
        self.hash_manager = HashManager()
        
        # Key management
        self.keys = {}  # key_id -> CryptoKey
        self.master_key = None
        self.secure_storage = None
        self.token_manager = None
        
        # Configuration
        self.default_encryption_type = EncryptionType.AES_256_GCM
        self.key_rotation_interval = 86400 * 30  # 30 days
        self.max_key_usage = 1000000
        
        self.lock = threading.Lock()
    
    async def setup(self) -> bool:
        """Setup encryption manager"""
        try:
            # Generate or load master key
            await self._initialize_master_key()
            
            # Initialize secure storage
            storage_path = self.app.file_manager.data_dir / "secure_storage.json"
            self.secure_storage = SecureStorage(storage_path, self.master_key)
            
            # Initialize token manager
            self.token_manager = TokenManager(self.master_key)
            
            # Generate default encryption keys
            await self._generate_default_keys()
            
            self.app.logger.info("Encryption manager initialized successfully")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "EncryptionManager.setup")
            return False
    
    async def _initialize_master_key(self):
        """Initialize master encryption key"""
        key_file = self.app.file_manager.data_dir / ".master_key"
        
        if key_file.exists():
            # Load existing key
            key_data = await self.app.file_manager.read_binary_async(key_file)
            if key_data and len(key_data) == 32:
                self.master_key = key_data
                return
        
        # Generate new master key
        self.master_key = secrets.token_bytes(32)
        
        # Save securely
        await self.app.file_manager.write_binary_async(key_file, self.master_key)
        
        # Set restrictive permissions
        try:
            os.chmod(key_file, 0o600)
        except:
            pass  # Windows doesn't support chmod
        
        self.app.logger.info("New master key generated and saved")
    
    async def _generate_default_keys(self):
        """Generate default encryption keys"""
        # AES-256 key for general encryption
        aes_key = secrets.token_bytes(32)
        await self.add_key("default_aes", EncryptionType.AES_256_GCM, aes_key)
        
        # RSA key pair for asymmetric operations
        private_key, public_key = self.asymmetric_crypto.generate_rsa_keypair(2048)
        await self.add_key("default_rsa_private", EncryptionType.RSA_2048, private_key)
        await self.add_key("default_rsa_public", EncryptionType.RSA_2048, public_key)
    
    async def add_key(self, key_id: str, key_type: EncryptionType, 
                     key_data: bytes, expires_in: int = None) -> bool:
        """Add encryption key"""
        try:
            with self.lock:
                expires_at = time.time() + expires_in if expires_in else None
                
                crypto_key = CryptoKey(
                    key_id=key_id,
                    key_type=key_type,
                    key_data=key_data,
                    created_at=time.time(),
                    expires_at=expires_at,
                    max_usage=self.max_key_usage
                )
                
                self.keys[key_id] = crypto_key
                
                # Store in secure storage
                await self._store_key_securely(crypto_key)
                
                return True
        except Exception as e:
            self.app.error_handler.handle_error(e, f"EncryptionManager.add_key({key_id})")
            return False
    
    async def _store_key_securely(self, crypto_key: CryptoKey):
        """Store key in secure storage"""
        key_info = {
            'key_type': crypto_key.key_type.name,
            'key_data': base64.b64encode(crypto_key.key_data).decode('utf-8'),
            'created_at': crypto_key.created_at,
            'expires_at': crypto_key.expires_at,
            'usage_count': crypto_key.usage_count,
            'max_usage': crypto_key.max_usage,
            'metadata': crypto_key.metadata
        }
        
        self.secure_storage.store(f"key_{crypto_key.key_id}", key_info)
    
    async def encrypt_data(self, data: Union[str, bytes], key_id: str = "default_aes",
                          associated_data: bytes = None) -> Optional[Dict[str, str]]:
        """Encrypt data using specified key"""
        try:
            if key_id not in self.keys:
                return None
            
            crypto_key = self.keys[key_id]
            
            # Check key expiration and usage
            if not await self._validate_key_usage(crypto_key):
                return None
            
            # Convert string to bytes
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Encrypt based on key type
            if crypto_key.key_type == EncryptionType.AES_256_GCM:
                ciphertext, iv, tag = self.symmetric_crypto.encrypt_aes_gcm(
                    data, crypto_key.key_data, associated_data
                )
                
                return {
                    'algorithm': 'AES_256_GCM',
                    'ciphertext': base64.b64encode(ciphertext).decode('utf-8'),
                    'iv': base64.b64encode(iv).decode('utf-8'),
                    'tag': base64.b64encode(tag).decode('utf-8'),
                    'key_id': key_id
                }
            
            elif crypto_key.key_type == EncryptionType.CHACHA20_POLY1305:
                ciphertext_with_tag, nonce = self.symmetric_crypto.encrypt_chacha20_poly1305(
                    data, crypto_key.key_data, associated_data
                )
                
                return {
                    'algorithm': 'CHACHA20_POLY1305',
                    'ciphertext': base64.b64encode(ciphertext_with_tag).decode('utf-8'),
                    'nonce': base64.b64encode(nonce).decode('utf-8'),
                    'key_id': key_id
                }
            
            # Update key usage
            crypto_key.usage_count += 1
            await self._store_key_securely(crypto_key)
            
        except Exception as e:
            self.app.error_handler.handle_error(e, f"EncryptionManager.encrypt_data({key_id})")
            return None
    
    async def decrypt_data(self, encrypted_data: Dict[str, str], 
                          associated_data: bytes = None) -> Optional[bytes]:
        """Decrypt data"""
        try:
            key_id = encrypted_data.get('key_id')
            if not key_id or key_id not in self.keys:
                return None
            
            crypto_key = self.keys[key_id]
            algorithm = encrypted_data.get('algorithm')
            
            if algorithm == 'AES_256_GCM':
                ciphertext = base64.b64decode(encrypted_data['ciphertext'])
                iv = base64.b64decode(encrypted_data['iv'])
                tag = base64.b64decode(encrypted_data['tag'])
                
                return self.symmetric_crypto.decrypt_aes_gcm(
                    ciphertext, crypto_key.key_data, iv, tag, associated_data
                )
            
            elif algorithm == 'CHACHA20_POLY1305':
                ciphertext_with_tag = base64.b64decode(encrypted_data['ciphertext'])
                nonce = base64.b64decode(encrypted_data['nonce'])
                
                return self.symmetric_crypto.decrypt_chacha20_poly1305(
                    ciphertext_with_tag, crypto_key.key_data, nonce, associated_data
                )
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "EncryptionManager.decrypt_data")
            return None
    
    async def _validate_key_usage(self, crypto_key: CryptoKey) -> bool:
        """Validate key usage limits"""
        current_time = time.time()
        
        # Check expiration
        if crypto_key.expires_at and current_time > crypto_key.expires_at:
            return False
        
        # Check usage limit
        if crypto_key.max_usage and crypto_key.usage_count >= crypto_key.max_usage:
            return False
        
        return True
    
    def hash_data(self, data: Union[str, bytes], 
                 algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
        """Hash data"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return self.hash_manager.hash_data(data, algorithm)
    
    def hash_password(self, password: str) -> str:
        """Hash password securely"""
        return self.hash_manager.hash_password(password)
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password"""
        return self.hash_manager.verify_password(password, hashed)
    
    def generate_token(self, payload: Dict[str, Any], expires_in: int = 3600) -> str:
        """Generate authentication token"""
        return self.token_manager.generate_token(payload, expires_in)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify authentication token"""
        return self.token_manager.verify_token(token)
    
    def get_key_info(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get key information"""
        if key_id in self.keys:
            key = self.keys[key_id]
            return {
                'key_id': key.key_id,
                'key_type': key.key_type.name,
                'created_at': key.created_at,
                'expires_at': key.expires_at,
                'usage_count': key.usage_count,
                'max_usage': key.max_usage,
                'is_expired': key.expires_at and time.time() > key.expires_at if key.expires_at else False
            }
        return None
    
    def list_keys(self) -> List[str]:
        """List all key IDs"""
        return list(self.keys.keys())


class SecurityManager:
    """Main security manager"""
    
    def __init__(self, app):
        self.app = app
        self.encryption_manager = None
        self.security_audit = SecurityAudit()
        self.security_policies = {}
        self.active_sessions = {}  # session_id -> session_info
        self.rate_limiters = defaultdict(deque)  # endpoint -> request_times
        
        # Default security policy
        self.default_policy = SecurityPolicy("default")
        self.security_policies["default"] = self.default_policy
    
    async def setup(self) -> bool:
        """Setup security manager"""
        try:
            # Initialize encryption manager
            self.encryption_manager = EncryptionManager(self.app)
            if not await self.encryption_manager.setup():
                return False
            
            # Load security policies
            await self._load_security_policies()
            
            self.app.logger.info("Security manager initialized successfully")
            return True
            
        except Exception as e:
            self.app.error_handler.handle_error(e, "SecurityManager.setup")
            return False
    
    async def _load_security_policies(self):
        """Load security policies from configuration"""
        try:
            policies_file = self.app.file_manager.config_dir / "security_policies.json"
            
            if policies_file.exists():
                content = await self.app.file_manager.read_text_async(policies_file)
                if content:
                    policies_data = json.loads(content)
                    
                    for name, policy_data in policies_data.items():
                        policy = SecurityPolicy(
                            name=name,
                            min_password_length=policy_data.get('min_password_length', 12),
                            require_uppercase=policy_data.get('require_uppercase', True),
                            require_lowercase=policy_data.get('require_lowercase', True),
                            require_digits=policy_data.get('require_digits', True),
                            require_special_chars=policy_data.get('require_special_chars', True),
                            max_login_attempts=policy_data.get('max_login_attempts', 5),
                            lockout_duration=policy_data.get('lockout_duration', 900),
                            session_timeout=policy_data.get('session_timeout', 3600),
                            require_2fa=policy_data.get('require_2fa', False),
                            allowed_ip_ranges=policy_data.get('allowed_ip_ranges', []),
                            blocked_ip_ranges=policy_data.get('blocked_ip_ranges', []),
                            encryption_level=SecurityLevel[policy_data.get('encryption_level', 'HIGH')]
                        )
                        
                        self.security_policies[name] = policy
        except Exception as e:
            self.app.logger.warning(f"Could not load security policies: {e}")
    
    def validate_password(self, password: str, policy_name: str = "default") -> Tuple[bool, List[str]]:
        """Validate password against security policy"""
        policy = self.security_policies.get(policy_name, self.default_policy)
        errors = []
        
        if len(password) < policy.min_password_length:
            errors.append(f"Password must be at least {policy.min_password_length} characters long")
        
        if policy.require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if policy.require_lowercase and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if policy.require_digits and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        if policy.require_special_chars and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        return len(errors) == 0, errors
    
    def create_session(self, user_id: str, ip_address: str = None, 
                      policy_name: str = "default") -> str:
        """Create authenticated session"""
        policy = self.security_policies.get(policy_name, self.default_policy)
        session_id = secrets.token_urlsafe(32)
        
        session_info = {
            'session_id': session_id,
            'user_id': user_id,
            'created_at': time.time(),
            'last_activity': time.time(),
            'ip_address': ip_address,
            'expires_at': time.time() + policy.session_timeout,
            'policy': policy_name
        }
        
        self.active_sessions[session_id] = session_info
        
        # Log session creation
        self.security_audit.log_event(
            "session_created", "INFO",
            source_ip=ip_address, user_id=user_id,
            description=f"Session created for user {user_id}",
            session_id=session_id
        )
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate session"""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        current_time = time.time()
        
        # Check expiration
        if current_time > session['expires_at']:
            del self.active_sessions[session_id]
            
            self.security_audit.log_event(
                "session_expired", "INFO",
                user_id=session['user_id'],
                description=f"Session expired for user {session['user_id']}",
                session_id=session_id
            )
            
            return None
        
        # Update last activity
        session['last_activity'] = current_time
        
        return session
    
    def invalidate_session(self, session_id: str):
        """Invalidate session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            del self.active_sessions[session_id]
            
            self.security_audit.log_event(
                "session_invalidated", "INFO",
                user_id=session['user_id'],
                description=f"Session invalidated for user {session['user_id']}",
                session_id=session_id
            )
    
    def check_rate_limit(self, identifier: str, max_requests: int = 100, 
                        window_seconds: int = 3600) -> bool:
        """Check rate limiting"""
        current_time = time.time()
        requests = self.rate_limiters[identifier]
        
        # Remove old requests outside the window
        while requests and current_time - requests[0] > window_seconds:
            requests.popleft()
        
        # Check if limit exceeded
        if len(requests) >= max_requests:
            return False
        
        # Add current request
        requests.append(current_time)
        return True
    
    def is_ip_allowed(self, ip_address: str, policy_name: str = "default") -> bool:
        """Check if IP address is allowed"""
        if self.security_audit.is_ip_blocked(ip_address):
            return False
        
        policy = self.security_policies.get(policy_name, self.default_policy)
        
        # Check blocked ranges
        for blocked_range in policy.blocked_ip_ranges:
            if self._ip_in_range(ip_address, blocked_range):
                return False
        
        # Check allowed ranges (if specified)
        if policy.allowed_ip_ranges:
            for allowed_range in policy.allowed_ip_ranges:
                if self._ip_in_range(ip_address, allowed_range):
                    return True
            return False  # Not in any allowed range
        
        return True  # No restrictions
    
    def _ip_in_range(self, ip: str, ip_range: str) -> bool:
        """Check if IP is in range (simplified implementation)"""
        # This is a simplified implementation
        # In production, use proper IP address libraries
        if '/' in ip_range:
            # CIDR notation
            network, prefix = ip_range.split('/')
            # Simplified check - in production use ipaddress module
            return ip.startswith(network.rsplit('.', 1)[0])
        else:
            # Single IP or wildcard
            return ip == ip_range or ip_range == '*'
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics"""
        return {
            'encryption_manager': {
                'keys_count': len(self.encryption_manager.keys) if self.encryption_manager else 0,
                'secure_storage_keys': len(self.encryption_manager.secure_storage.list_keys()) if self.encryption_manager and self.encryption_manager.secure_storage else 0
            },
            'sessions': {
                'active_sessions': len(self.active_sessions),
                'rate_limiters': len(self.rate_limiters)
            },
            'audit': self.security_audit.get_statistics(),
            'policies': len(self.security_policies)
        }

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