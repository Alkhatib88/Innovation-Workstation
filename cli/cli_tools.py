from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from requests import *
from os import *
from importlib import *

#cli_tools.py
class CliTools:
    def __init__(self, app):
        self.app = app
        self.active_commands = {}  # Tracks active commands
        self.extensions = []
        self.session = PromptSession()

    def get_command_completer(self):
        # Use the CustomCompleter class instead
        return CustomCompleter(self)

    def get_subcommands(self, command_name):
        """
        Retrieves a list of subcommands for the given command.
        """
        # Assuming `self.commands` is a dictionary where each command has its subcommands listed
        command_info = self.app.command.commands.get(command_name, {})
        return list(command_info.get('subcommands', {}).keys())

    def get_argument_format(self, command_name, subcommand_name=None):
        """
        Returns the expected argument format for the command or subcommand.
        """
        command_info = self.app.command.commands.get(command_name, {})
        if subcommand_name:
            # Fetching argument format for subcommand
            subcommand_info = command_info.get('subcommands', {}).get(subcommand_name, {})
            return subcommand_info.get('arg_format', '')
        else:
            # Fetching argument format for command
            return command_info.get('arg_format', '')

    def log_command(self, command_name):
        ''' Log the execution of a command. '''
        self.app.logger.info(f"Command executed: {command_name}")

    def check_dependencies(self, command_name):
        command_info = self.app.command.commands.get(command_name, {})
        missing_dependencies = []

        for dep_type, dep_param in command_info.get('dependencies', []):
            if not self.is_dependency_met(dep_type, dep_param):
                missing_dependencies.append((dep_type, dep_param))

        return missing_dependencies if missing_dependencies else None

    def is_dependency_met(self, dep_type, dep_param):
        if dep_type == "external_service":
            return self.check_some_external_service(dep_param)
        elif dep_type == "file_exists":
            return self.check_if_file_exists(dep_param)
        # ... other dependency checks ...

        return False

    def check_some_external_service(self, site):
        # Logic to check if a web service is available
        try:
            response = requests.get(site)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def check_if_file_exists(self, file_path):
        return os.path.exists(file_path)

    def load_extension(self, extension_name):
        ''' Dynamically load an extension or plugin. '''
        try:
            module = importlib.import_module(extension_name)
            self.extensions[extension_name] = module
            print(f"Extension {extension_name} loaded successfully.")
        except ImportError:
            print(f"Failed to load extension {extension_name}.")

    def unload_extension(self, extension_name):
        ''' Unload an already loaded extension. '''
        if extension_name in self.extensions:
            del self.extensions[extension_name]
            print(f"Extension {extension_name} unloaded successfully.")
        else:
            print(f"Extension {extension_name} is not loaded.")

    def setup(self):
        self.app.command.add_command(command_name="load-extension", function=self.load_extension, description="Loads extension", category="Module Management", dependencies = ['extension_name'])
        self.app.command.add_command(command_name="unload-extension", function=self.unload_extension, description="Unloads extension", category="Module Management", dependencies = ['extension_name'])

class CustomCompleter(Completer):
    def __init__(self, cli_tools_instance):
        self.cli_tools = cli_tools_instance

    def is_typing_argument(self, words):
        """Determine if the user is currently typing an argument."""
        if len(words) <= 1:
            return False
        command = words[0].lstrip('/')
        if command not in self.cli_tools.app.command.commands:
            return False
        if len(words) == 2:
            # If there are subcommands, we're not in argument context after the command
            subcommands = self.cli_tools.get_subcommands(command)
            return not bool(subcommands)
        # More than two words means we're past the command and subcommand, so it's an argument
        return True

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        words = text.split()

        if not text.startswith('/'):
            return  # Only provide completions for commands

        if len(words) == 1 and text.endswith(' '):
            # User has completed a main command and is possibly looking for subcommands
            command = words[0].lstrip('/')
            if command in self.cli_tools.app.command.active_commands and not self.cli_tools.app.command.active_commands[command]:
                return  # Do not suggest subcommands for deactivated commands
            subcommands = self.cli_tools.get_subcommands(command)
            for subcommand in subcommands:
                yield Completion(subcommand, start_position=0)

        elif len(words) == 1:
            # When only part of a command is typed, filter and suggest commands
            partial_command = words[-1].lstrip('/')
            for command in self.cli_tools.app.command.commands.keys():
                if command.startswith(partial_command) and self.cli_tools.app.command.active_commands.get(command, True):
                    yield Completion('/' + command, start_position=-len(words[-1]))

        elif len(words) > 1 and not self.is_typing_argument(words):
            # Handle subcommands, user has typed at least one character of subcommand
            command = words[0].lstrip('/')
            if command in self.cli_tools.app.command.active_commands and not self.cli_tools.app.command.active_commands[command]:
                return  # Do not suggest arguments for deactivated commands
            if command in self.cli_tools.app.command.commands:
                subcommands = self.cli_tools.get_subcommands(command)
                partial_subcommand = words[-1]
                for subcommand in subcommands:
                    if subcommand.startswith(partial_subcommand):
                        yield Completion(subcommand, start_position=-len(words[-1]))
