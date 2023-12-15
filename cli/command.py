# command.py

import argparse
import time
from tabulate import tabulate

class Command:
    def __init__(self, app):
        self.app = app
        self.commands = {}
        self.categories = {}
        self.active_commands = {}  # Tracks active commands
        

    def add_command(self, command_name, function, description=None, help=None, category=None, dependencies=None, arg_definitions=None):
        """Register a new command with argument parsing."""
        self.app.logger.info(f'command added: {command_name}')
        parser = argparse.ArgumentParser(description=description, add_help=False)
        if arg_definitions:
            for arg_name, arg_params in arg_definitions.items():
                parser.add_argument(arg_name, **arg_params)

        self.commands[command_name] = {
            'function': function,
            'parser': parser,
            'description': description,
            'help': help,
            'category': category,
            'dependencies': dependencies or [],
            'active': True
        }

        if category:
            self.categories.setdefault(category, []).append(command_name)

    def add_subcommand(self, main_command, subcommand_name, function, description=None, help=None, dependencies=None, arg_definitions=None):
        """Add a subcommand to an existing command."""
        if main_command not in self.commands:
            self.app.logger.error(f"Main command '{main_command}' not found.")
            return

        main_command_info = self.commands[main_command]
        if 'subcommands' not in main_command_info:
            main_command_info['subcommands'] = {}
            main_command_info['subparsers'] = main_command_info['parser'].add_subparsers(dest='subcommand')

        self.app.logger.info(f'Subcommand added: {main_command} -> {subcommand_name}')

        sub_parser = main_command_info['subparsers'].add_parser(subcommand_name, description=description, help=help)
        if arg_definitions:
            for arg_name, arg_params in arg_definitions.items():
                sub_parser.add_argument(arg_name, **arg_params)

        main_command_info['subcommands'][subcommand_name] = {
            'function': function,
            'parser': sub_parser,
            'dependencies': dependencies or []
        }


    def execute(self, command_name, *args):
        if command_name not in self.commands:
            raise ValueError(f"Command '{command_name}' not found.")

        if not self.active_commands.get(command_name, True):
            print(f"Command '{command_name}' is currently deactivated.")
            return

        command_info = self.commands[command_name]

        # Parse arguments
        try:
            parsed_args = command_info['parser'].parse_args(args)
        except SystemExit:
            return  # Exit if there is an error in parsing arguments or help is called

        # Check for subcommands
        if 'subcommands' in command_info and hasattr(parsed_args, 'subcommand'):
            subcommand = parsed_args.subcommand
            if subcommand and subcommand in command_info['subcommands']:
                subcommand_info = command_info['subcommands'][subcommand]
                subcommand_args = vars(parsed_args)  # Convert Namespace to dict
                del subcommand_args['subcommand']  # Remove the 'subcommand' entry
                return subcommand_info['function'](**subcommand_args)  # Execute subcommand
            else:
                print(f"Invalid subcommand: {subcommand}")
                return
        else:
            # Execute main command if there are no subcommands or subcommand is not provided
            return command_info['function'](**vars(parsed_args))

        execution_time = time.time() - start_time
        self.app.logger.info(f"Executed command: {command_name} in {execution_time:.2f} seconds")

        return result

    def display_commands_table(self):
        """Generate and return the commands table as a string."""
        output = ""
        for category, commands in self.categories.items():
            output += f"Category: {category}\n"
            table_data = []
            for cmd_name in commands:
                cmd_info = self.commands[cmd_name]
                description = cmd_info.get('description', 'No description')
                requirements = ', '.join(cmd_info.get('dependencies', [])) or 'None'
                subcommands_list = ', '.join(cmd_info['subcommands'].keys()) if 'subcommands' in cmd_info else 'None'

                table_data.append([cmd_name, description, requirements, subcommands_list])

            table = tabulate(table_data, headers=['Command', 'Description', 'Requirements', 'Subcommands'], tablefmt='grid')
            output += table + "\n"

        return output


    def deactivate_command(self, command_name):
        """Temporarily deactivate a command."""
        if command_name in self.commands:
            self.active_commands[command_name] = False
        else:
            print(f"Command {command_name} does not exist.")

    def activate_command(self, command_name):
        """Reactivate a previously deactivated command."""
        if command_name in self.commands:
            self.active_commands[command_name] = True
        else:
            print(f"Command {command_name} does not exist.")
