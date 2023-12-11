#input_sys
import shlex

class Input_sys:

    def __init__(self, app, CustomError, DependencyError, UnknownCommandError):
        self.app = app
        self.CustomError = CustomError
        self.DependencyError = DependencyError
        self.UnknownCommandError = UnknownCommandError

    def parse_input(self, user_input):
        self.app.logger.info(f"User input: {user_input}")

        if user_input.startswith('/'):
            self.handle_command_input(user_input[1:])
        elif user_input.startswith('@'):
            self.handle_special_action(user_input[1:])
        elif user_input.startswith('#'):
            self.handle_another_action(user_input[1:])
        else:
            self.handle_text_message(user_input)

    def handle_command_input(self, command):
        try:
            # Splitting the command and arguments, handling quoted strings
            parsed_command = shlex.split(command)
            command_name = parsed_command[0]
            args = parsed_command[1:]

            # Handling command with possible subcommands and arguments
            if command_name not in self.app.command.commands:
                raise self.UnknownCommandError(f"Unknown command: {command_name}")

            # Check if it's a command with subcommands
            subcommands = self.app.cli_tools.get_subcommands(command_name)
            if subcommands and args:
                subcommand = args[0]
                if subcommand in subcommands:
                    # Handling subcommand and its arguments
                    sub_args = args[1:]
                    result = self.app.command.execute(command_name, subcommand, *sub_args)
                else:
                    raise self.UnknownCommandError(f"Unknown subcommand: {subcommand} for command {command_name}")
            else:
                # Handling command without subcommands
                result = self.app.command.execute(command_name, *args)

            print(result)
        except self.CustomError as custom_error:
            self.app.error_handler.handle_error(custom_error)
        except Exception as e:
            self.app.error_handler.handle_error(e)

    def handle_special_action(self, action):
        # Logic for special actions starting with '@'
        pass

    def handle_another_action(self, action):
        # Actual logic for actions starting with '#'
        # Example:
        print(f"Handling another action: {action}")

    def handle_text_message(self, message):
        # Logic to handle regular text messages
        pass
