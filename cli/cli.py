#cli.py
import sys

class Cli:
    def __init__(self, app):
        self.app = app

    def run(self):
        print(f"Welcome to Mansor's App.\nType '/list commands' to get a list of all the commands.\nType /help <command name> for information on how to run the command.\nType /quit to quit.")

        while True:
            try:
                completer = self.app.cli_tools.get_command_completer()
                user_input = self.app.cli_tools.session.prompt('> ', completer=completer)  # Corrected variable name
                self.app.input.parse_input(user_input)

            except KeyboardInterrupt:
                print("\nInterrupt received. Type 'exit' or 'quit' to terminate the application.")
                continue  # Or handle it as you see fit
            except Exception as e:
                print(f"An error occurred: {e}")
                break
                # Handle other exceptions if necessary

    def exit_application(self):
        # Perform any cleanup here
        print("Exiting the application. Goodbye!")
        sys.exit(0)  # Exit the application

    def setup(self):
        self.app.command.add_command(command_name="quit", function=self.exit_application, description="shuts down application", category="General Commands")
        self.app.command.add_command(command_name="list-commands", function=self.app.command.display_commands_table, description="displays commands", category="General Commands")
                # Register commands for dynamic settings management
        self.app.command.add_command(command_name="deactivate-command",function=self.app.command.deactivate_command,description="Temporarily deactivate a command.",category="Command Management",dependencies=['command_name'],arg_definitions={'command_name': {'help': 'Name of the command to deactivate', 'type': str}})
        self.app.command.add_command(command_name="activate-command",function=self.app.command.activate_command,description="Reactivate a previously deactivated command.",category="Command Management",dependencies=['command_name'],arg_definitions={'command_name': {'help': 'Name of the command to activate', 'type': str}})