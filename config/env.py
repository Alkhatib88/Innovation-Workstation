# env.py
import os
from threading import Lock

class EnhancedEnvironmentVariables:
    """
    A comprehensive class to manage environment variables similar to the config setup,
    specifically designed for handling .env files.
    """

    def __init__(self, app):
        self.app = app
        self._env_vars_caches = {}  # Stores environment variables for multiple sources
        self._locks = {}  # Separate locks for each environment variable set
        self._env_sources_paths = {}  # Stores the paths/sources for each environment variable set

    def setup(self, env_name, env_source, source_type='file'):
        """Setup a specific environment variable source."""
        self._env_vars_caches[env_name] = {}
        self._locks[env_name] = Lock()
        self._env_sources_paths[env_name] = env_source  # Store the source path
        self.load_or_initialize_env(env_name, env_source, source_type)

    def load_or_initialize_env(self, env_name, env_source, source_type):
        """Load environment variables from a source or create a new source if it doesn't exist."""
        if source_type == 'file':
            if not os.path.exists(env_source):
                # Create a new .env file with empty environment variables
                open(env_source, 'w').close()
                self.app.logger.info(f"New environment variable source created for {env_name} at {env_source}.")
            self.load_env(env_name, env_source)
        elif source_type == 'system':
            self.load_from_system(env_name)

    def load_env(self, env_name, file_path):
        """Load environment variables from an .env file."""
        with self._locks[env_name]:
            try:
                with open(file_path, 'r') as file:
                    for line in file:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            self._env_vars_caches[env_name][key] = value
            except Exception as e:
                self.app.logger.error(f"Failed to load environment variables for '{env_name}': {e}")

    def load_from_system(self, env_name):
        """Load environment variables from the system environment."""
        with self._locks[env_name]:
            self._env_vars_caches[env_name] = {key: value for key, value in os.environ.items()}

    def get_env_variable(self, env_name, key, default=None):
        """Retrieves an environment variable value for a given key from a specified set."""
        with self._locks[env_name]:
            return self._env_vars_caches[env_name].get(key, default)

    def update_env_variable(self, env_name, key, value):
        """Updates the environment variable value for a given key in a specific set."""
        with self._locks[env_name]:
            self._env_vars_caches[env_name][key] = value
            self.app.logger.info(f"Environment variable '{key}' updated for '{env_name}'.")

    def save_env(self, env_name):
        """Saves the environment variables of a specific set to an .env file."""
        env_cache = self._env_vars_caches[env_name]
        file_path = self._env_sources_paths[env_name]
        if not file_path:
            self.app.logger.error(f"No environment variable source path set for '{env_name}'.")
            return

        with self._locks[env_name], open(file_path, 'w') as file:
            for key, value in env_cache.items():
                file.write(f"{key}={value}\n")
            self.app.logger.info(f"Environment variables for '{env_name}' saved.")

    def encrypt_value(self, value):
        """Encrypts a value."""
        return self.app.encrypt.encrypt(value)

    def decrypt_value(self, value):
        """Decrypts a value."""
        return self.app.encrypt.decrypt(value)

    def add_to_env_file(self, env_name, key, value):
        """Adds or updates a variable in an .env file."""
        self.update_env_variable(env_name, key, value)
        self.save_env(env_name)

# Example usage:
# env_vars = EnhancedEnvironmentVariables(app_instance)
# env_vars.setup('main', 'path/to/main_env.env', 'file')
# encrypted_db_password = env_vars.encrypt_value('my_secret_password')
# env_vars.update_env_variable('main', 'DB_PASSWORD', encrypted_db_password)
# db_password = env_vars.get_env_variable('main', 'DB_PASSWORD')
