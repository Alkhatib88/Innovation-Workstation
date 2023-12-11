import os
import json
import yaml
import configparser
from threading import Lock

class ConfigSystem:
    def __init__(self, app):
        self.app = app
        self._config_caches = {}  # Stores configurations for multiple files
        self._locks = {}  # Separate locks for each configuration set
        self._observers = {}
        self._config_file_paths = {}  # Stores the file paths for each config

    def setup(self, config_name, config_file, file_format):
        """Setup a specific configuration file."""
        self._config_caches[config_name] = {}
        self._locks[config_name] = Lock()
        self._observers[config_name] = []
        self._config_file_paths[config_name] = config_file  # Store the file path
        self.load_or_initialize_config(config_name, config_file, file_format)

    def load_or_initialize_config(self, config_name, config_file, file_format):
        """Load configuration from a file or create a new file if it doesn't exist."""
        if os.path.exists(config_file):
            self.load_config(config_name, config_file, file_format)
        else:
            # Create a new file with empty configuration
            with open(config_file, 'w') as file:
                if file_format == 'json':
                    json.dump({}, file, indent=4)
                elif file_format == 'yaml':
                    yaml.dump({}, file)
                elif file_format == 'ini':
                    config = configparser.ConfigParser()
                    config.write(file)
            self.app.logger.info(f"New configuration file created for {config_name}.")

    def load_config(self, config_name, file_path, file_format):
        """Load configuration from a file."""
        with self._locks[config_name]:
            try:
                with open(file_path, 'r') as file:
                    if file_format == 'json':
                        self._config_caches[config_name] = json.load(file)
                    elif file_format == 'yaml':
                        self._config_caches[config_name] = yaml.safe_load(file)
                    elif file_format == 'ini':
                        config = configparser.ConfigParser()
                        config.read(file_path)
                        self._config_caches[config_name] = {section: dict(config.items(section)) for section in config.sections()}

                # Decrypt sensitive fields after loading
                config_cache = self._config_caches[config_name]
                for key in config_cache:
                    if self.is_sensitive(key):
                        encrypted_value = config_cache[key]
                        decrypted_value = self.app.encrypt.decrypt(encrypted_value)
                        config_cache[key] = decrypted_value

                self._notify_observers(config_name)
            except Exception as e:
                self.app.logger.error(f"Failed to load configuration for '{config_name}': {e}")


    def save_config(self, config_name, file_format='json'):
        """Saves the configuration of a specific set to a file."""
        config_cache = self._config_caches[config_name]
        file_path = self._config_file_paths[config_name]
        if not file_path:
            self.app.logger.error(f"No configuration file path set for '{config_name}'.")
            return

        try:
            with self._locks[config_name], open(file_path, 'w') as file:
                config_to_save = {key: (self.app.encrypt.encrypt(value) if self.is_sensitive(key) else value)
                                  for key, value in config_cache.items()}

                if file_format == 'json':
                    json.dump(config_to_save, file, indent=4)
                elif file_format == 'yaml':
                    yaml.dump(config_to_save, file)
                elif file_format == 'ini':
                    config = configparser.ConfigParser()
                    for section, values in config_to_save.items():
                        config[section] = values
                    config.write(file)

                self.app.logger.info(f"Configuration for '{config_name}' saved in {file_format} format.")
        except Exception as e:
            self.app.logger.error(f"Failed to save configuration for '{config_name}': {e}")


    def get_config(self, config_name, key, default=None):
        """Retrieves a configuration value for a given key from a specified config set."""
        return self._config_caches[config_name].get(key, default)

    def is_sensitive(self, key):
        # Implement logic to determine if the key is sensitive (requires encryption)
        return key in ['SENSITIVE_KEY1', 'SENSITIVE_KEY2']  # Example sensitive keys

    def register_observer(self, config_name, observer_func):
        """Registers an observer function for a specific configuration set."""
        self._observers[config_name].append(observer_func)

    def _notify_observers(self, config_name):
        """Notifies all observers of a specific configuration set about changes."""
        for observer in self._observers[config_name]:
            observer(self._config_caches[config_name])

    def update_config(self, config_name, key, value, file_format='json'):
        """Updates the configuration value for a given key in a specific configuration set."""
        lock = self._locks[config_name]
        with lock:
            config_cache = self._config_caches[config_name]
            if file_format == 'ini':
                config = configparser.ConfigParser()
                config.read_dict(config_cache)
                section, key = key.split('.')  # Assuming 'section.key' format
                if not config.has_section(section):
                    config.add_section(section)
                config.set(section, key, value)
                file_path = self._config_file_paths[config_name]
                with open(file_path, 'w') as file:
                    config.write(file)
            else:
                config_cache[key] = value
                self.save_config(config_name, file_format)



# Example observer function
def on_config_change(new_config):
    print("Configuration has changed:", new_config)

# Example usage
# app = YourAppInstance()  # Replace with your actual app instance
# config_system = ConfigSystem(app)
# config_system.setup('main', 'path/to/main_config.json')
# config_system.setup('module', 'path/to/module_config.json')
# config_system.register_observer('main', on_config_change)
