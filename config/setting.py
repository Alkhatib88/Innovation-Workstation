import json
import asyncio

class EnhancedSettings:
    def __init__(self, app):
        self.app = app
        self.default_settings = {}
        self.user_settings = {}
        self.dynamic_settings = {}  # To store settings added dynamically

    def setup(self, config_file, pickle_file):
        self.config_file = config_file
        self.pickle_file = pickle_file
        # Load basic configurations using ConfigSystem from the app
        self.app.config.load_config(self.config_file)

        # Load user-specific settings
        asyncio.run(self.load_user_settings())

    def register_dynamic_setting(self, key, default_value, description=""):
        """ Allows external scripts to add new settings dynamically """
        if key not in self.dynamic_settings:
            self.dynamic_settings[key] = {'value': default_value, 'description': description}
            self.app.logger.info(f"Dynamic setting registered: {key}")

    async def load_user_settings(self):
        try:
            encrypted_data = await self.app.pickle.load_data_async(self.pickle_file)
            if encrypted_data:
                decrypted_data = self.app.encrypt.decrypt(encrypted_data)
                self.user_settings = json.loads(decrypted_data)
            else:
                self.app.logger.warning("No user-specific settings found.")
        except Exception as e:
            self.app.logger.error(f"Error loading user settings: {e}")
            self.app.error_handler.handle_error(e)

    def list_settings(self):
        """ List all settings with their current values and descriptions """
        all_settings = {**self.default_settings, **self.user_settings, **self.dynamic_settings}
        for key, value in all_settings.items():
            print(f"{key}: {value}")

    def validate_setting(self, key, value):
        """ Validate the new value for a setting """
        # Add validation logic here
        return True

    def activate_setting(self, key):
        """ Activate a setting """
        if key in self.user_settings:
            self.user_settings[key]['active'] = True
            self.app.logger.info(f"Setting {key} activated")
        else:
            self.app.logger.error(f"Setting {key} not found")

    def deactivate_setting(self, key):
        """ Deactivate a setting """
        if key in self.user_settings:
            self.user_settings[key]['active'] = False
            self.app.logger.info(f"Setting {key} deactivated")
        else:
            self.app.logger.error(f"Setting {key} not found")

    def get_dynamic_setting(self, key):
        """ Retrieve the value of a dynamically added setting """
        return self.dynamic_settings.get(key, {}).get('value')

    def get_setting(self, key, default=None):
        """ Get a setting value with a fallback to default """
        return self.app.config.get_config(key) or self.user_settings.get(key, default)

    async def update_setting(self, key, value, user_specific=False):
        """ Update a setting with proper validation and error handling """
        if not self.validate_setting(key, value):
            self.app.logger.error(f"Invalid value for setting {key}")
            return

        try:
            if user_specific:
                self.user_settings[key] = value
                encrypted_data = self.app.encrypt.encrypt(json.dumps(self.user_settings))
                await self.app.pickle.save_data_async(encrypted_data, self.pickle_file)
            else:
                self.app.config.update_config(key, value)

            self.app.logger.info(f"Setting updated: {key}")
        except Exception as e:
            self.app.logger.error(f"Error updating setting: {key}")
            self.app.error_handler.handle_error(e)

    def set_multi_option_setting(self, key, options, default=None):
        """ Set a setting with multiple options """
        if key not in self.dynamic_settings:
            self.dynamic_settings[key] = {'options': options, 'value': default}
            self.app.logger.info(f"Multi-option setting {key} created with options {options}")
        else:
            self.app.logger.error(f"Setting {key} already exists")

# Example usage
# app = App()  # Assuming App is initialized and setup
# settings = EnhancedSettings(app, 'config.json', 'user_settings.pkl')
# asyncio.run(settings.update_setting('theme', 'dark', user_specific=True))
