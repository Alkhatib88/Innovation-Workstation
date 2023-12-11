import os
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler, SMTPHandler
import logging.config
from json_logging import *
from pythonjsonlogger import jsonlogger
from contextlib import contextmanager
from datetime import datetime

class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, app):
        self.app = app
        self.logger = None
        self.formatter = None
        self.handlers = []

    @contextmanager
    def temporary_log_level(self, level):
        """Temporarily sets the logging level."""
        original_level = self.logger.level
        self.set_level(level)
        try:
            yield
        finally:
            self.set_level(original_level)

    def enable_json_logging(self):
        """Enables JSON formatted logging."""
        json_logging.ENABLE_JSON_LOGGING = True
        json_logging.init_non_web()

    def _archive_old_log(self, log_file):
        """If log file exists, rename it with a timestamp."""
        if os.path.exists(log_file):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            os.rename(log_file, f"{log_file}_{timestamp}")

    def setup(self, name=None, level=None, log_file=None, log_console=False, log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
        """Set up the logger with specified parameters."""
        # Archive old log file if it exists

        if log_file:
            self._archive_old_log(log_file)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.formatter = logging.Formatter(log_format)

        if log_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(self.formatter)
            self.add_handler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(self.formatter)
            self.add_handler(file_handler)

        if not self.handlers:
            self.add_console_handler()  # Add a default console handler

        for handler in self.handlers:
            self.logger.addHandler(handler)

    def debug(self, message, *args, lazy=False, **kwargs):
        if lazy and self.logger.isEnabledFor(logging.DEBUG):
            message = message()
        self.logger.debug(message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        self.logger.info(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs)


    def critical(self, message, *args, **kwargs):
        self.logger.critical(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)

    def set_level(self, level):
        self.logger.setLevel(level)
        for handler in self.handlers:
            handler.setLevel(level)

    def set_format(self, format_string):
        self.formatter = logging.Formatter(format_string)
        for handler in self.handlers:
            handler.setFormatter(self.formatter)

    def add_handler(self, handler, format_string=None):
        if format_string:
            handler.setFormatter(logging.Formatter(format_string))
        else:
            handler.setFormatter(self.formatter)
        for existing_handler in self.handlers:
            if type(existing_handler) == type(handler):
                return  # Handler already exists, exit method
        self.logger.addHandler(handler)
        self.handlers.append(handler)

    def add_file_handler(self, filename, level=None, format_string=None):
            handler = logging.FileHandler(filename)
            if level:
                handler.setLevel(level)
            if format_string:
                formatter = logging.Formatter(format_string)
                handler.setFormatter(formatter)
            self.add_handler(handler)

    def add_console_handler(self, level=None, format_string=None):
        handler = logging.StreamHandler()
        if level:
            handler.setLevel(level)
        if format_string:
            formatter = logging.Formatter(format_string)
            handler.setFormatter(formatter)
        self.add_handler(handler)

    def add_rotating_file_handler(self, filename, max_bytes=0, backup_count=0, encoding=None, delay=False):
        handler = RotatingFileHandler(filename, maxBytes=max_bytes, backupCount=backup_count, encoding=encoding, delay=delay)
        handler.setFormatter(self.formatter)
        self.add_handler(handler)

    def add_timed_rotating_file_handler(self, filename, when='h', interval=1, backup_count=0, encoding=None, delay=False, utc=False, at_time=None):
        handler = TimedRotatingFileHandler(filename, when=when, interval=interval, backupCount=backup_count, encoding=encoding, delay=delay, utc=utc, atTime=at_time)
        handler.setFormatter(self.formatter)
        self.add_handler(handler)

    def add_smtp_handler(self, mailhost, fromaddr, toaddrs, subject):
        smtp_handler = SMTPHandler(mailhost, fromaddr, toaddrs, subject)
        smtp_handler.setFormatter(self.formatter)
        self.add_handler(smtp_handler)

    def remove_handler(self, handler):
        self.logger.removeHandler(handler)
        self.handlers.remove(handler)

    def configure_from_file(self, config_filepath):
        logging.config.fileConfig(config_filepath)

    def add_filter(self, filter, handler=None):
        if handler:
            handler.addFilter(filter)
        else:
            self.logger.addFilter(filter)

    def add_custom_handler(self, handler):
        self.add_handler(handler)

    @staticmethod
    def log_method_call(method):
        def wrapper(self, *args, **kwargs):
            self.info(f"Calling method: {method.__name__}")
            result = method(self, *args, **kwargs)
            self.info(f"Method {method.__name__} completed")
            return result
        return wrapper