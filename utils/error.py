import logging

# Custom Exception Classes
class CustomError(Exception):
    def __init__(self, message, context=None):
        super().__init__(message)
        self.context = context

class DatabaseError(CustomError):
    pass

class ValidationError(CustomError):
    pass

class InputError(CustomError):
    pass

class DependencyError(CustomError):
    pass

class UnknownCommandError(CustomError):
    pass


class ErrorHandler:
    def __init__(self, app, logger=None):
        self.app = app
        self.logger = logger or logging.getLogger('ErrorHandler')
        self.handlers = {
            DatabaseError: self.handle_database_error,
            ValidationError: self._handle_validation_error,
            # Add new handlers for custom exceptions
            InputError: self._handle_input_error,
            DependencyError: self._handle_dependency_error
        }

    def handle_error(self, error):
        handler = self.handlers.get(type(error), self._default_error_handler)
        handler(error)

    def _log_error(self, error, level):
        message = str(error)
        context = getattr(error, 'context', None)
        if context:
            message += f' | Context: {context}'
        getattr(self.logger, level)(message)

    def handle_database_error(self, error):
        """Handles database-specific errors."""
        self.app.logger.error(f"Database Error: {error}")
        self._notify_developers(error)

    def _handle_validation_error(self, error):
        self._log_error(error, 'warning')

    def _default_error_handler(self, error):
        self._log_error(error, 'error')
        self._notify_developers(error)

    def _notify_developers(self, error):
        # Placeholder method for notifying developers.
        # This can be enhanced to send emails, messages, etc.
        pass

    def add_error_handler(self, error_type, handler):
        self.handlers[error_type] = handler

    def _handle_input_error(self, error):
        # Custom handling for input errors
        self._log_error(error, 'warning')

    def _handle_dependency_error(self, error):
        # Custom handling for dependency errors
        self._log_error(error, 'error')

# Usage example:
# logger = logging.getLogger('MyApp')
# error_handler = ErrorHandler(app, logger)
# try:
#     # Some operation that might raise an error
#     pass
# except Exception as e:
#     error_handler.handle_error(e)
