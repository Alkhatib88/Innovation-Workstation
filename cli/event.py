# event_handler.py

class EventHandler:
    def __init__(self):
        self.events = {}

    def register_event(self, event_name, function, description=None, validator=None):
        """Register a new event."""
        self.events[event_name] = {
            'function': function,
            'description': description,
            'validator': validator
        }

    def trigger(self, event_name, *args, **kwargs):
        """Trigger a registered event."""
        if event_name not in self.events:
            raise ValueError(f"Event '{event_name}' not found.")
        
        if self.events[event_name]['validator']:
            if not self.events[event_name]['validator'](*args, **kwargs):
                raise ValueError(f"Validation failed for event '{event_name}'.")
            
        return self.events[event_name]['function'](*args, **kwargs)
