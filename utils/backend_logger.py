import threading
from datetime import datetime, timezone

# Shared state for logs
_backend_events = []
_events_lock = threading.Lock()

def log_event(source: str, message: str, level: str = "info"):
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "level": level,
        "message": message,
    }
    with _events_lock:
        _backend_events.append(event)
        if len(_backend_events) > 200:
            _backend_events.pop(0)
    
    # Also print to console
    print(f"{event['timestamp_utc']} [{level.upper()}] {source}: {message}")

def get_events():
    with _events_lock:
        return list(_backend_events)
